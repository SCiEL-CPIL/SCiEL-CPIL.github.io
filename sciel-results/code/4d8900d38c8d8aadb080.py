import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Precompute all blocks and their triple memberships
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            tris.append(triple_to_idx[tri])
        block_triples.append(tris)
    
    # Phase 1: Build a candidate pool
    # Use greedy to get a good initial solution, then expand the candidate pool
    def greedy_initial():
        uncovered = set(range(num_triples))
        selected = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i, tris in enumerate(block_triples):
                gain = len(set(tris) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            for tri in block_triples[best_idx]:
                uncovered.discard(tri)
        
        return selected
    
    # Get initial greedy solution
    greedy_sol = greedy_initial()
    
    # Build candidate pool: greedy solution + random blocks
    candidate_pool = set(greedy_sol)
    
    # Add random blocks to the pool
    num_random = min(2000, num_blocks)
    random_blocks = random.sample(range(num_blocks), num_random)
    candidate_pool.update(random_blocks)
    
    # Also add blocks that cover many uncovered triples from greedy
    uncovered_after_greedy = set(range(num_triples))
    for idx in greedy_sol:
        for tri in block_triples[idx]:
            uncovered_after_greedy.discard(tri)
    
    if uncovered_after_greedy:
        # Find blocks that cover many of the uncovered triples
        gains = []
        for i in range(num_blocks):
            if i in candidate_pool:
                continue
            gain = len(set(block_triples[i]) & uncovered_after_greedy)
            if gain > 0:
                gains.append((gain, i))
        gains.sort(reverse=True)
        for gain, idx in gains[:500]:
            candidate_pool.add(idx)
    
    candidate_list = list(candidate_pool)
    num_candidates = len(candidate_list)
    candidate_to_global = {c: i for i, c in enumerate(candidate_list)}
    
    # Phase 2: Solve ILP on candidate pool
    # Variables: x_j = 1 if block j is selected, 0 otherwise
    # Minimize sum(x_j)
    # Subject to: for each triple i, sum_{j: i in block_j} x_j >= 1
    
    # Build constraint matrix
    # A[i, j] = 1 if triple i is covered by candidate block j
    A_rows = []
    A_cols = []
    A_vals = []
    
    for j, global_idx in enumerate(candidate_list):
        for tri_idx in block_triples[global_idx]:
            A_rows.append(tri_idx)
            A_cols.append(j)
            A_vals.append(1)
    
    A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(num_triples, num_candidates))
    
    # Objective: minimize sum of x
    c = np.ones(num_candidates)
    
    # Constraints: A @ x >= 1 for all triples
    # milp requires: lb <= A @ x <= ub
    # So: 1 <= A @ x <= inf
    constraints = LinearConstraint(A, lb=np.ones(num_triples), ub=np.inf)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
    
    # Integrate variables
    integrality = np.ones(num_candidates)
    
    # Solve
    try:
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result.status == 0:  # Optimal
            x = result.x
            ilp_sol = [candidate_list[j] for j in range(num_candidates) if x[j] > 0.5]
        else:
            # Fallback to greedy
            ilp_sol = greedy_sol
    except Exception:
        ilp_sol = greedy_sol
    
    # Phase 3: Local search to improve
    def local_search(sol, time_limit=10):
        selected = set(sol)
        
        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time.time() < start_time + time_limit:
            changed = False
            counts = [0] * num_triples
            for idx in selected:
                for tri in block_triples[idx]:
                    counts[tri] += 1
            
            to_remove = []
            for idx in list(selected):
                redundant = True
                for tri in block_triples[idx]:
                    if counts[tri] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    selected.remove(idx)
                    for tri in block_triples[idx]:
                        counts[tri] -= 1
                changed = True
        
        # Phase 2: Try swaps
        if time.time() < start_time + time_limit - 2:
            for _ in range(5):
                if time.time() > start_time + time_limit - 2:
                    break
                
                counts = [0] * num_triples
                for idx in selected:
                    for tri in block_triples[idx]:
                        counts[tri] += 1
                
                # Try removing each block and see if we can replace it
                for idx in list(selected):
                    if time.time() > start_time + time_limit - 2:
                        break
                    
                    selected.remove(idx)
                    for tri in block_triples[idx]:
                        counts[tri] -= 1
                    
                    uncovered = [tri for tri in block_triples[idx] if counts[tri] == 0]
                    if not uncovered:
                        # We can remove this block entirely
                        continue
                    
                    # Try to find a replacement block
                    found = False
                    for j in candidate_list:
                        if j in selected:
                            continue
                        covers_all = True
                        for tri in uncovered:
                            if tri not in block_triples[j]:
                                covers_all = False
                                break
                        if covers_all:
                            selected.add(j)
                            for tri in block_triples[j]:
                                counts[tri] += 1
                            found = True
                            break
                    
                    if not found:
                        selected.add(idx)
                        for tri in block_triples[idx]:
                            counts[tri] += 1
        
        return list(selected)
    
    # Apply local search
    best_sol = local_search(ilp_sol, time_limit=15)
    
    # Phase 4: Additional iterations if time permits
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            # Random restart: perturb the solution
            perturbed = list(best_sol)
            if perturbed:
                for _ in range(3):
                    if perturbed:
                        perturbed[random.randint(0, len(perturbed)-1)] = random.choice(candidate_list)
            improved = local_search(perturbed, time_limit=5)
            if len(improved) < len(best_sol):
                best_sol = improved
    
    # Final local search
    if time.time() < deadline - 5:
        best_sol = local_search(best_sol, time_limit=5)
    
    # Convert to final blocks
    final_blocks = [all_blocks[idx] for idx in best_sol]
    
    # Ensure validity
    covered = set()
    for block in final_blocks:
        for tri in itertools.combinations(block, t):
            covered.add(tri)
    
    # If invalid, fall back to greedy
    if len(covered) < num_triples:
        final_blocks = [all_blocks[idx] for idx in greedy_sol]
    
    return final_blocks
