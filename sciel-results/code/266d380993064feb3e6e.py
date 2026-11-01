import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map t-subset to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Helper to get triples for a block
    def block_triples(block):
        return list(itertools.combinations(block, t))

    # Phase 1: Greedy to get initial candidate set
    random.seed(42)
    uncovered = set(all_tsubs)
    initial_blocks = []
    
    while uncovered:
        start_triple = random.choice(list(uncovered))
        block = list(start_triple)
        random.shuffle(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        initial_blocks.append(block)
        for ts in block_triples(block):
            if ts in uncovered:
                uncovered.remove(ts)
                
        if time.time() - start_time > time_limit * 0.2:
            break
            
    # Ensure we have a valid covering if greedy stopped early
    if uncovered:
        while uncovered:
            start_triple = min(uncovered)
            block = list(start_triple)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            initial_blocks.append(block)
            for ts in block_triples(block):
                if ts in uncovered:
                    uncovered.remove(ts)
    
    # Phase 2: ILP Column Generation
    # Candidates are unique blocks
    candidates = set(tuple(b) for b in initial_blocks)
    
    # We will maintain a list of candidate blocks
    # and solve ILP on them.
    
    def solve_ilp(candidate_blocks):
        n_blocks = len(candidate_blocks)
        if n_blocks == 0:
            return None
            
        # A: n_tsubs x n_blocks matrix
        # A[i, j] = 1 if block j covers tsub i
        # We construct this sparsely or densely? n_tsubs=2024, n_blocks can be large.
        # Dense is 2024 * 10000 ~ 20M floats. ~160MB. Might be okay.
        # But constructing it in Python is slow.
        
        # Let's build the constraint matrix row by row (per tsub)
        # Or better, use the structure: 
        # For each block j, it covers C(k,t) tsubs.
        
        # SciPy milp expects:
        # c: objective coefficients (all 1)
        # constraints: LinearConstraint(A, lb, ub)
        # A is (n_constraints, n_vars)
        
        # We need A[i, j] = 1 if tsub i is in block j.
        
        # To speed up, we can use a dictionary of lists for columns?
        # No, milp wants a matrix.
        
        # Let's try to build A as a numpy array.
        # If n_blocks is too large, we might run out of memory or time.
        # We will limit candidates to ~5000-10000.
        
        A = np.zeros((n_tsubs, n_blocks), dtype=np.float64)
        
        for j, block in enumerate(candidate_blocks):
            for ts in block_triples(block):
                idx = tsub_to_idx[ts]
                A[idx, j] = 1.0
                
        c = np.ones(n_blocks)
        
        # Constraints: sum_j A[i, j] x_j >= 1
        lb = np.ones(n_tsubs)
        ub = np.full(n_tsubs, np.inf)
        
        # Bounds for x_j: 0 <= x_j <= 1
        res_bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
        
        # Integrality: x_j is integer (0 or 1)
        integrality = np.ones(n_blocks)
        
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb, ub), 
                       integrality=integrality, bounds=res_bounds)
            if res.success:
                # Extract solution
                sol = res.x
                blocks = []
                for j in range(n_blocks):
                    if sol[j] > 0.5:
                        blocks.append(list(candidate_blocks[j]))
                return blocks
            else:
                return None
        except Exception as e:
            return None

    best_solution = initial_blocks
    best_size = len(initial_blocks)
    
    # Column Generation Loop
    max_iter = 5
    for _ in range(max_iter):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Limit candidate size to keep ILP fast
        # We prioritize blocks from the current best solution and random additions
        # For now, just use the current candidate set.
        # If it's too big, we might need to prune.
        # Let's say we keep at most 2000 candidates for the ILP.
        # If we have more, we pick the best ones?
        # Actually, if we have a valid solution, we can use its blocks as the initial set for the next ILP.
        
        cand_list = list(candidates)
        if len(cand_list) > 3000:
            # Keep blocks from best_solution and some random ones
            best_set = set(tuple(b) for b in best_solution)
            cand_list = [b for b in cand_list if b in best_set]
            # Add random ones to allow exploration
            random.shuffle(candidates)
            for b in candidates:
                if b not in best_set and len(cand_list) < 3000:
                    cand_list.append(b)
                    
        sol = solve_ilp(cand_list)
        
        if sol:
            # Check if sol is valid (it should be, but just in case of solver issues)
            covered = set()
            for b in sol:
                for ts in block_triples(b):
                    covered.add(ts)
            
            if covered == set(all_tsubs):
                if len(sol) < best_size:
                    best_solution = sol
                    best_size = len(sol)
                    # Update candidates to focus on this solution
                    candidates = set(tuple(b) for b in sol)
                    # Add some random blocks to allow further improvement
                    for _ in range(100):
                        b = sorted(random.sample(range(v), k))
                        candidates.add(tuple(b))
            else:
                # Find uncovered triples
                uncovered = set(all_tsubs) - covered
                # Generate new blocks covering these uncovered triples
                for ts in list(uncovered):
                    # Create a block containing this triple
                    remaining = [x for x in range(v) if x not in ts]
                    random.shuffle(remaining)
                    new_block = sorted(list(ts) + remaining[:k-t])
                    candidates.add(tuple(new_block))
        else:
            # ILP failed or timed out, break
            break
            
    return best_solution
