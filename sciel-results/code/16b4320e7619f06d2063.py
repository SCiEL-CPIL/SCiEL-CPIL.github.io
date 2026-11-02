import itertools
import random
import time
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each point, which triples contain it (for fast mandatory point logic)
    point_in_triples = defaultdict(list)
    for i, triple in enumerate(all_triples):
        for p in triple:
            point_in_triples[p].append(i)
    
    # Cache for block triple indices
    block_triple_cache = {}
    
    def get_block_triples(block):
        key = tuple(sorted(block))
        if key in block_triple_cache:
            return block_triple_cache[key]
        result = [triple_to_idx[triple] for triple in itertools.combinations(key, t)]
        block_triple_cache[key] = result
        return result
    
    def get_block_triples_array(block):
        """Return numpy array of triple indices for a block."""
        tris = get_block_triples(block)
        return np.array(tris, dtype=np.int32)
    
    def vectorized_greedy(random_seed=None):
        """Fast vectorized greedy set cover using precomputed block-triple matrix."""
        if random_seed is not None:
            random.seed(random_seed)
        
        # Precompute: for each possible block, which triples it covers
        # Use a dictionary mapping block tuple -> list of triple indices
        # Instead of storing all C(24,7) blocks, we generate candidates on the fly
        # but use numpy for fast gain computation
        
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered_mask = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while uncovered_mask.any():
            if time.time() > deadline:
                break
                
            # Pick an uncovered triple (randomly)
            uncovered_indices = np.where(uncovered_mask)[0]
            target_triple_idx = random.choice(uncovered_indices)
            target_triple = all_triples[target_triple_idx]
            target_set = set(target_triple)
            
            remaining_points = [p for p in points if p not in target_set]
            
            best_block = None
            best_gain = -1
            best_gain_array = None
            
            # Generate candidate blocks containing the target triple
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(target_triple) + list(combo)))
                
                # Calculate gain: how many currently-uncovered triples does this block cover?
                b_triples = get_block_triples(block)
                gain = int(uncovered_mask[b_triples].sum())
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_gain_array = np.array(b_triples, dtype=np.int32)
                    if best_gain == 35:  # Maximum possible gain
                        break
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update coverage
            for idx in best_gain_array:
                if uncovered_mask[idx]:
                    uncovered_mask[idx] = False
                    coverage_count[idx] += 1
                else:
                    coverage_count[idx] += 1
                    
        return blocks, coverage_count

    def solve_residual_ilp(blocks, coverage_count, time_budget=10.0):
        """Solve the residual set-cover problem using MILP.
        
        After the greedy phase, some triples may still be uncovered.
        We formulate a small ILP to find the minimum additional blocks needed.
        Also tries to replace existing blocks with fewer total blocks.
        """
        if time.time() > deadline - time_budget:
            return blocks
        
        # Identify uncovered triples
        uncovered_indices = [i for i in range(num_triples) if coverage_count[i] == 0]
        
        if not uncovered_indices:
            return blocks
        
        # For each uncovered triple, find all possible blocks that cover it
        # and are NOT already in our solution
        existing_block_set = set(tuple(sorted(b)) for b in blocks)
        
        # Build: for each uncovered triple, list of candidate blocks (as frozensets)
        # We'll use a compact representation: assign an index to each unique candidate block
        triple_to_candidates = {}
        all_candidate_blocks = {}  # block tuple -> index
        
        for t_idx in uncovered_indices:
            triple = all_triples[t_idx]
            triple_set = set(triple)
            candidates = []
            
            for combo in itertools.combinations(
                [p for p in points if p not in triple_set], k - t
            ):
                block = tuple(sorted(list(triple) + list(combo)))
                if block not in existing_block_set:
                    candidates.append(block)
                    if block not in all_candidate_blocks:
                        all_candidate_blocks[block] = len(all_candidate_blocks)
            
            triple_to_candidates[t_idx] = candidates
        
        if not all_candidate_blocks:
            return blocks
        
        num_candidates = len(all_candidate_blocks)
        
        # If too many candidates, fall back to greedy for the residual
        if num_candidates > 5000:
            # Use greedy to cover remaining uncovered triples
            residual_blocks = []
            still_uncovered = list(uncovered_indices)
            
            while still_uncovered and time.time() < deadline - 2:
                target_idx = still_uncovered[0]
                target_triple = all_triples[target_idx]
                target_set = set(target_triple)
                
                best_block = None
                best_gain = -1
                
                for block in triple_to_candidates[target_idx]:
                    b_triples = get_block_triples(block)
                    gain = sum(1 for idx in b_triples if idx in set(still_uncovered))
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                
                if best_block is None:
                    break
                
                residual_blocks.append(list(best_block))
                
                # Update still_uncovered
                b_triples = set(get_block_triples(best_block))
                still_uncovered = [idx for idx in still_uncovered if idx not in b_triples]
            
            return blocks + residual_blocks
        
        # Build the ILP constraint matrix
        # Variables: x_j = 1 if we use candidate block j
        # Constraint: for each uncovered triple i, sum of x_j for blocks covering i >= 1
        # Objective: minimize sum x_j
        
        # Build sparse constraint matrix
        # Row i (for uncovered triple i), Column j (for candidate block j)
        # A[i,j] = 1 if block j covers triple i
        
        # We'll build it as a dense matrix since num_candidates is manageable
        A = np.zeros((len(uncovered_indices), num_candidates), dtype=np.float64)
        
        for t_idx in uncovered_indices:
            row = uncovered_indices.index(t_idx)
            for block in triple_to_candidates[t_idx]:
                col = all_candidate_blocks[block]
                A[row, col] = 1.0
        
        # MILP formulation
        c = np.ones(num_candidates)  # Minimize number of blocks
        
        # Constraints: A @ x >= 1 for each row
        constraint = LinearConstraint(A, lb=np.ones(len(uncovered_indices)), ub=np.inf)
        
        # Variables are binary
        integrality = np.ones(num_candidates)
        
        bounds = Bounds(lb=0, ub=1)
        
        try:
            result = milp(
                c=c,
                constraints=constraint,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_budget - 2}
            )
            
            if result.status == 0:  # Optimal
                new_blocks = []
                for j in range(num_candidates):
                    if result.x[j] > 0.5:
                        block = list(all_candidate_blocks.keys())[j]
                        new_blocks.append(list(block))
                
                return blocks + new_blocks
            elif result.status in [1, 2]:  # Locally optimal or feasible
                # Use the feasible solution found
                new_blocks = []
                for j in range(num_candidates):
                    if result.x[j] is not None and result.x[j] > 0.5:
                        block = list(all_candidate_blocks.keys())[j]
                        new_blocks.append(list(block))
                
                if new_blocks:
                    return blocks + new_blocks
                return blocks
            else:
                return blocks
                
        except Exception:
            return blocks

    def local_search(blocks, coverage_count=None):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        # Rebuild coverage from scratch to be safe and consistent
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            for t_idx in get_block_triples(b):
                coverage_count[t_idx] += 1
                
        # Precompute reciprocal coverage for faster density scoring
        inv_cov = 1.0 / (coverage_count + 1)  # 1/(cov+1) for added triples
        inv_cov_current = 1.0 / np.maximum(coverage_count, 1)  # 1/cov for removed triples
        
        def update_inv_cov():
            nonlocal inv_cov, inv_cov_current
            inv_cov = 1.0 / (coverage_count + 1)
            inv_cov_current = 1.0 / np.maximum(coverage_count, 1)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Deletion - remove blocks where all triples are covered >= 2 times
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                if all(coverage_count[t_idx] >= 2 for t_idx in b_triples):
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    del block_tuples[i]
                    update_inv_cov()
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            if improved:
                continue
            
            # Pass 2: Exchange using density-based scoring
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                b_set = set(b_triples)
                unique_triples = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 1]
                
                if not unique_triples:
                    continue
                
                # Determine mandatory points for replacement
                mandatory_points = set()
                for t_idx in unique_triples:
                    mandatory_points.update(all_triples[t_idx])
                
                if len(mandatory_points) > k:
                    continue
                
                available_points = [p for p in points if p not in mandatory_points]
                num_extra_needed = k - len(mandatory_points)
                
                if num_extra_needed > len(available_points):
                    continue
                
                best_new_block = None
                best_density_score = 0.0  # Only accept if strictly positive
                
                # Sample candidates if search space is too large
                candidates = list(itertools.combinations(available_points, num_extra_needed))
                if len(candidates) > 3000:
                    random.shuffle(candidates)
                    candidates = candidates[:3000]
                
                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue
                    
                    new_triples = get_block_triples(new_block)
                    new_set = set(new_triples)
                    
                    # Check that all unique triples are still covered
                    if not all(t_idx in new_set for t_idx in unique_triples):
                        continue
                    
                    removed = b_set - new_set
                    added = new_set - b_set
                    
                    # Fast density calculation using precomputed arrays
                    density_delta = 0.0
                    for t_idx in removed:
                        density_delta -= inv_cov_current[t_idx]
                    for t_idx in added:
                        density_delta += inv_cov[t_idx]
                    
                    if density_delta > best_density_score:
                        best_density_score = density_delta
                        best_new_block = new_block
                
                if best_new_block is not None:
                    # Perform swap
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    for t_idx in get_block_triples(best_new_block):
                        coverage_count[t_idx] += 1
                    block_tuples[i] = best_new_block
                    update_inv_cov()
                    improved = True
                    break
                    
        # Final polish: more aggressive multi-round optimization
        if time.time() < deadline - 5:
            for polish_round in range(5):
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # Deletion pass
                for i in range(len(block_tuples)):
                    if time.time() > deadline - 2:
                        break
                    b = block_tuples[i]
                    b_triples = get_block_triples(b)
                    if all(coverage_count[t_idx] >= 2 for t_idx in b_triples):
                        for t_idx in b_triples:
                            coverage_count[t_idx] -= 1
                        del block_tuples[i]
                        update_inv_cov()
                        improved = True
                        break
                
                if not improved:
                    # Exchange pass
                    for i in range(len(block_tuples)):
                        if time.time() > deadline - 2:
                            break
                        b = block_tuples[i]
                        b_triples = get_block_triples(b)
                        b_set = set(b_triples)
                        unique_triples = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 1]
                        
                        if not unique_triples:
                            continue
                        
                        mandatory_points = set()
                        for t_idx in unique_triples:
                            mandatory_points.update(all_triples[t_idx])
                        
                        if len(mandatory_points) > k:
                            continue
                        
                        available_points = [p for p in points if p not in mandatory_points]
                        num_extra_needed = k - len(mandatory_points)
                        
                        if num_extra_needed > len(available_points):
                            continue
                        
                        best_new_block = None
                        best_density_score = 0.0
                        
                        candidates = list(itertools.combinations(available_points, num_extra_needed))
                        if len(candidates) > 3000:
                            random.shuffle(candidates)
                            candidates = candidates[:3000]
                        
                        for combo in candidates:
                            new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                            if new_block == b:
                                continue
                            
                            new_triples = get_block_triples(new_block)
                            new_set = set(new_triples)
                            
                            if not all(t_idx in new_set for t_idx in unique_triples):
                                continue
                            
                            removed = b_set - new_set
                            added = new_set - b_set
                            
                            density_delta = 0.0
                            for t_idx in removed:
                                density_delta -= inv_cov_current[t_idx]
                            for t_idx in added:
                                density_delta += inv_cov[t_idx]
                            
                            if density_delta > best_density_score:
                                best_density_score = density_delta
                                best_new_block = new_block
                        
                        if best_new_block is not None:
                            for t_idx in b_triples:
                                coverage_count[t_idx] -= 1
                            for t_idx in get_block_triples(best_new_block):
                                coverage_count[t_idx] += 1
                            block_tuples[i] = best_new_block
                            update_inv_cov()
                            improved = True
                            break
        
        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Fast vectorized greedy with multiple restarts
    min_restarts = 3
    max_restarts = 15
    
    for i in range(min_restarts):
        if time.time() > deadline - 30:
            break
            
        blocks, cov = vectorized_greedy(random_seed=i)
        
        # Use ILP to solve the residual
        blocks = solve_residual_ilp(blocks, cov, time_budget=min(15.0, deadline - time.time() - 5))
        
        # Local search refinement
        blocks = local_search(blocks, None)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 70:
            break
    
    # Phase 2: More restarts with different seeds if time permits
    if best_blocks is not None and time.time() < deadline - 20:
        for i in range(min_restarts, max_restarts):
            if time.time() > deadline - 10:
                break
                
            blocks, cov = vectorized_greedy(random_seed=i + 100)
            
            # Use ILP to solve the residual
            blocks = solve_residual_ilp(blocks, cov, time_budget=min(10.0, deadline - time.time() - 5))
            
            # Local search refinement
            blocks = local_search(blocks, None)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
            if best_count <= 70:
                break
    
    # Phase 3: Final polish - try ILP on the best solution's residual
    if best_blocks is not None and time.time() < deadline - 15:
        # Rebuild coverage for best solution
        cov = np.zeros(num_triples, dtype=np.int32)
        for b in best_blocks:
            for t_idx in get_block_triples(tuple(sorted(b))):
                cov[t_idx] += 1
        
        # Try ILP completion
        polished = solve_residual_ilp(best_blocks, cov, time_budget=min(10.0, deadline - time.time() - 3))
        
        # Also do local search
        polished = local_search(polished, None)
        
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished

    if best_blocks is None:
        return []
        
    return best_blocks
