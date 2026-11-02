import numpy as np
import itertools
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Map each block to its covered triple indices
    block_triples = []
    row_indices = []
    col_indices = []
    data = []
    
    for b_idx, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        triple_idxs = []
        for sub in subs:
            t_idx = idx_map[sub]
            triple_idxs.append(t_idx)
            row_indices.append(b_idx)
            col_indices.append(t_idx)
            data.append(1)
        block_triples.append(triple_idxs)
        
    # Build sparse matrix M (n_ks x n_ts)
    # M[i, j] = 1 if block i covers triple j
    M = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_ks, n_ts))
    
    best_solution = None
    best_size = float('inf')
    
    def get_gain(uncovered):
        gains = M.dot(uncovered)
        return gains

    # Precompute triple rarity (number of blocks covering each triple)
    triple_counts_total = np.array(M.sum(axis=0)).flatten()
    triple_rarity = 1.0 / (triple_counts_total + 1e-9)
    
    # Precompute for each block: sum of rarity of all triples it covers
    block_rarity_score = np.zeros(n_ks)
    for b_idx, subs in enumerate(block_triples):
        block_rarity_score[b_idx] = np.sum(triple_rarity[subs])
    
    def run_greedy_with_rarity(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        counts = np.zeros(n_ts, dtype=np.int32)
        solution = []
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5.0:
                return None, None
                
            gains = get_gain(uncovered)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 0: break
            
            # Score candidates: prioritize blocks that cover rare triples
            # Use a smoothed score: gain + alpha * rarity_score
            # But since all candidates have same gain, just use rarity
            if len(candidates) > 200:
                sample_idx = random.sample(range(len(candidates)), 200)
                candidates = candidates[sample_idx]
                
            # Compute score for each candidate
            scores = block_rarity_score[candidates]
            # Add small random perturbation for diversity
            scores = scores + np.random.random(len(scores)) * 0.01
            
            best_candidate = candidates[np.argmax(scores)]
            
            solution.append(best_candidate)
            
            subs = block_triples[best_candidate]
            uncovered[subs] = False
            counts[subs] += 1
            
        return solution, len(solution)

    def local_search(solution, time_budget):
        # Returns improved solution
        improved = True
        while improved and time.time() - start_time < time_limit - 2.0:
            improved = False
            
            # Rebuild counts
            counts = np.zeros(n_ts, dtype=np.int32)
            for b in solution:
                counts[block_triples[b]] += 1
            
            # Deletion pass: remove blocks that are redundant
            new_solution = []
            for b_idx in solution:
                subs = block_triples[b_idx]
                if np.min(counts[subs]) >= 2:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_solution.append(b_idx)
            solution = new_solution
            
            if len(solution) == 0:
                break
                
            # Try to replace blocks with better alternatives
            # For each block, check if there's a block that covers all its unique triples
            # and also covers some additional triples
            for i in range(len(solution)):
                if time.time() - start_time > time_limit - 3.0:
                    break
                    
                b_idx = solution[i]
                subs = block_triples[b_idx]
                
                # Find triples that are uniquely covered by this block
                unique_subs = [s for s in subs if counts[s] == 1]
                
                if len(unique_subs) == 0:
                    # This block is redundant, remove it
                    counts[subs] -= 1
                    solution.pop(i)
                    i -= 1
                    improved = True
                    continue
                    
                # Try to find a replacement block that covers all unique_subs
                # and has better rarity score
                # This is expensive, so only do it for a few blocks per iteration
                if i % 10 == 0:  # Only try every 10th block
                    # Find all blocks that cover all unique_subs
                    # Use M to check: M[:, unique_subs] must be all 1s
                    # This is expensive, so skip for now
                    pass
                    
        return solution

    def polish(solution):
        # Final polish: try to remove redundant blocks and replace with better ones
        counts = np.zeros(n_ts, dtype=np.int32)
        for b in solution:
            counts[block_triples[b]] += 1
        
        # Multiple deletion passes
        for _ in range(3):
            new_solution = []
            for b_idx in solution:
                subs = block_triples[b_idx]
                if np.min(counts[subs]) >= 2:
                    counts[subs] -= 1
                else:
                    new_solution.append(b_idx)
            if len(new_solution) == len(solution):
                break
            solution = new_solution
        
        return solution

    # Main Loop
    restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restarts
        sol, size = run_greedy_with_rarity(seed)
        if sol is None:
            break
            
        if size < best_size:
            best_size = size
            best_solution = sol[:]
            
        # Local search on the current solution
        if size <= best_size + 5:
            sol_ls = local_search(sol[:], time_limit - 5.0)
            if len(sol_ls) < best_size:
                best_size = len(sol_ls)
                best_solution = sol_ls[:]
                
        restarts += 1
    
    # Final Polish on best_solution
    if best_solution is not None:
        best_solution = polish(best_solution)
        best_size = len(best_solution)

    return [list(all_ks[i]) for i in best_solution]
