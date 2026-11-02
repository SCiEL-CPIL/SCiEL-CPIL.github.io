import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers (as a set of indices)
    # Use a list of arrays for efficient access
    block_t_sets = [None] * n_ks
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        block_t_sets[i] = np.array([idx_map[sub] for sub in subs], dtype=np.int32)
    
    # Also build reverse mapping: for each t-subset, which blocks cover it
    # This is expensive for large v, but for v=24, k=7, t=3 it's manageable
    # n_ks = C(24,7) = 346104, n_ts = C(24,3) = 2024
    # Each t-subset is in C(21,4) = 5985 blocks... that's a lot
    # Let's build it as a list of lists
    ts_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for ts_idx in block_t_sets[i]:
            ts_to_blocks[ts_idx].append(i)
    
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """Try harder to remove blocks, including blocks where most subsets are over-covered."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            
            for b_idx in current_blocks:
                subs_in_block = block_t_sets[b_idx]
                # Count how many of this block's t-subsets are covered by other blocks
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    # Safe to remove
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
        return current_blocks
    
    def polish(block_set, max_rounds=50):
        """Try to replace blocks with better ones or remove redundant ones."""
        if not block_set:
            return block_set
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in block_set:
            coverage_count[block_t_sets[b_idx]] += 1
        
        current_blocks = list(block_set)
        block_set_set = set(current_blocks)
        
        for _ in range(max_rounds):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            improved = False
            new_blocks = []
            
            for b_idx in current_blocks:
                subs_in_block = block_t_sets[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            block_set_set = set(current_blocks)
            
            if not improved:
                break
        
        return current_blocks
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track how many times each t-subset has been covered (for weighted scoring)
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # Build a priority queue of uncovered t-subsets
        # For each uncovered t-subset, we know which blocks cover it
        # We want to pick a block that covers the most uncovered t-subsets
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Find uncovered t-subsets
            uncovered_mask = ~covered
            uncovered_ts = np.where(uncovered_mask)[0]
            
            if len(uncovered_ts) == 0:
                break
            
            # For each uncovered t-subset, count how many blocks in the current solution cover it
            # Actually, we want to find a new block that covers the most uncovered t-subsets
            
            # Strategy: sample some uncovered t-subsets and find blocks that cover them
            # Then pick the block that covers the most uncovered t-subsets overall
            
            sample_size = min(len(uncovered_ts), 20)
            sampled_ts = uncovered_ts[random.sample(range(len(uncovered_ts)), sample_size)]
            
            # Collect candidate blocks from these sampled t-subsets
            candidate_set = set()
            for ts_idx in sampled_ts:
                # Add some blocks that cover this t-subset
                blocks_covering = ts_to_blocks[ts_idx]
                if len(blocks_covering) > 10:
                    candidate_set.update(random.sample(blocks_covering, 10))
                else:
                    candidate_set.update(blocks_covering)
            
            if not candidate_set:
                # Fallback: random block
                best_candidate = random.randint(0, n_ks - 1)
            else:
                # Score each candidate by how many uncovered t-subsets it covers
                best_candidate = None
                best_gain = -1
                for cand in candidate_set:
                    subs = block_t_sets[cand]
                    gain = np.sum(uncovered_mask[subs])
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate is None:
                    best_candidate = random.randint(0, n_ks - 1)
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_sets[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_sets[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Polish pass
        current_blocks = polish(current_blocks)
        
        return current_blocks, len(current_blocks)
    
    def compute_fitness(block_set):
        """
        Compute fitness for a block set.
        Fitness = number of blocks + penalty for uncovered triples.
        Lower is better.
        """
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in block_set:
            coverage_count[block_t_sets[b_idx]] += 1
        
        uncovered = np.sum(coverage_count == 0)
        fitness = len(block_set) + uncovered * 1.0
        return fitness, coverage_count, uncovered
    
    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    # Phase 2: Final polish on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 2.0:
        polished = polish(best_solution, max_rounds=100)
        if len(polished) < best_score:
            best_score = len(polished)
            best_solution = polished
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
