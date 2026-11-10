"""Optimized covering design C(19,8,5) using vectorized greedy with bottleneck heuristic and local search."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

# Global precomputation cache
_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
    
    # Precompute t-subsets covered by each k-subset as numpy arrays for speed
    block_tsubs_arrays = []
    # tsub_blocks[j] is a list of k-subset indices
    tsub_blocks = [[] for _ in range(n_tsubs)]
    
    for i, ks in enumerate(all_ksubs):
        ts_indices = []
        for ts in itertools.combinations(ks, t):
            j = tsub_to_idx[ts]
            ts_indices.append(j)
            tsub_blocks[j].append(i)
        block_tsubs_arrays.append(np.array(ts_indices, dtype=np.int32))
    
    _CACHE[(v, k, t)] = (all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs_arrays, tsub_blocks, n_tsubs, n_ksubs)
    return _CACHE[(v, k, t)]

def _greedy_bottleneck(v, k, t, seed=42, max_time=15.0):
    """
    Greedy that selects blocks by minimizing the maximum uncovered degree
    among candidate blocks (bottleneck heuristic).
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs_arrays, tsub_blocks, n_tsubs, n_ksubs = _get_precomputed(v, k, t)
    
    covered = np.zeros(n_tsubs, dtype=bool)
    uncovered_count = n_tsubs
    blocks_in_cover = set()
    blocks_list = []
    
    # Track degree: number of times each t-subset is covered (we only care if 0 or >0)
    # For bottleneck heuristic, we look at uncovered t-subsets and their "degree"
    # degree[t] = number of candidate blocks (not yet in cover) that cover t-subset t
    # But that's expensive to maintain. Instead, we use a simpler heuristic:
    # Among uncovered t-subsets, find one with fewest candidate blocks remaining.
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find uncovered t-subsets
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
            
        # For each uncovered t-subset, count how many blocks in tsub_blocks are NOT yet in cover
        # This is the "degree" - lower degree means more constrained
        # Sample up to 200 uncovered t-subsets for efficiency
        if len(uncovered_indices) > 200:
            sample_idx = rng.sample(range(len(uncovered_indices)), 200)
            sample_tsubs = uncovered_indices[sample_idx]
        else:
            sample_tsubs = uncovered_indices
            
        best_tsub = None
        best_degree = float('inf')
        
        for ts_idx in sample_tsubs:
            candidates = tsub_blocks[ts_idx]
            # Count candidates not in cover
            deg = 0
            for b_idx in candidates:
                if b_idx not in blocks_in_cover:
                    deg += 1
                    if deg >= best_degree:
                        break
            if deg < best_degree:
                best_degree = deg
                best_tsub = ts_idx
                
        if best_tsub is None:
            break
            
        # Get candidate blocks for the most constrained t-subset
        candidates = tsub_blocks[best_tsub]
        active_candidates = [b_idx for b_idx in candidates if b_idx not in blocks_in_cover]
        
        if not active_candidates:
            break
            
        # Among active candidates, pick one that covers the most uncovered t-subsets
        # (standard greedy gain) but with a slight bias toward blocks that cover 
        # t-subsets with low degree (more constrained ones)
        best_block_idx = -1
        best_gain = -1
        best_bottleneck = float('inf')
        
        # Limit candidates to top 100 by gain if too many
        if len(active_candidates) > 100:
            # First pass: quick gain estimate for all
            gains = []
            for b_idx in active_candidates:
                ts_arr = block_tsubs_arrays[b_idx]
                gain = np.count_nonzero(~covered[ts_arr])
                gains.append((gain, b_idx))
            gains.sort(reverse=True)
            active_candidates = [x[1] for x in gains[:100]]
        
        for b_idx in active_candidates:
            ts_arr = block_tsubs_arrays[b_idx]
            gain = np.count_nonzero(~covered[ts_arr])
            
            # Bottleneck score: min degree among t-subsets this block covers
            # (lower is better - it helps the most constrained t-subsets)
            min_deg = float('inf')
            for ts_idx in ts_arr:
                if not covered[ts_idx]:
                    # Count active blocks covering this t-subset
                    cand_count = 0
                    for cb in tsub_blocks[ts_idx]:
                        if cb not in blocks_in_cover:
                            cand_count += 1
                    if cand_count < min_deg:
                        min_deg = cand_count
                        if min_deg == 1:
                            break
                            
            # Combined score: prioritize high gain, then low bottleneck
            if gain > best_gain or (gain == best_gain and min_deg < best_bottleneck):
                best_gain = gain
                best_bottleneck = min_deg
                best_block_idx = b_idx
            elif gain == best_gain and min_deg == best_bottleneck and best_block_idx != -1 and rng.random() < 0.15:
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        blocks_in_cover.add(best_block_idx)
        blocks_list.append(best_block_idx)
        
        # Update coverage
        ts_arr = block_tsubs_arrays[best_block_idx]
        newly_covered = ~covered[ts_arr]
        covered[ts_arr] = True
        uncovered_count -= int(np.sum(newly_covered))
                
    return [all_ksubs[i] for i in blocks_list]

def _local_search(blocks, v, k, t, max_time=100.0):
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs_arrays, tsub_blocks, n_tsubs, n_ksubs = _get_precomputed(v, k, t)
    
    block_indices = set()
    for b in blocks:
        block_indices.add(ksub_to_idx[b])
    
    cov_count = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in block_indices:
        cov_count[block_tsubs_arrays[b_idx]] += 1
            
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        current_indices = list(block_indices)
        random.shuffle(current_indices)
        
        for b_idx in current_indices:
            if time.time() > deadline:
                break
                
            ts_arr = block_tsubs_arrays[b_idx]
            
            # Check if redundant
            if np.all(cov_count[ts_arr] > 1):
                block_indices.remove(b_idx)
                cov_count[ts_arr] -= 1
                improved = True
                continue
            
            # Find t-subsets that would become uncovered
            lost_mask = cov_count[ts_arr] == 1
            lost_tsubs = ts_arr[lost_mask]
            
            if len(lost_tsubs) == 0:
                block_indices.remove(b_idx)
                cov_count[ts_arr] -= 1
                improved = True
                continue
                
            # Find replacements
            candidate_replacements = set(tsub_blocks[lost_tsubs[0]])
            for ts_idx in lost_tsubs[1:]:
                candidate_replacements &= set(tsub_blocks[ts_idx])
            candidate_replacements.discard(b_idx)
            
            if not candidate_replacements:
                continue
                
            candidates_list = list(candidate_replacements)
            if len(candidates_list) > 50:
                random.shuffle(candidates_list)
                candidates_list = candidates_list[:50]
                
            best_repl = None
            best_score = -1
            
            for repl_idx in candidates_list:
                if repl_idx in block_indices:
                    continue
                score = int(np.sum(cov_count[block_tsubs_arrays[repl_idx]] >= 2))
                if score > best_score:
                    best_score = score
                    best_repl = repl_idx
                elif score == best_score and best_repl is not None and random.random() < 0.1:
                    best_repl = repl_idx
                    
            if best_repl is not None:
                block_indices.remove(b_idx)
                cov_count[block_tsubs_arrays[b_idx]] -= 1
                block_indices.add(best_repl)
                cov_count[block_tsubs_arrays[best_repl]] += 1
                improved = True
                
    return [all_ksubs[i] for i in block_indices]

def _greedy_standard(v, k, t, seed=42, max_time=15.0):
    """Standard greedy for comparison/fallback."""
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs_arrays, tsub_blocks, n_tsubs, n_ksubs = _get_precomputed(v, k, t)
    
    covered = np.zeros(n_tsubs, dtype=bool)
    uncovered_count = n_tsubs
    blocks_in_cover = set()
    blocks_list = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
            
        focus_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        candidates = tsub_blocks[focus_idx]
        active_candidates = [b_idx for b_idx in candidates if b_idx not in blocks_in_cover]
        
        if not active_candidates:
            break
            
        best_block_idx = -1
        best_gain = -1
        
        if len(active_candidates) > 100:
            gains = []
            for b_idx in active_candidates:
                ts_arr = block_tsubs_arrays[b_idx]
                gain = int(np.count_nonzero(~covered[ts_arr]))
                gains.append((gain, b_idx))
            gains.sort(reverse=True)
            active_candidates = [x[1] for x in gains[:100]]
        
        for b_idx in active_candidates:
            ts_arr = block_tsubs_arrays[b_idx]
            gain = int(np.count_nonzero(~covered[ts_arr]))
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and best_block_idx != -1 and rng.random() < 0.1:
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        blocks_in_cover.add(best_block_idx)
        blocks_list.append(best_block_idx)
        
        ts_arr = block_tsubs_arrays[best_block_idx]
        newly_covered = ~covered[ts_arr]
        covered[ts_arr] = True
        uncovered_count -= int(np.sum(newly_covered))
                
    return [all_ksubs[i] for i in blocks_list]

def construct(v, k, t):
    total_start = time.time()
    total_deadline = total_start + 140.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy restarts with different strategies
    greedy_deadline = total_start + 25.0
    
    strategies = [
        ("bottleneck", _greedy_bottleneck),
        ("standard", _greedy_standard),
    ]
    
    seeds = [1, 42, 123, 777, 2024, 999, 314, 271]
    for seed in seeds:
        if time.time() > greedy_deadline:
            break
        for name, func in strategies:
            if time.time() > greedy_deadline:
                break
            try:
                blocks = func(v, k, t, seed=seed, max_time=3.0)
                if blocks and len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks
            except Exception:
                continue
        
    if best_blocks is None:
        best_blocks = _greedy_standard(v, k, t, seed=0, max_time=15.0)
        if not best_blocks:
            return []
    
    # Phase 2: Local search refinement
    remaining_time = total_deadline - time.time()
    if remaining_time > 10.0:
        best_blocks = _local_search(best_blocks, v, k, t, max_time=remaining_time - 3.0)
        
    return best_blocks
