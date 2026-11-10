"""Optimized covering design C(19,8,5) using precomputed block-tsub mappings and local search."""
import time
import random
import itertools
import numpy as np

# Global precomputation cache
_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Map k-subset to index
    ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
    
    # Precompute t-subsets covered by each k-subset
    # block_tsubs[i] is a set of t-subset indices
    block_tsubs = [set() for _ in range(n_ksubs)]
    # Precompute k-subsets containing each t-subset
    # tsub_blocks[j] is a list of k-subset indices
    tsub_blocks = [[] for _ in range(n_tsubs)]
    
    for i, ks in enumerate(all_ksubs):
        ks_set = set(ks)
        for ts in itertools.combinations(ks, t):
            j = tsub_to_idx[ts]
            block_tsubs[i].add(j)
            tsub_blocks[j].append(i)
            
    _CACHE[(v, k, t)] = (all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs, tsub_blocks, n_tsubs, n_ksubs)
    return _CACHE[(v, k, t)]

def _greedy_cover_fast(v, k, t, seed=42, max_time=20.0):
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs, tsub_blocks, n_tsubs, n_ksubs = _get_precomputed(v, k, t)
    
    covered = [False] * n_tsubs
    uncovered_count = n_tsubs
    blocks_in_cover = set()
    blocks_list = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset
        # Randomly pick one to avoid bias
        # To be faster, we can just pick the first one found, but random helps diversity
        # Let's pick a random index from uncovered ones. 
        # Efficient way: maintain a list of uncovered indices? 
        # Or just scan. n_tsubs is 11628. Scanning is fast.
        uncovered_indices = [i for i in range(n_tsubs) if not covered[i]]
        if not uncovered_indices:
            break
            
        focus_idx = rng.choice(uncovered_indices)
        
        # Get candidate blocks containing this t-subset
        candidates = tsub_blocks[focus_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidates:
            if b_idx in blocks_in_cover:
                continue
                
            # Calculate gain: how many uncovered t-subsets does this block cover?
            gain = 0
            for ts_idx in block_tsubs[b_idx]:
                if not covered[ts_idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and best_block_idx != -1 and rng.random() < 0.1:
                # Random tie break
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            # Should not happen if candidates is not empty
            break
            
        blocks_in_cover.add(best_block_idx)
        blocks_list.append(best_block_idx)
        
        for ts_idx in block_tsubs[best_block_idx]:
            if not covered[ts_idx]:
                covered[ts_idx] = True
                uncovered_count -= 1
                
    return [all_ksubs[i] for i in blocks_list]

def _local_search(blocks, v, k, t, max_time=100.0):
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, all_ksubs, tsub_to_idx, ksub_to_idx, block_tsubs, tsub_blocks, n_tsubs, n_ksubs = _get_precomputed(v, k, t)
    
    # Convert blocks to indices
    block_indices = set()
    for b in blocks:
        block_indices.add(ksub_to_idx[b])
    
    # Coverage count for each t-subset
    # cov_count[i] = number of blocks in block_indices covering t-subset i
    cov_count = [0] * n_tsubs
    for b_idx in block_indices:
        for ts_idx in block_tsubs[b_idx]:
            cov_count[ts_idx] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid getting stuck in specific order
        current_indices = list(block_indices)
        random.shuffle(current_indices)
        
        for b_idx in current_indices:
            if time.time() > deadline:
                break
                
            # Check if b_idx is redundant
            redundant = True
            for ts_idx in block_tsubs[b_idx]:
                if cov_count[ts_idx] <= 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove it
                block_indices.remove(b_idx)
                for ts_idx in block_tsubs[b_idx]:
                    cov_count[ts_idx] -= 1
                improved = True
                continue
            
            # Try to replace b_idx with another block
            # Identify t-subsets that would become uncovered if we remove b_idx
            lost_tsubs = []
            for ts_idx in block_tsubs[b_idx]:
                if cov_count[ts_idx] == 1:
                    lost_tsubs.append(ts_idx)
                    
            if not lost_tsubs:
                # Should have been caught by redundant check, but just in case
                block_indices.remove(b_idx)
                for ts_idx in block_tsubs[b_idx]:
                    cov_count[ts_idx] -= 1
                improved = True
                continue
                
            # Find a replacement block that covers all lost_tsubs
            # We can find candidates by intersecting tsub_blocks for each lost t-subset
            # Since lost_tsubs is usually small (1-3), this is fast
            # Start with the first lost t-subset's blocks
            candidate_replacements = set(tsub_blocks[lost_tsubs[0]])
            for ts_idx in lost_tsubs[1:]:
                candidate_replacements &= set(tsub_blocks[ts_idx])
                
            # Remove current block from candidates
            candidate_replacements.discard(b_idx)
            
            if not candidate_replacements:
                continue
                
            # Pick a candidate. Heuristic: pick one that maximizes "redundancy creation"
            # i.e., covers t-subsets that are already covered by multiple blocks
            # Or just pick randomly to explore
            # Let's pick the one that covers the most t-subsets with cov_count > 1
            best_repl = None
            best_score = -1
            
            # Limit search to a few candidates if too many
            candidates_list = list(candidate_replacements)
            if len(candidates_list) > 50:
                random.shuffle(candidates_list)
                candidates_list = candidates_list[:50]
                
            for repl_idx in candidates_list:
                if repl_idx in block_indices:
                    continue
                    
                # Score: how many t-subsets in repl_idx have cov_count >= 2?
                # This indicates it might make other blocks redundant
                score = 0
                for ts_idx in block_tsubs[repl_idx]:
                    if cov_count[ts_idx] >= 2:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_repl = repl_idx
                elif score == best_score and best_repl is not None and random.random() < 0.1:
                    best_repl = repl_idx
                    
            if best_repl is not None:
                # Perform swap
                block_indices.remove(b_idx)
                for ts_idx in block_tsubs[b_idx]:
                    cov_count[ts_idx] -= 1
                    
                block_indices.add(best_repl)
                for ts_idx in block_tsubs[best_repl]:
                    cov_count[ts_idx] += 1
                    
                improved = True
                
    return [all_ksubs[i] for i in block_indices]

def construct(v, k, t):
    total_start = time.time()
    total_deadline = total_start + 145.0
    
    # 1. Greedy Phase
    # Run multiple greedy restarts to find a good initial solution
    best_blocks = None
    best_len = float('inf')
    
    # Allocate 20s for greedy
    greedy_deadline = total_start + 20.0
    seeds = [1, 42, 123, 777, 2024]
    for seed in seeds:
        if time.time() > greedy_deadline:
            break
        blocks = _greedy_cover_fast(v, k, t, seed=seed, max_time=5.0)
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover_fast(v, k, t, seed=0, max_time=10.0)
        
    # 2. Local Search Phase
    # Use remaining time
    remaining_time = total_deadline - time.time()
    if remaining_time > 5.0:
        best_blocks = _local_search(best_blocks, v, k, t, max_time=remaining_time - 2.0)
        
    return best_blocks
