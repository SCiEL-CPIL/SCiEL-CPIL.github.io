"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: NumPy-accelerated greedy + local search with removal/swap.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    
    # block_cov[b] = array of t-indices covered by block b
    block_cov = []
    for ks in all_k:
        cov = []
        for ts in itertools.combinations(ks, t):
            cov.append(t_index[ts])
        block_cov.append(np.array(cov, dtype=np.int32))
    
    # t_to_blocks[s] = list of block indices that cover t-subset s
    t_to_blocks = [[] for _ in range(n_t)]
    for b, cov in enumerate(block_cov):
        for s in cov:
            t_to_blocks[s].append(b)
    # Convert to numpy arrays for faster access
    t_to_blocks_arr = [np.array(blks, dtype=np.int32) for blks in t_to_blocks]
    
    return all_t, t_index, all_k, block_cov, t_to_blocks_arr, n_t, n_k


def _greedy_construct(v, k, t, rng):
    all_t, t_index, all_k, block_cov, t_to_blocks_arr, n_t, n_k = _precompute(v, k, t)
    
    selected = set()
    cover_count = np.zeros(n_t, dtype=np.int32)
    uncovered_count = n_t
    
    while uncovered_count > 0 and _elapsed() < _DEADLINE - 25:
        # Find block with maximum gain
        best_b = -1
        best_gain = 0
        
        # For efficiency, only consider blocks that cover at least one uncovered t-subset
        # We can sample or iterate all (n_k=50388, each has 35 t-subsets)
        for b in range(n_k):
            cov = block_cov[b]
            # Gain = number of uncovered t-subsets in this block
            gain = np.sum(cover_count[cov] == 0)
            if gain > best_gain:
                best_gain = gain
                best_b = b
        
        if best_b < 0 or best_gain == 0:
            break
        
        selected.add(best_b)
        cov = block_cov[best_b]
        for s in cov:
            if cover_count[s] == 0:
                uncovered_count -= 1
            cover_count[s] += 1
    
    return selected, all_k, block_cov, t_to_blocks_arr, n_t, n_k


def _local_search(selected, block_cov, t_to_blocks_arr, n_t, n_k, rng, time_budget):
    selected = set(selected)
    
    # Compute cover count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov[b]:
            cover_count[s] += 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        # Sort blocks by how "redundant" they are (fewer unique contributions)
        blocks_list = list(selected)
        # Try removing each block
        for i in range(len(blocks_list)):
            if _elapsed() >= time_budget:
                break
            b = blocks_list[i]
            if b not in selected:
                continue
            # Check if removing b still covers everything
            # A t-subset is uniquely covered by b if cover_count[s] == 1 and s in block_cov[b]
            unique = 0
            for s in block_cov[b]:
                if cover_count[s] == 1:
                    unique += 1
                    break
            if unique == 0:
                # Can remove b
                selected.discard(b)
                for s in block_cov[b]:
                    cover_count[s] -= 1
                improved = True
                blocks_list = list(selected)  # Restart removal pass
        
        # --- Swap pass: replace block with another that enables removals ---
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            b = rng.choice(list(selected))
            new_b = rng.randrange(n_k)
            if new_b in selected:
                continue
            
            # Tentatively swap
            old_b = b
            # Remove old
            for s in block_cov[old_b]:
                cover_count[s] -= 1
            selected.discard(old_b)
            
            # Add new
            selected.add(new_b)
            for s in block_cov[new_b]:
                cover_count[s] += 1
            
            # Try to remove a redundant block
            removed = False
            for cb in list(selected):
                unique = False
                for s in block_cov[cb]:
                    if cover_count[s] == 1:
                        unique = True
                        break
                if not unique:
                    selected.discard(cb)
                    for s in block_cov[cb]:
                        cover_count[s] -= 1
                    removed = True
                    break
            
            if not removed:
                # Revert
                selected.discard(new_b)
                for s in block_cov[new_b]:
                    cover_count[s] -= 1
                selected.add(old_b)
                for s in block_cov[old_b]:
                    cover_count[s] += 1
            else:
                improved = True
    
    return selected


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)
    
    # 1. Greedy construction
    selected, all_k, block_cov, t_to_blocks_arr, n_t, n_k = _greedy_construct(v, k, t, rng)
    
    # 2. Local search
    time_budget = _DEADLINE - 5
    selected = _local_search(selected, block_cov, t_to_blocks_arr, n_t, n_k, rng, time_budget)
    
    # 3. Safety check & repair
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov[b]:
            cover_count[s] += 1
    
    missing = np.where(cover_count == 0)[0]
    if len(missing) > 0:
        missing_set = set(missing.tolist())
        while missing_set and _elapsed() < _DEADLINE - 2:
            # Find block covering most missing
            gain_counts = {}
            for s in missing_set:
                for b in t_to_blocks_arr[s]:
                    gain_counts[b] = gain_counts.get(b, 0) + 1
            if not gain_counts:
                break
            best_b = max(gain_counts, key=gain_counts.get)
            selected.add(best_b)
            for s in block_cov[best_b]:
                if s in missing_set:
                    missing_set.discard(s)
    
    # Return as list of lists
    result = [list(all_k[b]) for b in selected]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
