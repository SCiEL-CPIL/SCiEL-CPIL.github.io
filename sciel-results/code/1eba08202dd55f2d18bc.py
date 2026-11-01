"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + numpy-vectorized local search with precomputed coverage arrays.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and coverage structures."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # coverage_bits: list of int bitmasks
    coverage_bits = []
    # coverage_arr: 2D numpy array [n_k, n_t] with 0/1
    cov_rows = []
    for ks in k_subsets:
        mask = 0
        row = np.zeros(n_t, dtype=np.int8)
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
            row[idx] = 1
        coverage_bits.append(mask)
        cov_rows.append(row)
    
    coverage_arr = np.array(cov_rows, dtype=np.int8)
    
    # tsub_to_blocks: for each t-subset, list of k-subset indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t):
            if coverage_arr[i, j]:
                tsub_to_blocks[j].append(i)
    
    return all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks, coverage_arr


def _greedy_construct(v, k, t, rng):
    """Greedy set cover with bitset operations."""
    all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks, coverage_arr = _precompute(v, k, t)
    n_t = len(all_t)
    uncovered = (1 << n_t) - 1
    blocks = []
    
    while uncovered and _elapsed() < _DEADLINE - 25:
        best_i = -1
        best_gain = 0
        for i, mask in enumerate(coverage_bits):
            g = (mask & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
    return blocks, all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks, coverage_arr


def _local_search_fast(blocks, coverage_bits, coverage_arr, n_t, n_k, tsub_to_blocks, rng, time_budget):
    """Vectorized local search using numpy coverage arrays."""
    blocks = list(blocks)
    
    # cover_count: how many blocks cover each t-subset
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        cover_count += coverage_arr[b]
    
    # unique_cover: indices where cover_count == 1
    # For each block, precompute which t-subsets it uniquely covers
    # unique_mask[i] = 1 if block i uniquely covers t-subset j
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Block is removable if no t-subset it covers has cover_count == 1
            row = coverage_arr[b]
            # Check if any position where row==1 also has cover_count==1
            if np.any(row[cover_count == 1]):
                i += 1
            else:
                # Remove block
                cover_count -= row
                blocks.pop(i)
                improved = True
        
        # --- Swap pass ---
        n_swaps = 40
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            
            i = rng.randrange(len(blocks))
            b_out = blocks[i]
            row_out = coverage_arr[b_out]
            
            # Find t-subsets uniquely covered by b_out
            unique_mask = (row_out == 1) & (cover_count == 1)
            if not np.any(unique_mask):
                # Can remove
                cover_count -= row_out
                blocks.pop(i)
                improved = True
                break
            
            unique_indices = np.where(unique_mask)[0]
            first_idx = unique_indices[0]
            candidates = tsub_to_blocks[first_idx]
            
            found = False
            for c in candidates:
                if c in blocks:
                    continue
                row_c = coverage_arr[c]
                # Check if c covers all unique t-subsets
                if np.all(row_c[unique_indices] == 1):
                    # Perform swap
                    cover_count -= row_out
                    cover_count += row_c
                    blocks[i] = c
                    found = True
                    improved = True
                    break
            
            if not found:
                continue
        
        # --- Add pass: repair if needed ---
        if len(blocks) > 0:
            total_cov = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                total_cov += coverage_arr[b]
            if np.any(total_cov == 0):
                missing_mask = (total_cov == 0).astype(np.int8)
                # Find block covering most missing
                gains = coverage_arr @ missing_mask
                best_i = int(np.argmax(gains))
                if gains[best_i] > 0:
                    blocks.append(best_i)
                    cover_count += coverage_arr[best_i]
    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks, coverage_arr = _precompute(v, k, t)
    n_t = len(all_t)
    n_k = len(k_subsets)

    # Greedy construction
    blocks, _, _, _, _, _, _ = _greedy_construct(v, k, t, rng)
    
    # Local search with numpy-vectorized approach
    time_budget = _DEADLINE - 8
    blocks = _local_search_fast(blocks, coverage_bits, coverage_arr, n_t, n_k, tsub_to_blocks, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    total_cov = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        total_cov += coverage_arr[b]
    
    if np.any(total_cov == 0):
        while _elapsed() < _DEADLINE - 2:
            missing_mask = (total_cov == 0).astype(np.int8)
            gains = coverage_arr @ missing_mask
            best_i = int(np.argmax(gains))
            if gains[best_i] == 0:
                break
            blocks.append(best_i)
            total_cov += coverage_arr[best_i]
            if not np.any(total_cov == 0):
                break

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
