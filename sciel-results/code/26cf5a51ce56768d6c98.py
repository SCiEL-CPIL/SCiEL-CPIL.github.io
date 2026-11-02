"""Football pool / ternary covering code K_3(8,1) via bit-parallel greedy set cover + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precomputation constants
_Q, _N, _R = 3, 8, 1
_TOTAL = _Q ** _N  # 6561
_NUM_NEIGHBORS = 1 + _N  # 9


def _precompute():
    """Build neighbor lists and the base solution (direct sum of tetracode)."""
    # neighbor_lists[i] = list of indices within distance 1 of i
    neighbor_lists = []
    for i in range(_TOTAL):
        # convert i to base-3 word
        w = []
        x = i
        for _ in range(_N):
            w.append(x % 3)
            x //= 3
        w = w[::-1]  # most significant first
        
        nbrs = [i]
        for pos in range(_N):
            for val in range(_Q):
                if val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = val
                # convert back to int
                idx = 0
                for d in new_w:
                    idx = idx * 3 + d
                nbrs.append(idx)
        neighbor_lists.append(nbrs)
    
    # Base solution: direct sum of tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_codewords = []
    for h in tetracode:
        for tail_val in range(3 ** 4):
            tail = []
            x = tail_val
            for _ in range(4):
                tail.append(x % 3)
                x //= 3
            tail = tail[::-1]
            base_codewords.append(h + tail)
    
    # Convert base codewords to indices
    base_indices = []
    for w in base_codewords:
        idx = 0
        for d in w:
            idx = idx * 3 + d
        base_indices.append(idx)
    
    return neighbor_lists, base_indices


def _bitmask_from_indices(indices, neighbor_lists):
    """Create a bitmask (list of numpy arrays of uint8) representing union of radius-1 balls."""
    # We use a flat bytearray for the bitmask: _TOTAL bits
    mask = bytearray((_TOTAL + 7) // 8)
    for idx in indices:
        for nbr in neighbor_lists[idx]:
            mask[nbr >> 3] |= (1 << (nbr & 7))
    return mask


def _mask_to_indices(mask, neighbor_lists):
    """Extract indices whose bit is set in mask."""
    indices = []
    for i in range(_TOTAL):
        if mask[i >> 3] & (1 << (i & 7)):
            indices.append(i)
    return indices


def _count_covered(mask):
    """Count number of 1-bits in mask."""
    return sum(bin(b).count('1') for b in mask)


def _greedy_cover(neighbor_lists, rng, time_limit):
    """Greedy set cover with randomization. Returns list of codeword indices."""
    start_time = time.time()
    
    # uncovered: bitmask of yet-uncovered words
    uncovered = bytearray([0xFF] * ((_TOTAL + 7) // 8))
    # Clear bits beyond _TOTAL
    extra = (_TOTAL + 7) // 8 * 8 - _TOTAL
    if extra > 0:
        uncovered[-1] &= (0xFF << extra) & 0xFF
    
    chosen = []
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Check if all covered
        all_covered = all(b == 0xFF for b in uncovered)
        # Actually we need to check if all bits are 1... let's use a different approach
        # Count uncovered
        unc_count = sum(bin(0xFF ^ b).count('1') for b in uncovered)
        # Adjust for extra bits
        if extra > 0:
            unc_count -= bin(0xFF ^ (uncovered[-1] | ((0xFF << extra) & 0xFF))).count('1')
        if unc_count == 0:
            break
        
        # Find best candidate: one that covers the most uncovered words
        best_gain = -1
        best_idx = -1
        best_indices = []
        
        # To speed up, we can sample candidates
        # But for correctness, let's do a full scan with early termination
        # Actually, let's just do full scan - 6561 * 9 = ~59k operations per iteration
        for cand in range(_TOTAL):
            gain = 0
            for nbr in neighbor_lists[cand]:
                if uncovered[nbr >> 3] & (1 << (nbr & 7)):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
                best_indices = neighbor_lists[cand]
        
        if best_gain <= 0:
            break
        
        # Add to chosen
        chosen.append(best_idx)
        
        # Mark covered
        for nbr in best_indices:
            uncovered[nbr >> 3] |= (1 << (nbr & 7))
    
    return chosen


def _local_search(indices, neighbor_lists, rng, time_limit):
    """Try to remove redundant codewords. Returns improved list."""
    start_time = time.time()
    n = len(indices)
    
    if n == 0:
        return indices
    
    # Build coverage count: for each word, how many codewords cover it
    cov_count = np.zeros(_TOTAL, dtype=np.int32)
    for idx in indices:
        for nbr in neighbor_lists[idx]:
            cov_count[nbr] += 1
    
    # Try removing each codeword if it's redundant
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle order for diversity
        order = list(range(n))
        rng.shuffle(order)
        
        for pos in order:
            idx = indices[pos]
            # Check if all neighbors are still covered by other codewords
            redundant = True
            for nbr in neighbor_lists[idx]:
                if cov_count[nbr] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for nbr in neighbor_lists[idx]:
                    cov_count[nbr] -= 1
                indices.pop(pos)
                n -= 1
                improved = True
    
    return indices


def _swap_local_search(indices, neighbor_lists, rng, time_limit):
    """Try swapping one codeword for another to reduce size or maintain coverage with fewer."""
    start_time = time.time()
    n = len(indices)
    
    if n == 0:
        return indices
    
    cov_count = np.zeros(_TOTAL, dtype=np.int32)
    for idx in indices:
        for nbr in neighbor_lists[idx]:
            cov_count[nbr] += 1
    
    # Build set of current indices for fast lookup
    current_set = set(indices)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        
        for pos in order:
            if time.time() - start_time > time_limit:
                break
            idx = indices[pos]
            
            # Find uncovered-if-removed words
            would_uncover = []
            for nbr in neighbor_lists[idx]:
                if cov_count[nbr] == 1:
                    would_uncover.append(nbr)
            
            if not would_uncover:
                # Redundant, remove
                for nbr in neighbor_lists[idx]:
                    cov_count[nbr] -= 1
                indices.pop(pos)
                n -= 1
                current_set.discard(idx)
                improved = True
                continue
            
            # Try to find a replacement that covers all would_uncover
            # and doesn't add too much
            # Sample some candidates
            candidates = rng.sample(range(_TOTAL), min(100, _TOTAL))
            for cand in candidates:
                if cand in current_set:
                    continue
                # Check if cand covers all would_uncover
                cand_nbrs = set(neighbor_lists[cand])
                if all(w in cand_nbrs for w in would_uncover):
                    # Do the swap
                    # Remove old
                    for nbr in neighbor_lists[idx]:
                        cov_count[nbr] -= 1
                    # Add new
                    for nbr in neighbor_lists[cand]:
                        cov_count[nbr] += 1
                    indices[pos] = cand
                    current_set.discard(idx)
                    current_set.add(cand)
                    improved = True
                    break
    
    return indices


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius."""
    start_time = time.time()
    total_time_limit = 110  # Safety margin before 120s
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: trivial solution (all words)
        return [list(map(int, format(i, '0{}b'.format(n)).zfill(n))) for i in range(q ** n)]
    
    rng = random.Random(42)
    
    # Precompute
    neighbor_lists, base_indices = _precompute()
    
    # Start with base solution and do local search
    best = _local_search(list(base_indices), neighbor_lists, rng, 20)
    best = _swap_local_search(best, neighbor_lists, rng, 30)
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Try greedy with different random seeds
    best_size = len(best)
    
    while remaining > 15:
        # Greedy cover
        t_start = time.time()
        greedy_time = min(10, remaining - 5)
        greedy_result = _greedy_cover(neighbor_lists, rng, greedy_time)
        
        if len(greedy_result) < best_size:
            greedy_result = _local_search(greedy_result, neighbor_lists, rng, min(15, remaining - 5))
            greedy_result = _swap_local_search(greedy_result, neighbor_lists, rng, min(15, remaining - 5))
            if len(greedy_result) < best_size:
                best = greedy_result
                best_size = len(best)
        
        remaining = total_time_limit - (time.time() - start_time)
    
    # Final polish
    if time.time() - start_time < total_time_limit - 5:
        best = _local_search(best, neighbor_lists, rng, min(10, total_time_limit - (time.time() - start_time)))
        best = _swap_local_search(best, neighbor_lists, rng, min(10, total_time_limit - (time.time() - start_time)))
    
    # Convert indices back to words
    result = []
    for idx in best:
        w = []
        x = idx
        for _ in range(n):
            w.append(x % q)
            x //= q
        w = w[::-1]
        result.append(w)
    
    return result
# EVOLVE-BLOCK-END
