"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _build_neighbor_bitsets(q, n, radius):
    """
    Precompute for every space point a list of neighbor indices (within Hamming distance <= radius),
    and a numpy int64 bitset (list of 64-bit words) representing that neighborhood.
    Also build a mapping from space-point index to the list of codeword-slots whose bitset covers it.
    """
    total = q ** n
    # Neighbor index lists
    neigh_lists = []
    # Bitset representation: list of int64 arrays, one per space point
    # Each bitset is an array of (total+63)//64 uint64 values
    nwords = (total + 63) // 64
    bitsets = np.zeros((total, nwords), dtype=np.uint64)
    
    for idx in range(total):
        word = _idx_to_word(idx, q, n)
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for nv in range(q):
                    if nv == orig:
                        continue
                    nw = word[:]
                    nw[pos] = nv
                    ni = _word_to_idx(nw, q, n)
                    nbrs.append(ni)
        neigh_lists.append(nbrs)
        for nb in nbrs:
            bitsets[idx, nb // 64] |= (1 << (nb % 64))
    
    return neigh_lists, bitsets


def _tetracode_solution(q, n, radius):
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _bitset_popcount(bs, nwords):
    """Popcount of a uint64 bitset array of length nwords."""
    return int(np.bit_count(bs).sum()) if hasattr(np, 'bit_count') else int(bin(int(np.prod(bs, dtype=np.uint64) if nwords == 1 else 0)).count('1'))


def _bitset_popcount_fast(bs, nwords):
    """Use numpy bit_count if available (numpy >= 1.24), else manual."""
    try:
        return int(np.bit_count(bs).sum())
    except AttributeError:
        # Fallback: unpack
        cnt = 0
        for w in bs:
            cnt += bin(int(w)).count('1')
        return cnt


def _bitset_and_popcount(a, b, nwords):
    """Popcount of a & b."""
    try:
        return int(np.bit_count(a & b).sum())
    except AttributeError:
        cnt = 0
        for i in range(nwords):
            cnt += bin(int(a[i] & b[i])).count('1')
        return cnt


def _greedy_cover_bitset(q, n, radius, deadline, start_time):
    """
    Greedy set cover using precomputed bitsets.
    At each step, pick the codeword maximizing (neighborhood & ~covered) popcount.
    """
    total = q ** n
    nwords = (total + 63) // 64
    
    neigh_lists, bitsets = _build_neighbor_bitsets(q, n, radius)
    
    # uncovered: bitset of all 1s
    uncovered = np.ones(nwords, dtype=np.uint64)
    # Mask out bits beyond total in the last word
    if total % 64 != 0:
        uncovered[-1] &= (1 << (total % 64)) - 1
    
    codewords = []
    
    while time.time() < deadline:
        # Find best: maximize popcount(bitsets[i] & uncovered)
        # Vectorized: compute popcount for all i
        # bitsets is (total, nwords); uncovered is (nwords,)
        # We need popcount of bitsets[i] & uncovered for each i
        # This is: sum over words of popcount(bitsets[:, w] & uncovered[w])
        # Use numpy: for each word position, compute popcount of (bitsets[:, w] & uncovered[w])
        # Then sum across words.
        
        # Efficient: loop over nwords (small, ~103 for 6561)
        scores = np.zeros(total, dtype=np.int64)
        for w in range(nwords):
            if uncovered[w] == 0:
                continue
            scores += np.bit_count(bitsets[:, w] & uncovered[w])
        
        best_idx = int(np.argmax(scores))
        best_score = int(scores[best_idx])
        
        if best_score == 0:
            break
        
        codewords.append(best_idx)
        uncovered &= ~bitsets[best_idx]
        
        if _bitset_popcount_fast(uncovered, nwords) == 0:
            break
    
    if _bitset_popcount_fast(uncovered, nwords) == 0:
        return [ _idx_to_word(i, q, n) for i in codewords ]
    return None


def _local_search_bitset(codeword_indices, q, n, radius, deadline, start_time):
    """
    Local search: try to remove codewords and swap (remove one, add one) to reduce count.
    Uses bitset representation for fast coverage checks.
    """
    total = q ** n
    nwords = (total + 63) // 64
    neigh_lists, bitsets = _build_neighbor_bitsets(q, n, radius)
    
    # Current coverage count per space point
    cover_count = np.zeros(total, dtype=np.int32)
    for ci in codeword_indices:
        for nb in neigh_lists[ci]:
            cover_count[nb] += 1
    
    cw_set = set(codeword_indices)
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Phase 1: Try removing each codeword
        for i, ci in enumerate(list(codeword_indices)):
            if time.time() > deadline - 2:
                break
            # Check if ci is redundant: all its neighbors have cover_count > 1
            nbrs = neigh_lists[ci]
            if np.all(cover_count[nbrs] > 1):
                # Remove ci
                for nb in nbrs:
                    cover_count[nb] -= 1
                codeword_indices.pop(i)
                cw_set.discard(ci)
                improved = True
                break  # Restart scan
        
        # Phase 2: Try swaps - remove one, add one (2-opt style)
        if not improved and time.time() < deadline - 5:
            # Find the "risky" points: those with cover_count == 1
            risky_mask = (cover_count == 1)
            risky_points = np.where(risky_mask)[0]
            
            if len(risky_points) > 0:
                # For each codeword that covers a risky point, try replacing it
                # A codeword ci is "risky" if any of its neighbors is a risky point
                # For each risky codeword, try adding a neighbor of its neighbors that covers more uncovered-when-removed points
                
                # Build: for each space point, which codewords cover it (only those with cover_count contribution)
                # Actually, for swap: remove ci, then find best new codeword cj that covers all points that would become uncovered
                # Points that would become uncovered when removing ci: {p in neigh(ci) : cover_count[p] == 1}
                
                for ci in list(codeword_indices):
                    if time.time() > deadline - 2:
                        break
                    nbrs_ci = neigh_lists[ci]
                    would_uncover = [p for p in nbrs_ci if cover_count[p] == 1]
                    if not would_uncover:
                        continue  # ci is redundant, should have been caught in phase 1
                    
                    # Find candidate cj: a space point that covers all of would_uncover
                    # i.e., cj must be within distance 1 of all points in would_uncover
                    # This means cj is in the intersection of neighborhoods of all points in would_uncover
                    # For small |would_uncover|, compute intersection
                    
                    if len(would_uncover) == 1:
                        # Any neighbor of would_uncover[0] works (except ci itself)
                        candidates = [x for x in neigh_lists[would_uncover[0]] if x != ci]
                    elif len(would_uncover) == 2:
                        s1 = set(neigh_lists[would_uncover[0]])
                        s2 = set(neigh_lists[would_uncover[1]])
                        candidates = list(s1 & s2)
                        if ci in candidates:
                            candidates.remove(ci)
                    elif len(would_uncover) <= 4:
                        # Intersection of neighborhoods
                        inter = set(neigh_lists[would_uncover[0]])
                        for p in would_uncover[1:]:
                            inter &= set(neigh_lists[p])
                        candidates = list(inter)
                        if ci in candidates:
                            candidates.remove(ci)
                    else:
                        continue  # Too many constraints, skip
                    
                    if not candidates:
                        continue
                    
                    # Among candidates, pick the one that maximizes coverage gain
                    # Gain = number of points with cover_count < 1 after removal that cj covers
                    # Actually: after removing ci, cover_count decreases by 1 for all nbrs_ci
                    # Then adding cj increases by 1 for all nbrs_cj
                    # Net effect on count: we want to minimize total codewords, so any valid swap works
                    # But to be safe, prefer cj that covers the most points (for future flexibility)
                    best_cj = -1
                    best_gain = -1
                    for cj in candidates:
                        # Quick score: popcount of bitsets[cj] & (cover_count == 0 after removal)
                        # Simpler: just pick the first valid one, or the one with max neighborhood overlap with risky area
                        # For speed, just pick the first candidate
                        best_cj = cj
                        break
                    
                    if best_cj >= 0:
                        # Perform swap
                        for nb in neigh_lists[ci]:
                            cover_count[nb] -= 1
                        for nb in neigh_lists[best_cj]:
                            cover_count[nb] += 1
                        if np.all(cover_count >= 1):
                            codeword_indices.remove(ci)
                            codeword_indices.append(best_cj)
                            improved = True
                            break
    
    return [ _idx_to_word(i, q, n) for i in codeword_indices ]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try tetracode first as a baseline
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_len = len(tc)
    else:
        best = []
        best_len = float('inf')
    
    # Try greedy cover with bitsets
    try:
        greedy_result = _greedy_cover_bitset(q, n, radius, deadline, start_time)
        if greedy_result is not None:
            if len(greedy_result) < best_len:
                best = greedy_result
                best_len = len(best)
    except Exception:
        pass
    
    # Local search to reduce further
    if best and time.time() < deadline - 10:
        try:
            cw_indices = [ _word_to_idx(cw, q, n) for cw in best ]
            ls_result = _local_search_bitset(cw_indices, q, n, radius, deadline, start_time)
            if ls_result and len(ls_result) < best_len:
                best = ls_result
                best_len = len(best)
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
