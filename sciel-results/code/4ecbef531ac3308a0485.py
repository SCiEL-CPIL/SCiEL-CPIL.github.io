"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _precompute_neighbors(q, n):
    """Precompute neighbor indices for all words in the space."""
    space_size = q ** n
    neighbors = [None] * space_size
    for i in range(space_size):
        word = [int(d) for d in np.unravel_index(i, [q] * n)]
        nbrs = [i]
        for pos in range(n):
            orig = word[pos]
            for d in range(q):
                if d == orig:
                    continue
                word[pos] = d
                nbrs.append(int(np.ravel_multi_index(tuple(word), [q] * n)))
            word[pos] = orig
        neighbors[i] = nbrs
    return neighbors


def _greedy_cover(q, n, radius, neighbors, time_limit):
    """
    Greedy set-cover: repeatedly pick the codeword whose ball covers the most
    currently-uncovered words. Returns list of chosen indices.
    """
    space_size = q ** n
    uncovered = np.ones(space_size, dtype=np.int32)
    chosen = []
    remaining = space_size

    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        # Find the index with maximum new coverage
        # For each candidate idx, new_coverage = sum(uncovered[neighbors[idx]])
        # We need to do this efficiently.
        # Option 1: vectorized over all candidates (slow for 6561 candidates * 17 neighbors)
        # Option 2: iterate candidates, compute sum
        # Let's try a vectorized approach: for each position in the neighbor list,
        # sum uncovered over all candidates.
        # neighbors is a list of lists. Convert to numpy array.
        # neighbors_arr shape: (space_size, ball_size)
        # new_cover = uncovered[neighbors_arr]  -> shape (space_size, ball_size)
        # totals = new_cover.sum(axis=1)
        # best = argmax(totals)
        
        # Precompute neighbors as a numpy array for vectorized operations
        # Do this once outside the loop
        # Actually, let's just do it here on first call
        # For simplicity and speed, let's use a different approach:
        # We'll precompute the neighbors array once.
        
        # Since we can't easily do this inside without re-computing,
        # let's use a simpler iterative approach that's still fast enough.
        # 6561 candidates, each with 17 neighbors. 6561*17 = ~111k operations per round.
        # We'll have maybe 300-500 rounds. That's ~30-50M ops total. Should be OK.
        
        best_idx = -1
        best_count = -1
        for idx in range(space_size):
            if uncovered[idx] == 0 and all(uncovered[b] == 0 for b in neighbors[idx]):
                continue  # This candidate adds nothing new (all its neighbors already covered)
            cnt = 0
            for b in neighbors[idx]:
                if uncovered[b]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        
        if best_idx == -1 or best_count <= 0:
            break
        
        # Add best_idx to chosen set
        chosen.append(best_idx)
        for b in neighbors[best_idx]:
            if uncovered[b]:
                uncovered[b] = 0
                remaining -= 1
    
    return chosen


def _greedy_cover_vectorized(q, n, radius, neighbors_arr, time_limit):
    """
    Vectorized greedy set-cover using numpy.
    neighbors_arr: numpy array of shape (space_size, ball_size)
    """
    space_size = q ** n
    uncovered = np.ones(space_size, dtype=np.int32)
    chosen = []
    remaining = space_size

    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        
        # new_coverage[i] = sum of uncovered[j] for j in neighbors_arr[i]
        # This is a matrix-vector product: neighbors_arr is (S, B), uncovered is (S,)
        # We need: for each row i, sum(uncovered[neighbors_arr[i]])
        # = np.sum(uncovered[neighbors_arr], axis=1)
        
        new_cov = np.sum(uncovered[neighbors_arr], axis=1)
        
        # Find best
        best_idx = int(np.argmax(new_cov))
        best_count = int(new_cov[best_idx])
        
        if best_count <= 0:
            break
        
        chosen.append(best_idx)
        for b in neighbors_arr[best_idx]:
            b = int(b)
            if uncovered[b]:
                uncovered[b] = 0
                remaining -= 1
    
    return chosen


def _iterative_prune(q, n, radius, chosen_indices, neighbors, time_limit):
    """
    Iteratively remove redundant codewords. A codeword is redundant if
    every word in its ball is covered by at least one other codeword.
    """
    space_size = q ** n
    active = set(chosen_indices)
    
    # Build coverage count array
    counts = np.zeros(space_size, dtype=np.int32)
    for idx in active:
        for b in neighbors[idx]:
            counts[b] += 1
    
    removed_total = 0
    while True:
        if time.time() - _start_time > time_limit - 5:
            break
        
        # Find all redundant codewords
        redundant = []
        for idx in list(active):
            # Check if all neighbors have count > 1
            # i.e., min count in ball > 1
            min_cnt = 10  # max possible count is len(active)
            for b in neighbors[idx]:
                c = counts[b]
                if c < min_cnt:
                    min_cnt = c
                    if min_cnt <= 1:
                        break
            if min_cnt > 1:
                redundant.append(idx)
        
        if not redundant:
            break
        
        # Remove all redundant at once
        for idx in redundant:
            for b in neighbors[idx]:
                counts[b] -= 1
            active.remove(idx)
            removed_total += 1
    
    return list(active)


def _verify_coverage(q, n, radius, words):
    space_size = q ** n
    covered = np.zeros(space_size, dtype=bool)
    for cw in words:
        idx = int(np.ravel_multi_index(tuple(cw), [q] * n))
        word = [int(d) for d in np.unravel_index(idx, [q] * n)]
        covered[idx] = True
        for pos in range(n):
            orig = word[pos]
            for d in range(q):
                if d == orig:
                    continue
                word[pos] = d
                nidx = int(np.ravel_multi_index(tuple(word), [q] * n))
                covered[nidx] = True
            word[pos] = orig
    return np.all(covered)


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    # If parameters don't match the ternary length-8 radius-1 case, fall back to tetracode
    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Precompute neighbors
    neighbors = _precompute_neighbors(q, n)
    space_size = q ** n
    
    # Convert neighbors to numpy array for vectorized greedy
    ball_size = len(neighbors[0])
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    # Phase 1: Greedy cover
    greedy_result = _greedy_cover_vectorized(q, n, radius, neighbors_arr, time_limit)
    
    # Phase 2: Iterative pruning
    pruned_result = _iterative_prune(q, n, radius, greedy_result, neighbors, time_limit)
    
    # Convert indices back to words
    result = []
    for idx in pruned_result:
        result.append([int(d) for d in np.unravel_index(idx, [q] * n)])
    
    # Verify coverage (sanity check)
    if not _verify_coverage(q, n, radius, result):
        # Fallback to tetracode solution
        return _tetracode_solution(q, n, radius)

    return result
# EVOLVE-BLOCK-END
