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


def _local_search_fast(q, n, radius, initial_words, time_limit=120.0):
    """
    Fast local search using numpy for coverage counts.
    """
    space_size = q ** n
    neighbors = _precompute_neighbors(q, n)
    
    # Convert initial words to indices
    active_indices = set()
    for w in initial_words:
        idx = int(np.ravel_multi_index(tuple(w), [q] * n))
        active_indices.add(idx)
    
    # Initialize coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    for idx in active_indices:
        for b in neighbors[idx]:
            counts[b] += 1
            
    def try_remove():
        """Try to remove redundant words. Returns number removed."""
        nonlocal counts, active_indices
        removed_count = 0
        # Create a list of active indices to iterate over
        # We must be careful not to modify the set while iterating
        to_check = list(active_indices)
        random.shuffle(to_check)
        
        for idx in to_check:
            if idx not in active_indices:
                continue
            # Check if any word in its ball has count == 1
            # If min > 1, it's safe to remove
            if np.min(counts[neighbors[idx]]) > 1:
                for b in neighbors[idx]:
                    counts[b] -= 1
                active_indices.remove(idx)
                removed_count += 1
        return removed_count

    # Phase 1: Iterative removal
    # We loop until no more removals are possible
    while True:
        if time.time() - _start_time > time_limit - 5:
            break
        removed = try_remove()
        if removed == 0:
            break
        # If we removed a lot, we might have created new redundancies
        # The loop handles this by repeating

    # Phase 2: Swaps to escape local minima
    # We try to swap an active word with an inactive word
    # If the swap maintains coverage (or allows further removal), we accept it
    swap_attempts = 0
    max_swap_attempts = 50000 # High limit, controlled by time
    
    while time.time() - _start_time < time_limit - 5:
        if len(active_indices) == 0:
            break
            
        # Pick a random active word to remove
        current = random.choice(list(active_indices))
        
        # Pick a random inactive word to add
        # We need a word that is NOT in active_indices
        # Efficient way: pick a random index, check if it's active. 
        # Since active is ~486/6561 ~ 7%, we can just try a few random indices
        cand = -1
        for _ in range(10):
            r = random.randint(0, space_size - 1)
            if r not in active_indices:
                cand = r
                break
        if cand == -1:
            continue

        # Simulate removal of current
        # We need to know which words become uncovered (count drops to 0)
        # Instead of updating counts and checking, we can check:
        # For each b in neighbors[current]:
        #   if counts[b] == 1: 
        #      this word b is ONLY covered by current.
        #      So cand MUST cover b.
        
        must_cover = []
        for b in neighbors[current]:
            if counts[b] == 1:
                must_cover.append(b)
                
        # If there are words that only current covers, cand must cover them all
        if len(must_cover) > 0:
            # Check if cand covers all must_cover words
            # cand's neighbors are neighbors[cand]
            # We need must_cover subset of neighbors[cand]
            # Since neighbors are lists, we can make a set of neighbors[cand] for speed?
            # Or just check if each b in must_cover is in neighbors[cand]
            # neighbors[cand] is a list of 17 ints. 
            # Checking 17 items for each b is fast.
            # But if must_cover is large, it's slow.
            # Let's use a set for cand's neighbors once
            cand_nbr_set = set(neighbors[cand])
            
            valid_swap = True
            for b in must_cover:
                if b not in cand_nbr_set:
                    valid_swap = False
                    break
            
            if not valid_swap:
                continue
            
        # If we get here, the swap is valid (coverage is maintained)
        # Perform the swap
        for b in neighbors[current]:
            counts[b] -= 1
        for b in neighbors[cand]:
            counts[b] += 1
        active_indices.remove(current)
        active_indices.add(cand)
        
        swap_attempts += 1
        
        # After a swap, try to remove redundant words
        # This is crucial: the swap might make other words redundant
        try_remove()
        
        # Occasionally, if we are stuck (no removals after swap), we might want to 
        # try a "random restart" of the local search? 
        # No, just keep swapping.

    # Convert back to list of words
    result = []
    for idx in active_indices:
        result.append([int(d) for d in np.unravel_index(idx, [q] * n)])
    return result


def _verify_coverage(q, n, radius, words):
    space_size = q ** n
    covered = np.zeros(space_size, dtype=bool)
    for cw in words:
        idx = int(np.ravel_multi_index(tuple(cw), [q] * n))
        # We need neighbors here, but we don't have them in scope. 
        # For verification, we can just iterate.
        # This is only for sanity check, so speed is less critical, but let's be efficient.
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

    # Start with the known tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)
    
    # To improve symmetry breaking, we can sort the initial words
    # This doesn't change the set, but ensures determinism if we do any sorting later
    
    # Run local search (removal + swaps)
    result = _local_search_fast(q, n, radius, seed_words, time_limit)

    # Verify coverage (sanity check)
    if not _verify_coverage(q, n, radius, result):
        # Fallback to seed
        return seed_words

    return result
# EVOLVE-BLOCK-END
