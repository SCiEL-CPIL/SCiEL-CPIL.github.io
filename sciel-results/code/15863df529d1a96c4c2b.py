"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _neighbors_indices(cw, q, n):
    """Return boolean numpy array of length q**n indicating words covered by cw within radius 1."""
    space_size = q ** n
    idx = np.zeros(space_size, dtype=bool)
    base = np.array(cw, dtype=np.int64)
    idx[int(np.ravel_multi_index(tuple(base), [q] * n))] = True
    for pos in range(n):
        for d in range(q):
            if d == base[pos]:
                continue
            cand = base.copy()
            cand[pos] = d
            idx[int(np.ravel_multi_index(tuple(cand), [q] * n))] = True
    return idx


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _precompute_balls(q, n):
    """Precompute the list of neighbor indices (within distance 1) for every word in the space."""
    space_size = q ** n
    balls = []
    for i in range(space_size):
        word = [int(d) for d in np.unravel_index(i, [q] * n)]
        ball = [i]
        for pos in range(n):
            orig = word[pos]
            for d in range(q):
                if d == orig:
                    continue
                word[pos] = d
                ball.append(int(np.ravel_multi_index(tuple(word), [q] * n)))
            word[pos] = orig
        balls.append(ball)
    return balls


def _local_search(q, n, radius, initial_words, time_limit=120.0):
    """
    Local search starting from initial_words.
    Uses randomized removal to reduce size, then tries swaps if time permits.
    Returns the list of codewords (lists).
    """
    space_size = q ** n
    balls = _precompute_balls(q, n)
    
    # Convert initial words to indices
    active_indices = set()
    for w in initial_words:
        idx = int(np.ravel_multi_index(tuple(w), [q] * n))
        active_indices.add(idx)
    
    # Initialize coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    for idx in active_indices:
        for b in balls[idx]:
            counts[b] += 1
    
    def try_remove():
        """Try to remove redundant words. Returns True if any removal happened."""
        nonlocal counts, active_indices
        removed_any = False
        # Shuffle for randomness
        to_check = list(active_indices)
        random.shuffle(to_check)
        for idx in to_check:
            if idx not in active_indices:
                continue
            # Check if any word in its ball has count == 1
            # If so, removing it would uncover that word
            if np.min(counts[balls[idx]]) > 1:
                # Safe to remove
                for b in balls[idx]:
                    counts[b] -= 1
                active_indices.remove(idx)
                removed_any = True
        return removed_any

    # Phase 1: Iterative removal
    while True:
        if time.time() - _start_time > time_limit - 10:
            break
        if not try_remove():
            break

    # Phase 2: Swaps (if time remains)
    if time.time() - _start_time < time_limit - 20:
        # We might try to improve by swapping a codeword with a neighbor
        # that allows further removals. This is complex, so we'll stick to
        # aggressive removal with random restarts if we get stuck.
        
        # Actually, let's just do multiple passes of random removal to escape
        # local minima (though removal is monotonic, the order matters for
        # which words remain). Wait, removal is just deletion. If a word is
        # removable, it is always removable regardless of other removable words
        # being present? No, removing one makes it easier to remove another.
        # But if we can't remove any, we are stuck.
        
        # To escape, we could *add* a word and then remove two? 
        # That's a swap.
        # Let's implement a simple swap: 
        # Pick a random active word A.
        # Pick a random inactive word B.
        # Remove A, Add B. Check validity.
        # If valid, try to remove more.
        
        # This is risky and might increase size.
        # Given the time constraint, let's just ensure we've done
        # multiple randomization passes on the removal order.
        
        # Actually, the simple `try_remove` loop already does one pass.
        # If it stops, no word is individually redundant.
        # To get smaller, we need collective redundancy (swaps).
        
        # Let's try a limited number of swap attempts.
        for _ in range(100):  # Limit swap attempts
            if time.time() - _start_time > time_limit - 10:
                break
            
            if len(active_indices) == 0:
                break
            
            # Pick a random active word to potentially swap out
            current = random.choice(list(active_indices))
            
            # We need to find a candidate B such that 
            # (Union of B_others U B_new) covers everything.
            # Equivalently: No word in B_current is covered *only* by B_current 
            # unless it is also covered by B_new or B_others.
            # Actually, simpler:
            # Remove current. Counts decrease.
            # If any count hits 0, we MUST add a word that covers it.
            # Let's just try random neighbors of current.
            
            # Generate neighbors of current (distance 1)
            curr_word = [int(d) for d in np.unravel_index(current, [q] * n)]
            candidates = []
            for pos in range(n):
                orig = curr_word[pos]
                for d in range(q):
                    if d == orig:
                        continue
                    curr_word[pos] = d
                    candidates.append(int(np.ravel_multi_index(tuple(curr_word), [q] * n)))
                curr_word[pos] = orig
            
            # Also try random words in the space? No, neighbors are better.
            
            random.shuffle(candidates)
            
            swapped = False
            for cand in candidates[:5]:  # Try top 5 random neighbors
                if cand in active_indices:
                    continue
                
                # Simulate swap
                # Remove current
                for b in balls[current]:
                    counts[b] -= 1
                
                # Check if coverage is still full
                if np.min(counts) < 0: # Should be 0 or more. If 0, it's uncovered.
                    # Uncovered words are those where counts == 0
                    uncovered = np.argwhere(counts == 0).flatten()
                    if len(uncovered) > 0:
                        # Can cand cover them?
                        cand_ball_set = set(balls[cand])
                        # We need cand to cover ALL uncovered words?
                        # No, just need the union to cover them.
                        # But we only added cand.
                        # So cand must cover every word that became uncovered.
                        # That's a strong condition.
                        
                        # Let's check if cand covers all uncovered
                        can_cover = True
                        for u in uncovered:
                            if u not in cand_ball_set:
                                can_cover = False
                                break
                        
                        if can_cover:
                            # Accept swap
                            for b in balls[cand]:
                                counts[b] += 1
                            active_indices.remove(current)
                            active_indices.add(cand)
                            swapped = True
                            break
                        else:
                            # Revert counts
                            for b in balls[current]:
                                counts[b] += 1
                    else:
                        # No uncovered words! Swap is free (and might allow removals)
                        for b in balls[cand]:
                            counts[b] += 1
                        active_indices.remove(current)
                        active_indices.add(cand)
                        swapped = True
                        break
                else:
                    # No uncovered words after removing current?
                    # That means current was redundant? But we already tried removing it.
                    # If it was redundant, try_remove should have caught it.
                    # Unless counts were > 1 for all neighbors, but removing it made some 0?
                    # No, if min > 1, removal keeps min >= 1.
                    # So if min == 0 after removal, it was NOT redundant.
                    # Wait, if min > 1 before, then min >= 1 after.
                    # So if we are here, min was <= 1 before.
                    # If min == 1 before, removal makes it 0.
                    # So `uncovered` will not be empty.
                    
                    # Just in case, revert
                    for b in balls[current]:
                        counts[b] += 1

            if swapped:
                # After a successful swap, try to remove more words
                try_remove()

    # Convert back to list of words
    result = []
    for idx in active_indices:
        result.append([int(d) for d in np.unravel_index(idx, [q] * n)])
    return result


def _verify_coverage(q, n, radius, words):
    space_size = q ** n
    covered = np.zeros(space_size, dtype=bool)
    for cw in words:
        covered |= _neighbors_indices(cw, q, n)
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

    # Run local search (removal + swaps)
    result = _local_search(q, n, radius, seed_words, time_limit)

    # Verify coverage (sanity check)
    if not _verify_coverage(q, n, radius, result):
        # Fallback to seed
        return seed_words

    return result
# EVOLVE-BLOCK-END
