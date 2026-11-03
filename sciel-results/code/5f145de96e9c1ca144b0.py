"""Ternary covering code K_3(8,1) construction via optimized greedy + local search."""
import time
import itertools
import random
import sys
import numpy as np

# Deadline: must return well before 150s wall-clock
_DEADLINE = 140.0  # seconds from start of construct call


def _hamming_dist(a, b):
    """Hamming distance between two equal-length lists."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _all_neighbors(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    yield tuple(word)
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    w = list(word)
                    w[pos] = val
                    yield tuple(w)
    if radius >= 2:
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for v1 in range(q):
                    if v1 == word[pos1]:
                        continue
                    for v2 in range(q):
                        if v2 == word[pos2]:
                            continue
                        w = list(word)
                        w[pos1] = v1
                        w[pos2] = v2
                        yield tuple(w)


def _precompute_balls_bitsets(q, n, radius):
    """Precompute the ball (set of neighbors) for every word in {0..q-1}^n.
    Returns a list of Python integers representing bitsets of covered indices.
    Also returns mapping functions.
    """
    total = q ** n
    
    # Precompute index mappings
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total)]
    
    # Build bitsets for each word's ball
    balls_bitsets = [0] * total
    for i in range(total):
        w = all_words[i]
        mask = 0
        for nb in _all_neighbors(w, q, n, radius):
            nb_idx = word_to_idx(nb)
            mask |= (1 << nb_idx)
        balls_bitsets[i] = mask
        
    return balls_bitsets, word_to_idx, idx_to_word, all_words


def _greedy_cover_bitsets(q, n, radius, balls_bitsets, rng, time_limit, start_time):
    """Greedy set cover using bitsets for fast intersection counting."""
    total = q ** n
    # uncovered is a bitset of all words
    uncovered = (1 << total) - 1
    codewords = []
    
    # We need to pick the codeword that maximizes popcount(uncovered & balls_bitsets[i])
    # To speed this up, we can maintain a list of candidates and filter them.
    # However, since total is 6561, iterating through all candidates is 6561 ops.
    # Each op is a bitwise AND and bit_count. This is very fast in Python 3.10+ or with int.bit_count().
    
    start = time.time()
    while uncovered:
        if time.time() - start > time_limit:
            break
            
        best_idx = -1
        best_count = 0
        best_mask = 0
        
        # Iterate through all possible codewords
        # Optimization: We can skip codewords that are already selected? 
        # No, we can re-select, but it's useless. We should only consider unselected or just pick best overall.
        # Actually, standard greedy picks best from ALL candidates.
        
        for i in range(total):
            # Count how many new words this codeword covers
            # intersection = uncovered & balls_bitsets[i]
            # count = intersection.bit_count()
            # Python int bit_count is C-speed
            count = (uncovered & balls_bitsets[i]).bit_count()
            
            if count > best_count:
                best_count = count
                best_idx = i
                best_mask = balls_bitsets[i]
                
        if best_idx == -1 or best_count == 0:
            # Should not happen if uncovered is not empty
            break
            
        codewords.append(best_idx)
        uncovered &= ~best_mask
        
    return codewords, uncovered


def _local_search_bitsets(codewords, q, n, radius, balls_bitsets, rng, time_limit, start_time):
    """Local search: try to remove a codeword if the set still covers everything."""
    total = q ** n
    improved = True
    cw_set = set(codewords)
    
    # Precompute union of balls for current codewords?
    # To check if a codeword is redundant, we need to know if there is any word covered ONLY by it.
    # This is equivalent to: Is (uncovered_by_others) & (ball_of_cw) == 0?
    # Or simpler: Compute the union of all OTHER balls. If ball_cw is a subset of union_others, it can be removed.
    
    start = time.time()
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break
            
        # Compute coverage union for all codewords
        # This is O(K) bit operations. K is ~500. Very fast.
        # But to check removal of i, we need union of all j != i.
        # We can compute total_union = OR(all balls).
        # Then for each i, check if ball_i is subset of (total_union ^ ball_i)? No.
        # Check if ball_i & ~ (OR of all others) == 0.
        
        # Efficient way:
        # 1. Compute total_union = OR(balls[cw] for cw in codewords)
        # 2. For each cw, compute others_union = total_union ^ (ball_cw & ~others)? 
        #    Actually, simpler: 
        #    Let U = Union of all balls.
        #    Let B_i = Ball of codeword i.
        #    Codeword i is redundant if B_i is a subset of Union_{j != i} B_j.
        #    Union_{j != i} B_j = U \ (B_i \ Union_{j != i} B_j) ... circular.
        
        # Alternative:
        # Compute count of coverage for each word? No, that's O(N*K).
        # Bitset approach:
        # For each candidate i to remove:
        #   others_union = 0
        #   for j in codewords:
        #       if j != i: others_union |= balls[j]
        #   if balls[i] & ~others_union == 0: remove i
        
        # This is O(K^2) bit ops per pass. K=500 -> 250k ops. Very fast.
        
        for i in range(len(codewords)):
            if time.time() - start > time_limit:
                break
            
            cw = codewords[i]
            # Calculate union of all other balls
            others_union = 0
            for j in range(len(codewords)):
                if j != i:
                    others_union |= balls_bitsets[codewords[j]]
            
            # Check if cw's ball is fully covered by others
            if (balls_bitsets[cw] & ~others_union) == 0:
                # Redundant, remove it
                codewords.pop(i)
                improved = True
                break # Restart loop as indices changed
                
    return codewords


def _verify_coverage_bitsets(codewords, q, n, radius, balls_bitsets):
    """Check that every word is within radius of some codeword."""
    total = q ** n
    full_mask = (1 << total) - 1
    union_mask = 0
    for cw in codewords:
        union_mask |= balls_bitsets[cw]
    return union_mask == full_mask


def construct(q, n, radius):
    start_time = time.time()
    rng = random.Random(42)

    # For the specific case q=3, n=8, radius=1
    # Precompute bitsets
    balls_bitsets, word_to_idx, idx_to_word, all_words = _precompute_balls_bitsets(q, n, radius)
    total = q ** n

    # Initial solution: Direct sum construction (Tetracode based)
    # Tetracode is a (4, 9, 2) code? No, Tetracode is a ternary code of length 4, size 9, min dist 2.
    # Wait, the standard construction for K_3(8,1) uses the Tetracode.
    # The Tetracode C is a set of 9 words of length 4.
    # The construction is C x {0,1,2}^4.
    # Size: 9 * 3^4 = 9 * 81 = 729.
    # This is a valid covering code.
    
    initial = []
    if q == 3 and n >= 4 and radius == 1:
        # Tetracode: a, b, a+b, a+2b mod 3
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                initial.append(word_to_idx(w))
    else:
        # Fallback: Greedy from scratch
        initial, _ = _greedy_cover_bitsets(q, n, radius, balls_bitsets, rng, 30, start_time)

    # Ensure the initial solution is valid
    if not _verify_coverage_bitsets(initial, q, n, radius, balls_bitsets):
        # Add more codewords to fix gaps
        uncovered = (1 << total) - 1
        for cw in initial:
            uncovered &= ~balls_bitsets[cw]
        
        while uncovered and time.time() - start_time < _DEADLINE - 10:
            best_cw = -1
            best_count = 0
            for i in range(total):
                cnt = (uncovered & balls_bitsets[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_cw = i
            if best_cw < 0 or best_count == 0:
                break
            initial.append(best_cw)
            uncovered &= ~balls_bitsets[best_cw]

    best_solution = list(initial)
    best_size = len(best_solution)

    # Local search to reduce size
    remaining_time = _DEADLINE - 10 - (time.time() - start_time)
    if remaining_time > 5:
        improved = _local_search_bitsets(list(initial), q, n, radius, balls_bitsets, rng, remaining_time, start_time)
        if len(improved) < best_size and _verify_coverage_bitsets(improved, q, n, radius, balls_bitsets):
            best_solution = improved
            best_size = len(improved)

    # Multi-start: try different random seeds for greedy
    # The greedy algorithm is deterministic if we always pick the first max. 
    # To add randomness, we can pick randomly from ties.
    # Let's implement a randomized greedy for multi-start.
    
    def _random_greedy_cover_bitsets(q, n, radius, balls_bitsets, rng, time_limit, start_time):
        total = q ** n
        uncovered = (1 << total) - 1
        codewords = []
        start = time.time()
        while uncovered:
            if time.time() - start > time_limit:
                break
            best_count = 0
            candidates = []
            for i in range(total):
                count = (uncovered & balls_bitsets[i]).bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = rng.choice(candidates)
            codewords.append(best_idx)
            uncovered &= ~balls_bitsets[best_idx]
        return codewords, uncovered

    while time.time() - start_time < _DEADLINE - 15:
        rng2 = random.Random(rng.randint(0, 2**31))
        greedy_sol, _ = _random_greedy_cover_bitsets(q, n, radius, balls_bitsets, rng2, 10, start_time)
        
        if len(greedy_sol) < best_size:
            ls_sol = _local_search_bitsets(list(greedy_sol), q, n, radius, balls_bitsets, rng2, 10, start_time)
            if _verify_coverage_bitsets(ls_sol, q, n, radius, balls_bitsets):
                if len(ls_sol) < best_size:
                    best_solution = ls_sol
                    best_size = len(ls_sol)
        elif len(greedy_sol) == best_size:
            ls_sol = _local_search_bitsets(list(greedy_sol), q, n, radius, balls_bitsets, rng2, 8, start_time)
            if _verify_coverage_bitsets(ls_sol, q, n, radius, balls_bitsets) and len(ls_sol) < best_size:
                best_solution = ls_sol
                best_size = len(ls_sol)

    # Convert indices back to words
    result = [idx_to_word(i) for i in best_solution]

    # Final safety check
    if not _verify_coverage_bitsets(best_solution, q, n, radius, balls_bitsets):
        # Fall back to the direct sum construction which is guaranteed valid
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Just return what we have (may be invalid, but better than nothing)
            pass

    return result
