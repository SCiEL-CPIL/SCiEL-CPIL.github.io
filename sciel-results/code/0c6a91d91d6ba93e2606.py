"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

_START_TIME = None
_DEADLINE_FRAC = 0.85  # safety margin: stop at 85% of 120s


def _elapsed():
    return time.time() - _START_TIME


def _within_deadline(margin=0.0):
    if _START_TIME is None:
        return True
    limit = 120.0 * _DEADLINE_FRAC
    return (time.time() - _START_TIME) < (limit - margin)


def _tetracode_sum(q, n, radius):
    """Direct sum of ternary tetracode with all tails: 9 * 3**(n-4) codewords."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_indices(codeword, q, n, radius):
    """Yield the flat index of every word within Hamming distance <= radius of codeword."""
    codeword = tuple(codeword)
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for vals in itertools.product(range(1, q), repeat=r):
                w = list(codeword)
                for pos, val in zip(positions, vals):
                    w[pos] = (codeword[pos] + val) % q
                yield _flat(w, q, n)


def _flat(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _uncovered_mask(codewords, q, n, radius):
    """Return a boolean numpy array of size q**n; True = covered."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        for idx in _ball_indices(cw, q, n, radius):
            covered[idx] = True
    return covered


def _is_valid(codewords, q, n, radius):
    covered = _uncovered_mask(codewords, q, n, radius)
    return covered.all()


def _greedy_construct(q, n, radius):
    """Greedy set-cover: repeatedly pick the codeword covering the most uncovered words."""
    total = q ** n
    # Precompute all possible codewords and their balls
    all_cw = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_balls = [set(_ball_indices(cw, q, n, radius)) for cw in all_cw]
    
    uncovered = set(range(total))
    codewords = []
    
    while uncovered and _within_deadline():
        best_i = -1
        best_count = 0
        for i, ball in enumerate(all_balls):
            if i in codewords:
                continue
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
            if best_count == len(uncovered):
                break
        if best_i < 0 or best_count == 0:
            break
        codewords.append(all_cw[best_i])
        uncovered -= all_balls[best_i]
    
    return codewords


def _local_search(codewords, q, n, radius, max_iters=500):
    """Local search: try removing redundant codewords and swapping pairs."""
    codewords = [list(cw) for cw in codewords]
    total = q ** n
    
    # Precompute balls for current codewords
    balls = [set(_ball_indices(cw, q, n, radius)) for cw in codewords]
    
    # Compute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _within_deadline():
        improved = False
        iters += 1
        
        # Phase 1: Try removing each codeword if it doesn't break coverage
        for i in range(len(codewords) - 1, -1, -1):
            if not _within_deadline():
                break
            # Check if removing codeword i leaves all words covered
            can_remove = True
            for idx in balls[i]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                for idx in balls[i]:
                    coverage[idx] -= 1
                del codewords[i]
                del balls[i]
                improved = True
                break  # restart after removal
        
        # Phase 2: Try swapping one codeword with another (add one, remove one)
        if not improved and len(codewords) > 1:
            # For each codeword, see if there's a replacement that allows removal
            for i in range(len(codewords)):
                if not _within_deadline():
                    break
                # Try adding a random codeword and then removing i
                # But first, let's just try to find if any single addition enables a removal
                pass  # Skip complex swap for now; focus on removals
    
    return codewords


def _swap_local_search_sa(codewords, q, n, radius, max_iters=2000, initial_temp=10.0, cooling=0.999):
    """Swap-based local search with simulated annealing acceptance.
    
    Tries to replace codewords to reduce total count while maintaining coverage.
    Uses SA to escape local minima by accepting worse solutions with some probability.
    """
    codewords = [list(cw) for cw in codewords]
    total = q ** n
    
    if not codewords:
        return codewords
    
    # Precompute balls for current codewords
    balls = [set(_ball_indices(cw, q, n, radius)) for cw in codewords]
    
    # Compute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1
    
    # Verify initial solution is valid
    if not (coverage >= 1).all():
        return codewords
    
    # Generate all possible codewords not currently in the set
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    current_set = set(tuple(cw) for cw in codewords)
    candidates = [w for w in all_words if w not in current_set]
    
    if not candidates:
        return codewords
    
    # Precompute balls for all candidates (this is expensive but needed for swap evaluation)
    # We'll compute on-demand to save memory
    candidate_balls_cache = {}
    
    def get_candidate_ball(cw_tuple):
        if cw_tuple not in candidate_balls_cache:
            candidate_balls_cache[cw_tuple] = set(_ball_indices(list(cw_tuple), q, n, radius))
        return candidate_balls_cache[cw_tuple]
    
    best_codewords = [list(cw) for cw in codewords]
    best_len = len(codewords)
    current_len = len(codewords)
    
    temp = initial_temp
    rng = random.Random(42)
    
    for it in range(max_iters):
        if not _within_deadline(margin=5.0):
            break
        
        # Try a swap: remove one codeword, add one candidate
        i = rng.randint(0, len(codewords) - 1)
        cand_idx = rng.randint(0, len(candidates) - 1)
        cand_tuple = candidates[cand_idx]
        
        # Check if this swap is valid
        # After removing codeword i and adding cand, all words must still be covered
        # A word is uncovered if: coverage[word] == 1 and word in balls[i] and word not in cand_ball
        
        cand_ball = get_candidate_ball(cand_tuple)
        
        # Count how many words would become uncovered
        # Words in balls[i] with coverage == 1 and NOT in cand_ball
        uncovered_count = 0
        for idx in balls[i]:
            if coverage[idx] == 1 and idx not in cand_ball:
                uncovered_count += 1
                if uncovered_count > 0:
                    break
        
        # Also check: words in cand_ball with coverage == 0 (shouldn't happen if valid)
        # But also: if we remove i, some words lose coverage; if cand doesn't cover them, bad
        
        # Compute the "cost" change: we want to minimize number of codewords
        # Swap keeps same count, so we need a different objective
        # Actually, for reduction we need: remove i, add cand, then check if another can be removed
        
        # Simpler approach: try to find if adding cand allows removal of i
        # i.e., is there a codeword j (possibly i) such that after adding cand, j can be removed?
        
        # Let's try: for each codeword j, check if after adding cand, j can be removed
        # This is O(n * ball_size) per candidate, which is expensive
        
        # Alternative: just try direct swap and check validity
        # New coverage = coverage - balls[i] + cand_ball
        # Valid if min(new_coverage) >= 1
        
        # Quick check: for each word in balls[i] \ cand_ball, coverage must be >= 2
        # For each word in cand_ball \ balls[i], coverage can be anything (adding helps)
        # For words in both, coverage stays same
        
        valid = True
        for idx in balls[i]:
            if coverage[idx] == 1 and idx not in cand_ball:
                valid = False
                break
        
        if valid:
            # Perform the swap
            # Update coverage
            for idx in balls[i]:
                coverage[idx] -= 1
            for idx in cand_ball:
                coverage[idx] += 1
            
            # Update codewords and balls
            codewords[i] = list(cand_tuple)
            balls[i] = cand_ball
            current_set.remove(tuple(codewords[i]))
            # Note: cand_tuple was in candidates, so it's not in current_set
            # But we should update candidates to remove it... skip for simplicity
            
            # Now try to remove any codeword that's now redundant
            removed_any = False
            for j in range(len(codewords) - 1, -1, -1):
                if not _within_deadline(margin=5.0):
                    break
                can_remove = True
                for idx in balls[j]:
                    if coverage[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in balls[j]:
                        coverage[idx] -= 1
                    del codewords[j]
                    del balls[j]
                    current_len -= 1
                    removed_any = True
                    break
            
            if current_len < best_len:
                best_len = current_len
                best_codewords = [list(cw) for cw in codewords]
            
            # Cool down
            temp *= cooling
        else:
            # Invalid swap, cool down anyway
            temp *= cooling
    
    # Return the best found
    return best_codewords


def _greedy_then_local(q, n, radius):
    """Start from tetracode sum, then apply local search."""
    # Start from the 81-word tetracode sum
    codewords = _tetracode_sum(q, n, radius)
    
    # Apply local search to reduce
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def _greedy_with_removal(q, n, radius):
    """Greedy construction followed by removal-based local search."""
    # First, build a greedy solution
    codewords = _greedy_construct(q, n, radius)
    
    # Then apply local search
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    # For the specific case q=3, n=6, radius=1, use the known structure
    if q == 3 and n == 6 and radius == 1:
        # Start from tetracode sum (81 words) and try to reduce to 73
        best = _tetracode_sum(q, n, radius)
        best_len = len(best)
        
        # Try local search starting from tetracode sum
        candidate = _local_search(best, q, n, radius)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
        
        # Try swap-based SA local search starting from tetracode sum
        candidate = _swap_local_search_sa(best, q, n, radius)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
        
        # Try greedy construction + local search
        candidate = _greedy_with_removal(q, n, radius)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
        
        # Try swap-based SA on greedy result
        candidate = _swap_local_search_sa(best, q, n, radius)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
        
        # Verify validity
        if _is_valid(best, q, n, radius):
            return best
        
        # Fallback: return the tetracode sum (known valid)
        return _tetracode_sum(q, n, radius)
    else:
        # General case: use greedy construction
        codewords = _greedy_construct(q, n, radius)
        if _is_valid(codewords, q, n, radius):
            return codewords
        # Fallback: brute-force all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
