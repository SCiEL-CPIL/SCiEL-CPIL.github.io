"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ball_table(q, n, radius):
    """
    Build a table mapping each word (as integer in [0, q^n)) to a list of
    all words within Hamming distance <= radius.
    For q=3, n=8, radius=1: each ball has 1 + 8*2 = 17 words.
    Total entries: 6561 * 17 = 111537, very manageable.
    """
    total = q ** n
    ball_table = [None] * total
    for w in range(total):
        # Convert w to list of digits
        digits = [0] * n
        tmp = w
        for i in range(n - 1, -1, -1):
            digits[i] = tmp % q
            tmp //= q
        ball = [w]
        if radius >= 1:
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != digits[pos]:
                        # Compute integer for modified word
                        delta = (new_digit - digits[pos]) * (q ** pos)
                        ball.append(w + delta)
        ball_table[w] = ball
    return ball_table


def _greedy_cover(q, n, radius, ball_table, rng, deadline):
    """
    Greedy set cover: repeatedly pick the word that covers the most uncovered words.
    Returns a list of codewords (as lists of integers).
    """
    total = q ** n
    uncovered = set(range(total))
    codewords = []
    
    # Precompute ball sizes (all same for radius=1, q=3, n=8: 17)
    # But for generality, we'll compute on the fly or precompute.
    
    while uncovered and time.time() < deadline:
        best_word = -1
        best_count = 0
        # Sample a subset of uncovered words to find a good candidate
        # For efficiency, we can't check all total words each time.
        # Instead, pick a random uncovered word, then check its ball for the best candidate.
        
        # Actually, a better greedy: pick the word that maximizes |ball(w) ∩ uncovered|
        # We can't check all 6561 words each iteration (too slow for many iterations).
        # Heuristic: pick a random uncovered word u, then among all words in ball(u),
        # find the one that covers the most uncovered words.
        
        if len(uncovered) > 5000:
            # Early stage: sample multiple random uncovered words
            candidates = set()
            sample_size = min(50, len(uncovered))
            for _ in range(sample_size):
                u = rng.choice(list(uncovered))
                for w in ball_table[u]:
                    candidates.add(w)
        else:
            # Late stage: be more thorough
            candidates = set()
            for u in list(uncovered)[:min(200, len(uncovered))]:
                for w in ball_table[u]:
                    candidates.add(w)
        
        for w in candidates:
            count = 0
            for v in ball_table[w]:
                if v in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word == -1 or best_count == 0:
            break
        
        codewords.append(best_word)
        for v in ball_table[best_word]:
            uncovered.discard(v)
    
    # If we still have uncovered words and time remains, force-add them
    if uncovered and time.time() < deadline:
        for u in uncovered:
            codewords.append(u)
        uncovered.clear()
    
    return codewords


def _local_search(q, n, radius, ball_table, initial_codewords, rng, deadline):
    """
    Local search: try to remove codewords that are redundant, or swap one for another
    to reduce the total count while maintaining coverage.
    """
    codewords = list(initial_codewords)
    total = q ** n
    
    def compute_coverage(codewords):
        """Return a set of all covered words."""
        covered = set()
        for w in codewords:
            for v in ball_table[w]:
                covered.add(v)
        return covered
    
    covered = compute_coverage(codewords)
    
    # Try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if removing codeword[i] still covers everything
            # Quick check: is there any word only covered by codeword[i]?
            # This is expensive. Instead, just try removing and checking.
            # For efficiency, only try removal if the codeword's ball is small relative to coverage.
            
            # Try removing
            trial = codewords[:i] + codewords[i+1:]
            trial_covered = set()
            for w in trial:
                for v in ball_table[w]:
                    trial_covered.add(v)
            if len(trial_covered) == total:
                codewords = trial
                covered = trial_covered
                improved = True
                break
    
    # Try swaps: replace one codeword with another
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            if len(codewords) < 2:
                break
            i = rng.randint(0, len(codewords) - 1)
            # Pick a random new word
            new_w = rng.randint(0, total - 1)
            if new_w in codewords:
                continue
            trial = codewords[:i] + [new_w] + codewords[i+1:]
            trial_covered = set()
            for w in trial:
                for v in ball_table[w]:
                    trial_covered.add(v)
            if len(trial_covered) == total:
                codewords = trial
                covered = trial_covered
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = random.Random(42)
    
    total = q ** n
    
    # Build ball table
    ball_table = _build_ball_table(q, n, radius)
    
    # Start with the tetracode-based construction as initial solution
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Convert to integer representations
    initial_ints = []
    for cw in initial:
        val = 0
        for i, d in enumerate(cw):
            val = val * q + d
        initial_ints.append(val)
    
    best_solution = initial_ints[:]
    best_size = len(best_solution)
    
    # Try greedy cover
    if time.time() < deadline - 10:
        greedy_sol = _greedy_cover(q, n, radius, ball_table, rng, deadline - 10)
        if len(greedy_sol) < best_size:
            best_solution = greedy_sol
            best_size = len(best_solution)
    
    # Local search on best solution
    if time.time() < deadline - 5 and len(best_solution) > 400:
        ls_sol = _local_search(q, n, radius, ball_table, best_solution, rng, deadline - 5)
        if len(ls_sol) < best_size:
            best_solution = ls_sol
            best_size = len(best_solution)
    
    # Convert back to list of lists
    result = []
    for w in best_solution:
        digits = [0] * n
        tmp = w
        for i in range(n - 1, -1, -1):
            digits[i] = tmp % q
            tmp //= q
        result.append(digits)
    
    return result
# EVOLVE-BLOCK-END
