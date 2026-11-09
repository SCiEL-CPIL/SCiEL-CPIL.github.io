"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _ball1_indices(q, n):
    """Return (self_index, list of (index, coord, value)) for each word in {0..q-1}^n."""
    total = q ** n
    self_idx = np.arange(total, dtype=np.int64)
    neighbors = []
    for pos in range(n):
        stride = q ** (n - 1 - pos)
        for d in range(1, q):
            # words that differ at position pos by +d mod q
            # index = base + pos_val * stride; neighbor = base + ((pos_val+d)%q)*stride
            # We'll compute via broadcasting
            base = (self_idx // (stride * q)) * (stride * q)
            pos_val = (self_idx // stride) % q
            new_val = (pos_val + d) % q
            nbr = base + new_val * stride
            neighbors.append(nbr)
    return self_idx, neighbors


def construct(q, n, radius):
    t_start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    if radius < 1:
        # Trivial: every word is a codeword
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    if radius >= n:
        return [list(range(q))[:0]] if q == 0 else [[0] * n]

    # For the specific case q=3, n=7, radius=1
    # Total words = 3^7 = 2187
    total = q ** n
    if total > 1000000:
        # Fallback: tetracode-like construction if available
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute all words as integers 0..total-1
    # Precompute the ball of radius 1 for each word
    # ball[word] = set of words within Hamming distance <= 1
    ball = [None] * total
    for w in range(total):
        digits = []
        x = w
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits = digits[::-1]  # most significant first
        covered = {w}
        for pos in range(n):
            for d in range(q):
                if d == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = d
                # Convert back to int
                val = 0
                for dig in new_digits:
                    val = val * q + dig
                covered.add(val)
        ball[w] = covered

    # Greedy set cover with randomization
    def greedy_cover(seed, max_iters=None):
        rng = random.Random(seed)
        uncovered = set(range(total))
        code = []
        
        while uncovered:
            # Find the word that covers the most uncovered points
            best_word = None
            best_count = -1
            best_cover = None
            
            # Sample a subset of candidates if too many remain
            candidates = list(uncovered)
            if len(candidates) > 500:
                # Use a heuristic: prefer words that are far from existing code
                # For speed, sample
                sample = rng.sample(candidates, 500)
                candidates = sample
            
            for w in candidates:
                cov = len(ball[w] & uncovered)
                if cov > best_count:
                    best_count = cov
                    best_word = w
                    best_cover = ball[w]
            
            if best_word is None or best_count == 0:
                # Should not happen if ball is correct
                break
                
            code.append(best_word)
            uncovered -= best_cover
            
            if max_iters and len(code) >= max_iters:
                break
            
            if time.time() - t_start > DEADLINE - 5:
                break
        
        return code

    def is_valid_cover(code):
        """Check if code covers all words."""
        if not code:
            return total == 0
        covered = set()
        for w in code:
            covered |= ball[w]
        return len(covered) == total

    def local_improve(code, max_rounds=3):
        """Try to remove redundant codewords."""
        code_set = set(code)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for i, w in enumerate(list(code_set)):
                # Check if removing w still covers everything
                remaining = code_set - {w}
                if not remaining:
                    if total <= 1:
                        code_set = remaining
                        improved = True
                    continue
                # Check coverage
                covered = set()
                for v in remaining:
                    covered |= ball[v]
                if len(covered) == total:
                    code_set = remaining
                    improved = True
            if time.time() - t_start > DEADLINE - 5:
                break
        return list(code_set)

    # Try multiple restarts with different seeds
    best_code = None
    best_size = float('inf')
    
    # First, try a deterministic greedy (seed=0)
    code0 = greedy_cover(0)
    if is_valid_cover(code0):
        code0 = local_improve(code0)
        best_code = code0
        best_size = len(code0)
    
    # Try random restarts
    seed = 1
    while time.time() - t_start < DEADLINE - 10:
        code = greedy_cover(seed)
        if is_valid_cover(code):
            code = local_improve(code, max_rounds=1)
            if len(code) < best_size:
                best_code = code
                best_size = len(code)
                # If we hit the known best, we can stop early
                if best_size <= 186:
                    break
        seed += 1
    
    if best_code is None:
        # Fallback: tetracode construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            best_code = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Convert integer codewords to lists of digits
    result = []
    for w in best_code:
        digits = []
        x = w
        for _ in range(n):
            digits.append(x % q)
            x //= q
        result.append(digits[::-1])
    
    return result
# EVOLVE-BLOCK-END
