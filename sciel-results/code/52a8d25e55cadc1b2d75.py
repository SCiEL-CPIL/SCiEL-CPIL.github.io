"""Improved covering code search using greedy construction with local refinement."""
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    Uses greedy set cover with local search refinement.
    """
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit
    
    # For the specific case q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(start_time, DEADLINE)
    
    # General case: greedy set cover
    return _general_greedy(q, n, radius, start_time, DEADLINE)


def _construct_ternary_8_1(start_time, deadline):
    """
    Specialized solver for K_3(8,1).
    Strategy:
    1. Start with a good initial code (tetracode direct sum gives 486).
    2. Try to reduce via local search: remove redundant centers, add better ones.
    3. Use simulated annealing or greedy swap to find smaller codes.
    """
    # Initial code: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Build initial code: for each tetracode word, extend with all 3^4 tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            code.append(h + list(tail))
    
    code = list(set(tuple(c) for c in code))
    code = [list(c) for c in code]
    
    # Precompute the "ball" of each center: all words within distance 1
    # A word w is covered by center c if Hamming distance <= 1
    # For each center, the ball includes: the center itself, and all words differing in exactly 1 position
    
    # For efficiency, represent words as integers 0..3^8-1
    def word_to_int(w):
        val = 0
        for d in w:
            val = val * 3 + d
        return val
    
    def int_to_word(val, n):
        w = []
        for _ in range(n):
            w.append(val % 3)
            val //= 3
        return w[::-1]
    
    n_words = 3 ** 8  # 6561
    
    # Precompute balls for all possible centers? That's 6561 * 9 = ~59000 entries
    # Actually each ball has 1 + 8*2 = 17 words
    # Let's precompute for all 6561 possible centers
    ball_cache = {}
    for w_int in range(n_words):
        w = int_to_word(w_int, 8)
        ball = {w_int}
        for pos in range(8):
            for delta in range(1, 3):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % 3
                new_int = word_to_int(new_w)
                ball.add(new_int)
        ball_cache[w_int] = ball
    
    # Convert current code to set of ints
    code_set = set(word_to_int(c) for c in code)
    
    # Check coverage
    def is_covered(code_set):
        covered = set()
        for c in code_set:
            covered.update(ball_cache[c])
        return len(covered) == n_words
    
    # Local search: try to remove a center if still covered
    def try_remove(code_set):
        """Try to remove a random center; return True if successful."""
        if not code_set:
            return False
        # Try removing centers in random order
        centers = list(code_set)
        random.shuffle(centers)
        for c in centers:
            new_set = code_set - {c}
            if is_covered(new_set):
                code_set.clear()
                code_set.update(new_set)
                return True
        return False
    
    # Local search: try swapping a center for another
    def try_swap(code_set):
        """Try to replace a center with another to maintain coverage with same size or reduce."""
        # This is expensive; do limited swaps
        if not code_set:
            return False
        centers = list(code_set)
        random.shuffle(centers)
        # Try a few random swaps
        for _ in range(50):
            c = random.choice(centers)
            # Try replacing c with a random other center
            new_c = random.randint(0, n_words - 1)
            if new_c in code_set:
                continue
            new_set = (code_set - {c}) | {new_c}
            if is_covered(new_set):
                code_set.clear()
                code_set.update(new_set)
                return True
        return False
    
    # Greedy improvement: repeatedly try to remove redundant centers
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Multiple passes of removal
        for _ in range(3):
            if try_remove(code_set):
                improved = True
        # Try some swaps to escape local minima
        if try_swap(code_set):
            improved = True
    
    # Final check
    if not is_covered(code_set):
        # Fallback to initial
        code_set = set(word_to_int(c) for c in code)
    
    return [int_to_word(c, 8) for c in sorted(code_set)]


def _general_greedy(q, n, radius, start_time, deadline):
    """
    General greedy set cover for covering codes.
    """
    n_words = q ** n
    
    def word_to_int(w):
        val = 0
        for d in w:
            val = val * q + d
        return val
    
    def int_to_word(val, n):
        w = []
        for _ in range(n):
            w.append(val % q)
            val //= q
        return w[::-1]
    
    # Precompute balls
    ball_cache = {}
    for w_int in range(n_words):
        w = int_to_word(w_int, n)
        ball = {w_int}
        # All words within Hamming distance <= radius
        # For radius=1: center + all single-position changes
        if radius == 1:
            for pos in range(n):
                for delta in range(1, q):
                    new_w = w[:]
                    new_w[pos] = (w[pos] + delta) % q
                    new_int = word_to_int(new_w)
                    ball.add(new_int)
        ball_cache[w_int] = ball
    
    # Greedy set cover
    uncovered = set(range(n_words))
    code_set = set()
    
    while uncovered and time.time() < deadline - 5:
        # Find center that covers most uncovered words
        best_center = None
        best_count = -1
        # Sample if too many centers
        candidates = list(range(n_words))
        random.shuffle(candidates)
        for c in candidates[:min(len(candidates), 200)]:
            count = len(ball_cache[c] & uncovered)
            if count > best_count:
                best_count = count
                best_center = c
        if best_center is None or best_count <= 0:
            break
        code_set.add(best_center)
        uncovered -= ball_cache[best_center]
    
    # Try to remove redundant centers
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        centers = list(code_set)
        random.shuffle(centers)
        for c in centers:
            new_set = code_set - {c}
            # Check coverage
            covered = set()
            for cc in new_set:
                covered.update(ball_cache[cc])
            if len(covered) == n_words:
                code_set = new_set
                improved = True
                break
    
    return [int_to_word(c, n) for c in sorted(code_set)]
