"""Large integer sets avoiding equal-gcd triples.

Strategy:
1. Start with all odd numbers in [1,128] (64 elements) - structurally avoids
   many equal-gcd triples since odd numbers have limited shared factors.
2. Greedily add even numbers that don't create bad triples.
3. Local search: try swapping out elements to add more.
4. Deadline guard to stay within 90s wall-clock.
"""

# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 88.0  # safety margin under 90s wall-clock kill


def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    g_ab = math.gcd(a, b)
    g_ac = math.gcd(a, c)
    g_bc = math.gcd(b, c)
    return g_ab == g_ac == g_bc


def _is_valid(S):
    """Check if set S has no bad triples."""
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            for k in range(j + 1, len(S_list)):
                if _bad_triple(S_list[i], S_list[j], S_list[k]):
                    return False
    return True


def _conflict_with_new(S, x):
    """Check if adding x to S creates any bad triple.
    A bad triple involving x and two existing elements a,b in S:
    gcd(x,a) == gcd(x,b) == gcd(a,b)
    """
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            g_ab = math.gcd(a, b)
            g_xa = math.gcd(x, a)
            if g_xa != g_ab:
                continue
            g_xb = math.gcd(x, b)
            if g_xb == g_ab:
                return True
    return False


def _conflict_count(S, x):
    """Count number of bad triples that would be created by adding x to S."""
    count = 0
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            g_ab = math.gcd(a, b)
            g_xa = math.gcd(x, a)
            if g_xa != g_ab:
                continue
            g_xb = math.gcd(x, b)
            if g_xb == g_ab:
                count += 1
    return count


def _precompute_gcd_matrix(n=128):
    """Precompute gcd for all pairs in [1, n]."""
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = math.gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    return gcd_mat


def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    random.seed(42)
    
    def within_deadline():
        return (time.time() - _START_TIME) < _DEADLINE - 2.0
    
    # Precompute gcd matrix for speed
    gcd_mat = _precompute_gcd_matrix(128)
    
    def _bad_triple_fast(a, b, c):
        """Fast bad triple check using precomputed gcd matrix."""
        return gcd_mat[a][b] == gcd_mat[a][c] == gcd_mat[b][c]
    
    full = list(range(1, 129))
    best_set = None
    best_size = 0
    
    # ---- Strategy 1: Start with odd numbers, greedily add evens ----
    def strategy_odd_plus_evens():
        S = set(range(1, 129, 2))  # All odd numbers: 64 elements
        evens = list(range(2, 129, 2))
        random.shuffle(evens)
        
        # Sort evens by how few conflicts they'd create
        # Try adding each even if it doesn't create a bad triple
        added = []
        for e in evens:
            if not within_deadline():
                break
            if not _conflict_with_new(S, e):
                S.add(e)
                added.append(e)
        
        return S
    
    # ---- Strategy 2: Greedy add with random ordering (multiple tries) ----
    def strategy_greedy_add(order_seed=0):
        random.seed(order_seed)
        S = set()
        remaining = list(full)
        random.shuffle(remaining)
        for x in remaining:
            if not within_deadline():
                break
            if not _conflict_with_new(S, x):
                S.add(x)
        return S
    
    # ---- Strategy 3: Local search - swap operations ----
    def strategy_local_search(initial_set, max_iters=5000):
        """Try to improve by swapping one element out for one in."""
        S = set(initial_set)
        S_list = list(S)
        improved = True
        iters = 0
        
        while improved and within_deadline() and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removing each element and see if we can add a better one
            for x in list(S):
                if not within_deadline():
                    break
                S.remove(x)
                # Try to add any element not in S
                for y in full:
                    if y in S:
                        continue
                    if not _conflict_with_new(S, y):
                        S.add(y)
                        improved = True
                        break
                if not improved:
                    S.add(x)  # put it back
                else:
                    break  # restart the outer loop with new S
        
        return S
    
    # ---- Strategy 4: Structured - numbers with distinct prime factor patterns ----
    def strategy_structured():
        """
        Key structural insight: if all elements are of the form p^k (prime powers),
        then gcd(a,b) is either 1 or a prime power. For three prime powers,
        the pairwise gcds being all equal requires specific structure.
        
        Let's try: all numbers that are powers of 2, plus all odd primes, plus 1.
        Actually, let's be more careful.
        
        A better structural approach: take all numbers whose prime factorization
        has at most one prime factor (prime powers), plus 1.
        """
        S = {1}
        for p in range(2, 129):
            if all(p % i != 0 for i in range(2, int(p**0.5) + 1)):
                # p is prime, add all powers of p up to 128
                val = p
                while val <= 128:
                    S.add(val)
                    val *= p
        return S
    
    # Run strategies
    # Strategy 1 first (strong structural start)
    S1 = strategy_odd_plus_evens()
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
    
    # Strategy 4 (structured)
    if within_deadline():
        S4 = strategy_structured()
        if len(S4) > best_size:
            best_size = len(S4)
            best_set = S4
    
    # Strategy 2 (greedy add, multiple seeds)
    if within_deadline():
        for seed in range(30):
            S2 = strategy_greedy_add(seed)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2
    
    # Strategy 3: Local search on the best set found so far
    if within_deadline() and best_set is not None:
        S3 = strategy_local_search(best_set, max_iters=10000)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = S3
    
    # Additional local search passes with different randomizations
    if within_deadline() and best_set is not None:
        for _ in range(5):
            if not within_deadline():
                break
            S3b = strategy_local_search(best_set, max_iters=5000)
            if len(S3b) > best_size:
                best_size = len(S3b)
                best_set = S3b
    
    # Final verification
    if best_set is None or len(best_set) == 0:
        return [1]
    
    # Verify validity
    S_list = list(best_set)
    valid = True
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            for k in range(j + 1, len(S_list)):
                if _bad_triple_fast(S_list[i], S_list[j], S_list[k]):
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fall back: find a valid subset by removing elements from bad triples
        S = set(best_set)
        while True:
            bad_found = False
            for a, b, c in combinations(sorted(S), 3):
                if _bad_triple_fast(a, b, c):
                    S.remove(a)
                    bad_found = True
                    break
            if not bad_found:
                break
        if len(S) > 0:
            best_set = S
        else:
            best_set = {1}
    
    return list(best_set)

# EVOLVE-BLOCK-END
