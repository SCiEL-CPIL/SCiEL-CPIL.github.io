import math
import time
import random
import itertools

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _hill_climb(U, max_iter=5000, time_limit=2.0):
    """Local search: try adding/removing/swapping elements."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = list(U)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        if random.random() < 0.3:
            # Try adding a random element
            mx = max(U)
            new_elem = random.randint(1, max(2 * mx, 10))
            if new_elem not in U:
                candidate = sorted(U + [new_elem])
                s = _score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
        elif random.random() < 0.5:
            # Try removing a random element (not 0)
            removable = [x for x in U if x != 0]
            if removable:
                rem = random.choice(removable)
                candidate = [x for x in U if x != rem]
                s = _score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
        else:
            # Try replacing a random element (not 0)
            replaceable = [x for x in U if x != 0]
            if replaceable:
                old = random.choice(replaceable)
                mx = max(U)
                new = random.randint(1, max(2 * mx, 10))
                if new != old and new not in U:
                    candidate = sorted([x if x != old else new for x in U])
                    s = _score(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = candidate
    return best_U, best_score

def _lacunary(n, base):
    """Lacunary sequence: 0, 1, base, base^2, ..."""
    U = [0, 1]
    for i in range(2, n):
        U.append(base ** (i - 1))
    return U

def _geometric_int(n, ratio):
    """Geometric progression with integer ratio."""
    U = [0, 1]
    for i in range(2, n):
        U.append(ratio ** (i - 1))
    return U

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 2, 4]
    best_score = _score(best_U)
    
    # Try known good starting points
    candidates = [
        [0, 1, 2, 4],
        [0, 1, 3, 7, 15],  # 2^k - 1
        [0, 1, 2, 4, 8, 16],  # powers of 2
        [0, 1, 3, 9, 27],  # powers of 3
        [0, 1, 4, 16, 64],  # powers of 4
        [0, 1, 2, 5, 10, 20],
        [0, 1, 3, 6, 10, 15],  # triangular numbers
    ]
    
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Try lacunary sequences with different bases
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        for n in range(3, 12):
            if time.time() > deadline - 30:
                break
            cand = _lacunary(n, base)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_U = list(cand)
    
    # Try geometric with various ratios
    for ratio in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for n in range(3, 12):
            if time.time() > deadline - 30:
                break
            cand = _geometric_int(n, ratio)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_U = list(cand)
    
    # Exhaustive search for small sets (size up to 5)
    if time.time() < deadline - 40:
        # Try all subsets of {0,1,2,...,12} of size 3-5 containing 0
        max_val = 12
        for size in range(3, 6):
            for subset in itertools.combinations(range(1, max_val + 1), size - 1):
                if time.time() > deadline - 30:
                    break
                U = sorted([0] + list(subset))
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    # Hill climbing from best and several random starts
    if time.time() < deadline - 20:
        # Hill climb from current best
        U_climbed, s_climbed = _hill_climb(best_U, max_iter=3000, time_limit=15)
        if s_climbed > best_score:
            best_score = s_climbed
            best_U = U_climbed
        
        # Hill climb from random starts
        random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            # Random start: pick 3-7 elements
            n_elem = random.randint(3, 7)
            elems = random.sample(range(1, 20), n_elem - 1)
            U_start = sorted([0] + elems)
            U_climbed, s_climbed = _hill_climb(U_start, max_iter=2000, time_limit=10)
            if s_climbed > best_score:
                best_score = s_climbed
                best_U = U_climbed
    
    # Final: ensure 0 is in the set and it's valid
    if 0 not in best_U:
        best_U = [0] + sorted(best_U)
    
    return sorted(set(best_U))
