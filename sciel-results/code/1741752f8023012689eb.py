"""Term-minimal smooth-antichain representation with optimized antichain for 2-term coverage."""
import time
from itertools import combinations

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(vals)

def _incompatible(a, b):
    if a == b:
        return True
    if b % a == 0 or a % b == 0:
        return True
    return False

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)

# Precompute incompatibility matrix
_INCOMP = [[False] * _N for _ in range(_N)]
for i in range(_N):
    for j in range(i + 1, _N):
        if _SMOOTH[i] % _SMOOTH[j] == 0 or _SMOOTH[j] % _SMOOTH[i] == 0:
            _INCOMP[i][j] = True
            _INCOMP[j][i] = True

def _build_optimal_antichain(deadline):
    """Build antichain maximizing 2-term coverage of non-smooth targets 1..128."""
    # Targets that need representation (non-smooth)
    targets = [n for n in range(2, 129) if n not in _SMOOTH_SET]
    
    # For each pair in smooth numbers, check if they form an antichain pair
    # and which targets they cover
    # We want to select a subset of smooth numbers forming an antichain
    # that maximizes the number of targets coverable by 2-term sums
    
    # Strategy: use local search / hill climbing
    # Start with a reasonable antichain, then try swaps to improve coverage
    
    # Initial: greedy by descending value
    antichain = []
    for v in sorted(_SMOOTH, reverse=True):
        ok = True
        for u in antichain:
            if _incompatible(u, v):
                ok = False
                break
        if ok:
            antichain.append(v)
    antichain_set = set(antichain)
    
    def count_two_term_coverage(ac_set):
        """Count how many targets can be represented as 2-term sum from ac_set."""
        count = 0
        for n in targets:
            found = False
            for a in ac_set:
                if a >= n:
                    break
                b = n - a
                if b in ac_set and b != a:
                    found = True
                    break
            if found:
                count += 1
        return count
    
    # Local search: try adding/removing/swapping elements
    best_count = count_two_term_coverage(antichain_set)
    improved = True
    start = time.time()
    
    while improved and time.time() - start < deadline:
        improved = False
        # Try adding each smooth number not in antichain
        for v in _SMOOTH:
            if v in antichain_set:
                continue
            # Check if v is compatible with all in antichain
            ok = True
            for u in antichain:
                if _incompatible(u, v):
                    ok = False
                    break
            if not ok:
                continue
            new_set = antichain_set | {v}
            new_count = count_two_term_coverage(new_set)
            if new_count > best_count:
                antichain.append(v)
                antichain_set = new_set
                best_count = new_count
                improved = True
                break
        
        # Try removing each element and see if we can add two others
        for i, v in enumerate(antichain):
            if len(antichain) <= 2:
                continue
            trial_set = antichain_set - {v}
            # Try to add back the best compatible element
            for w in _SMOOTH:
                if w in trial_set:
                    continue
                ok = True
                for u in trial_set:
                    if _incompatible(u, w):
                        ok = False
                        break
                if not ok:
                    continue
                new_set = trial_set | {w}
                new_count = count_two_term_coverage(new_set)
                if new_count > best_count:
                    antichain[i] = w
                    antichain_set = new_set
                    best_count = new_count
                    improved = True
                    break
            if improved:
                break
    
    return sorted(antichain), best_count

def _find_two_term(n, ac, ac_set):
    for a in ac:
        if a >= n:
            break
        b = n - a
        if b == a or b not in ac_set:
            continue
        if _incompatible(a, b):
            continue
        return [a, b]
    return None

def _find_three_term(n, ac, ac_set):
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a > n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if b >= rem1:
                break
            if _incompatible(a, b):
                continue
            c = rem1 - b
            if c <= b or c not in ac_set:
                continue
            if _incompatible(a, c) or _incompatible(b, c):
                continue
            return [a, b, c]
    return None

def _find_four_term(n, ac, ac_set):
    """Find 4-term antichain sum."""
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a > n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if a + b >= rem1:
                break
            if _incompatible(a, b):
                continue
            rem2 = rem1 - b
            for k in range(j + 1, L):
                c = ac[k]
                if c >= rem2:
                    break
                if _incompatible(a, c) or _incompatible(b, c):
                    continue
                d = rem2 - c
                if d <= c or d not in ac_set:
                    continue
                if _incompatible(a, d) or _incompatible(b, d) or _incompatible(c, d):
                    continue
                return [a, b, c, d]
    return None

def _greedy_fill(n):
    """Find any valid div-free smooth representation."""
    if n in _SMOOTH_SET:
        return [n]
    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b == a or b not in _SMOOTH_SET:
            continue
        if _incompatible(a, b):
            continue
        return [a, b]
    for i in range(_N):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b or c not in _SMOOTH_SET:
                continue
            if _incompatible(a, b) or _incompatible(a, c) or _incompatible(b, c):
                continue
            return [a, b, c]
    # 4 terms as last resort
    for i in range(_N):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b > n:
                break
            for k in range(j + 1, _N):
                c = _SMOOTH[k]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d <= c or d not in _SMOOTH_SET:
                    continue
                if (_incompatible(a, b) or _incompatible(a, c) or _incompatible(a, d) or
                    _incompatible(b, c) or _incompatible(b, d) or _incompatible(c, d)):
                    continue
                return [a, b, c, d]
    return [n]

def construct():
    start = time.time()
    DEADLINE = 82.0
    
    # Build optimized antichain with time budget
    antichain, coverage = _build_optimal_antichain(min(30.0, DEADLINE - (time.time() - start)))
    ac_set = set(antichain)
    
    answers = [None] * 129
    total_terms = 0
    
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
        
        if n in _SMOOTH_SET:
            answers[n] = [n]
            total_terms += 1
            continue
        
        found = _find_two_term(n, antichain, ac_set)
        if found is not None:
            answers[n] = found
            total_terms += 2
            continue
        
        found = _find_three_term(n, antichain, ac_set)
        if found is not None:
            answers[n] = found
            total_terms += 3
            continue
        
        found = _find_four_term(n, antichain, ac_set)
        if found is not None:
            answers[n] = found
            total_terms += 4
            continue
        
        # Fallback
        answers[n] = _greedy_fill(n)
        total_terms += len(answers[n])
    
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _greedy_fill(n)
    
    return answers[1:129]
