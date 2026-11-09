"""Optimized search for finite set U maximizing sumset/difference-set exponent bound."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _set_sum_diff(U):
    """Return (|U+U|, |U-U|) for a finite set U of ints."""
    Us = list(U)
    n = len(Us)
    plus = set()
    minus = set()
    for i in range(n):
        a = Us[i]
        for j in range(i, n):
            b = Us[j]
            plus.add(a + b)
        for j in range(n):
            b = Us[j]
            minus.add(a - b)
    return len(plus), len(minus)

def _score(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def _lacunary(base, n):
    """Lacunary sequence: base^0, base^1, ..., base^(n-1)."""
    return [base**k for k in range(n)]

def _geometric(ratio, n):
    """Geometric-like: 0, 1, ratio, ratio^2, ..."""
    res = [0, 1]
    for k in range(2, n):
        res.append(res[-1] * ratio)
    return res

def _arithmetic(start, step, n):
    """Arithmetic progression starting at 0."""
    return [start + i * step for i in range(n)]

def _random_set(rng, n, maxv):
    """Random set containing 0."""
    vals = rng.sample(range(1, maxv + 1), min(n - 1, maxv))
    return [0] + sorted(vals)

def _scaled_lacunary(base, n, scale):
    """Scaled lacunary: scale * base^k."""
    return [scale * base**k for k in range(n)]

def _hybrid(rng, base, n1, n2, maxv):
    """Hybrid: first part lacunary, second part random."""
    first = _lacunary(base, n1)
    rest_max = max(first[-1] * 2, maxv) if first else maxv
    rest = rng.sample(range(1, rest_max + 1), n2)
    return sorted(set([0] + first + rest))

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = random.Random(42)

    best_val = -1e9
    best_set = [0, 1, 3]

    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # Structured families - lacunary sequences with various bases
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        for n in range(2, 25):
            U = _lacunary(base, n)
            consider(U)

    # Geometric sequences
    for ratio in [2, 3, 4, 5, 6, 8]:
        for n in range(3, 20):
            U = _geometric(ratio, n)
            consider(U)

    # Arithmetic progressions
    for step in [2, 3, 4, 5, 7, 10, 15, 20]:
        for n in range(3, 30):
            U = _arithmetic(0, step, n)
            consider(U)

    # Scaled lacunary
    for base in [2, 3, 4, 5]:
        for scale in [1, 2, 3, 5, 8]:
            for n in range(2, 18):
                U = _scaled_lacunary(base, n, scale)
                consider(U)

    # Triangular numbers
    for n in range(2, 25):
        U = [k*(k+1)//2 for k in range(n)]
        consider(U)

    # Quadratic
    for n in range(2, 25):
        U = [k*k for k in range(n)]
        consider(U)

    # Fibonacci
    fibs = [0, 1]
    for k in range(2, 30):
        fibs.append(fibs[-1] + fibs[-2])
    for n in range(2, 20):
        consider(fibs[:n])

    # Hybrid constructions
    for _ in range(2000):
        if time.time() > deadline:
            break
        base = rng.choice([2, 3, 4, 5, 6, 7])
        n1 = rng.randint(2, 8)
        n2 = rng.randint(1, 10)
        maxv = rng.choice([50, 100, 200, 500, 1000])
        U = _hybrid(rng, base, n1, n2, maxv)
        consider(U)

    # Random sets with various sizes and ranges
    for _ in range(5000):
        if time.time() > deadline:
            break
        n = rng.randint(3, 20)
        maxv = rng.choice([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        U = _random_set(rng, n, maxv)
        consider(U)

    # Local search with multiple simultaneous changes
    cur = list(best_set)
    improved = True
    it = 0
    while improved and time.time() < deadline - 2.0 and it < 800:
        improved = False
        it += 1
        mx = max(cur) if cur else 0
        n_cur = len(cur)
        
        # Generate diverse candidates for addition
        cands_add = set()
        # Powers and multiples
        for c in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]:
            cands_add.add(c)
        # Relative to max
        cands_add.add(mx + 1)
        cands_add.add(mx * 2 + 1)
        cands_add.add(mx * 3 + 1)
        cands_add.add(mx * 2 - 1)
        cands_add.add(mx * 2)
        cands_add.add(mx + mx // 2)
        cands_add.add(mx + mx // 3)
        cands_add.add(max(1, mx // 2))
        cands_add.add(max(1, mx // 3))
        cands_add.add(max(1, mx // 5))
        # Gaps based on existing structure
        if len(cur) >= 2:
            sorted_cur = sorted(cur)
            for i in range(len(sorted_cur) - 1):
                gap = sorted_cur[i+1] - sorted_cur[i]
                if gap > 1:
                    cands_add.add(sorted_cur[i] + gap // 2)
                    cands_add.add(sorted_cur[i] + gap - 1)
        # Random candidates
        for _ in range(30):
            cands_add.add(rng.randint(1, max(2, mx * 4)))
        cands_add = [c for c in cands_add if c >= 0 and c not in cur]
        rng.shuffle(cands_add)
        
        # Try additions
        for c in cands_add:
            U = cur + [c]
            v, s = _score(U)
            if v > best_val + 1e-12:
                best_val = v
                best_set = s
                cur = s
                improved = True
                break
        
        # Try removals if no improvement
        if not improved:
            for c in list(cur):
                if c == 0:
                    continue
                U = [x for x in cur if x != c]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break
        
        # Try swaps if still no improvement
        if not improved and len(cur) > 2:
            for idx in range(1, len(cur)):
                old_val = cur[idx]
                # Try replacing with nearby values
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 8, 13]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in cur:
                        continue
                    U = list(cur)
                    U[idx] = new_val
                    U = sorted(set(U))
                    if 0 not in U:
                        U.append(0)
                    v, s = _score(U)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_set = s
                        cur = s
                        improved = True
                        break
                if improved:
                    break

    # Final random perturbation search
    if time.time() < deadline - 1.0:
        for _ in range(3000):
            if time.time() > deadline - 0.5:
                break
            # Perturb best set
            cur = list(best_set)
            op = rng.choice(['add', 'remove', 'swap', 'add2', 'remove2'])
            if op == 'add':
                c = rng.randint(1, max(2, max(cur) * 3))
                if c not in cur:
                    cur.append(c)
            elif op == 'add2':
                # Add two elements
                c1 = rng.randint(1, max(2, max(cur) * 2))
                c2 = rng.randint(1, max(2, max(cur) * 3))
                cur.extend([c1, c2])
                cur = sorted(set(cur))
            elif op == 'remove' and len(cur) > 2:
                idx = rng.randint(1, len(cur) - 1)
                cur.pop(idx)
            elif op == 'remove2' and len(cur) > 3:
                # Remove two elements
                idx1 = rng.randint(1, len(cur) - 1)
                idx2 = rng.randint(1, len(cur) - 1)
                if idx1 != idx2:
                    cur.pop(max(idx1, idx2))
                    cur.pop(min(idx1, idx2))
            elif op == 'swap' and len(cur) > 2:
                idx = rng.randint(1, len(cur) - 1)
                cur[idx] = rng.randint(1, max(2, max(cur) * 3))
                cur = sorted(set(cur))
                if 0 not in cur:
                    cur.append(0)
            consider(cur)

    return best_set
# EVOLVE-BLOCK-END
