"""Equal-gcd-triple-free subsets of [1,128].

Model-and-optimize approach:
  * Precompute a full GCD matrix and per-element gcd-buckets (fast constraint model).
  * Build valid sets by greedy with 2-step look-ahead.
  * Optimize the exact objective |S| with simulated annealing + hill-climbing over the
    128-element inclusion vector, accepting moves by set size (penalizing violations).
  * Bounded 2-opt swaps and a final "add anything that fits" pass.
  * Strict wall-clock guards: always return the best valid set found so far.

Constraint: no three distinct a,b,c in S with gcd(a,b)=gcd(a,c)=gcd(b,c).
"""
import time
import math
import random
from itertools import combinations

_DEADLINE = 88.0          # wall-clock safety margin under the 90 s kill
_START = None
_N = 128
_CAND = list(range(1, _N + 1))
_PRE = None               # GCD[i][j] for i,j in 0..128
_BUCKETS = None           # BUCKETS[i] = dict g -> list of j with gcd(i,j)==g


def _ensure_pre():
    global _PRE, _BUCKETS
    if _PRE is not None:
        return
    PRE = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            PRE[i][j] = PRE[j][i] = math.gcd(i, j)
    _PRE = PRE

    BUCKETS = [None] * (_N + 1)
    for i in range(1, _N + 1):
        d = {}
        row = PRE[i]
        for j in range(1, _N + 1):
            g = row[j]
            d.setdefault(g, []).append(j)
        BUCKETS[i] = d
    _BUCKETS = BUCKETS


def _violations_mask(mask):
    """Number of equal-gcd triples in the set encoded by bit-mask."""
    P = _PRE
    B = _BUCKETS
    cnt = 0
    s = [i for i in range(1, _N + 1) if (mask >> (i - 1)) & 1]
    n = len(s)
    # Fast path: count via buckets
    # For each pair (a,b) with g=gcd(a,b), count c in S\{a,b} with gcd(a,c)==g and gcd(b,c)==g
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g = P[a][b]
            # c must satisfy gcd(a,c)==g and gcd(b,c)==g, c distinct from a,b
            ba = B[a].get(g, [])
            # count b? we iterate s, so count elements in s (not a,b) that are in ba and in B[b][g]
            # small sets; do a linear scan
            ccount = 0
            for k in range(i + 1, n):
                c = s[k]
                if P[a][c] == g and P[b][c] == g:
                    ccount += 1
            cnt += ccount
    return cnt


def _can_add(mask, x):
    """True if adding x (not in mask) creates no equal-gcd triple."""
    P = _PRE
    B = _BUCKETS
    if (mask >> (x - 1)) & 1:
        return False
    # gather, per g, the S elements y with gcd(x,y)==g
    Bx = B[x]
    for g, lst in Bx.items():
        inball = []
        for u in range(1, _N + 1):
            if u == x:
                continue
            if not ((mask >> (u - 1)) & 1):
                continue
            if P[x][u] == g:
                inball.append(u)
        m = len(inball)
        if m < 2:
            continue
        # need a pair (a,b) in inball with gcd(a,b)==g
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _swap_valid(mask, r, x):
    """True if (S - {r}) + {x} is valid (S currently valid)."""
    if (mask >> (x - 1)) & 1 or (mask >> (r - 1)) & 0 == 0 and (mask >> (r - 1)) & 1 == 0:
        pass
    # r must be in mask, x not in mask
    newmask = (mask & ~(1 << (r - 1))) | (1 << (x - 1))
    P = _PRE
    B = _BUCKETS
    Bx = B[x]
    for g, lst in Bx.items():
        inball = []
        for u in range(1, _N + 1):
            if u == r or u == x:
                continue
            if not ((newmask >> (u - 1)) & 1):
                continue
            if P[x][u] == g:
                inball.append(u)
        m = len(inball)
        if m < 2:
            continue
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _greedy_lookahead(order):
    mask = 0
    S = []
    for x in order:
        if _can_add(mask, x):
            mask |= 1 << (x - 1)
            S.append(x)
    return S, mask


def _local_add(mask, S):
    """Add any excluded element that fits, repeatedly."""
    P = _PRE
    changed = True
    S = list(S)
    while changed:
        changed = False
        for x in _CAND:
            if not ((mask >> (x - 1)) & 1) and _can_add(mask, x):
                mask |= 1 << (x - 1)
                S.append(x)
                changed = True
    return S, mask


def _popcount(mask):
    return bin(mask).count("1")


def _size_of(mask):
    return bin(mask).count("1")


def _anneal(start_mask, max_iters, seed):
    """Simulated annealing over the 128-bit inclusion vector.

    State: mask. Score = popcount - 1000*violations. We always track best valid mask.
    Moves: flip a random bit (add/remove), swap two bits, or multi-addition swap.
    """
    rng = random.Random(seed)
    P = _PRE
    B = _BUCKETS

    cur_mask = start_mask
    cur_viol = _violations_mask(cur_mask)
    cur_score = _size_of(cur_mask) - 1000 * cur_viol
    best_mask = cur_mask
    best_valid_size = _size_of(cur_mask) if cur_viol == 0 else -1
    if cur_viol == 0:
        best_mask = cur_mask
        best_valid_size = _size_of(cur_mask)

    # temperature schedule
    T0 = 2.0
    T1 = 0.02
    it = 0
    deadline = _DEADLINE - 8
    while it < max_iters and time.time() < deadline:
        T = T0 * (T1 / T0) ** (it / max(1, max_iters))
        it += 1
        # propose move
        op = rng.random()
        if op < 0.35:
            # flip a random bit
            pos = rng.randrange(1, _N + 1)
            newmask = cur_mask ^ (1 << (pos - 1))
        elif op < 0.65:
            # swap: remove one in-set, add one out-of-set
            inpos = []
            outpos = []
            for i in range(1, _N + 1):
                if (cur_mask >> (i - 1)) & 1:
                    inpos.append(i)
                else:
                    outpos.append(i)
            if not inpos or not outpos:
                continue
            r = rng.choice(inpos)
            a = rng.choice(outpos)
            newmask = (cur_mask & ~(1 << (r - 1))) | (1 << (a - 1))
        elif op < 0.85:
            # 2-swap: remove two in-set, add two out-of-set
            inpos = []
            outpos = []
            for i in range(1, _N + 1):
                if (cur_mask >> (i - 1)) & 1:
                    inpos.append(i)
                else:
                    outpos.append(i)
            if len(inpos) < 2 or len(outpos) < 2:
                continue
            r1, r2 = rng.sample(inpos, 2)
            a1, a2 = rng.sample(outpos, 2)
            newmask = (cur_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))) | (1 << (a1 - 1)) | (1 << (a2 - 1))
        else:
            # multi-addition swap: remove 1-2 in-set, add 2-3 out-of-set
            inpos = []
            outpos = []
            for i in range(1, _N + 1):
                if (cur_mask >> (i - 1)) & 1:
                    inpos.append(i)
                else:
                    outpos.append(i)
            if not inpos or len(outpos) < 2:
                continue
            n_rem = rng.choice([1, 2]) if len(inpos) >= 2 else 1
            n_add = rng.choice([2, 3]) if len(outpos) >= 3 else min(2, len(outpos))
            if len(inpos) < n_rem or len(outpos) < n_add:
                continue
            rems = rng.sample(inpos, n_rem)
            adds = rng.sample(outpos, n_add)
            newmask = cur_mask
            for r in rems:
                newmask &= ~(1 << (r - 1))
            for a in adds:
                newmask |= (1 << (a - 1))

        new_viol = _violations_mask(newmask)
        new_score = _size_of(newmask) - 1000 * new_viol
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / max(1e-9, T)):
            cur_mask = newmask
            cur_score = new_score
            cur_viol = new_viol
            if cur_viol == 0:
                if _size_of(cur_mask) > best_valid_size:
                    best_valid_size = _size_of(cur_mask)
                    best_mask = cur_mask
    return best_mask, best_valid_size


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()

    P = _PRE

    # Prime-power / divisor heuristics for ordering
    def num_divisors(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c

    divs = {x: num_divisors(x) for x in range(1, _N + 1)}

    def is_prime_power(x):
        if x < 2:
            return False
        v = x
        f0 = None
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    return False
                while v % p == 0:
                    v //= p
                break
            p += 1
        return True

    prime_powers = [x for x in range(1, _N + 1) if is_prime_power(x)]

    orders = [
        list(range(1, _N + 1)),
        list(range(_N, 0, -1)),
        sorted(range(1, _N + 1), key=lambda x: divs[x]),
        sorted(range(1, _N + 1), key=lambda x: divs[x], reverse=True),
        prime_powers + [x for x in range(1, _N + 1) if x not in prime_powers],
        sorted(range(1, _N + 1), key=lambda x: (0 if is_prime_power(x) else 1, x)),
        sorted(range(1, _N + 1), key=lambda x: (-divs[x], x)),
        sorted(range(1, _N + 1), key=lambda x: (divs[x], x)),
    ]

    best_mask = 0
    best_size = 0
    best_list = []

    for order in orders:
        if time.time() > _DEADLINE - 20:
            break
        S, mask = _greedy_lookahead(order)
        S, mask = _local_add(mask, S)
        if len(S) > best_size:
            best_size = len(S)
            best_mask = mask
            best_list = list(S)

    # Annealing from several starts, climbing the exact objective
    starts = []
    # start from best greedy
    starts.append(best_mask)
    # start from a couple of other greedy orders
    for order in orders[:4]:
        S, mask = _greedy_lookahead(order)
        S, mask = _local_add(mask, S)
        starts.append(mask)
        if len(S) > best_size:
            best_size = len(S)
            best_mask = mask
            best_list = list(S)

    seed = 999
    for sm in starts:
        if time.time() > _DEADLINE - 12:
            break
        bmask, bsize = _anneal(sm, max_iters=25000, seed=seed)
        seed += 1
        if bsize > best_size:
            best_size = bsize
            best_mask = bmask

    # Final add pass on best_mask
    S = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    S, best_mask = _local_add(best_mask, S)
    best_size = len(S)

    # Bounded 2-opt: try removing a pair and adding a different pair
    if time.time() < _DEADLINE - 6:
        improved = True
        rounds = 0
        while improved and rounds < 8:
            improved = False
            rounds += 1
            Slist = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
            Sset = set(Slist)
            for r1, r2 in combinations(Slist, 2):
                if time.time() > _DEADLINE - 6:
                    break
                base = (best_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1)))
                # try adding pairs (a,b) not in set
                for a in _CAND:
                    if a in Sset or a == r1 or a == r2:
                        continue
                    for b in _CAND:
                        if a >= b:
                            continue
                        if b in Sset or b == r1 or b == r2:
                            continue
                        newmask = base | (1 << (a - 1)) | (1 << (b - 1))
                        if _violations_mask(newmask) == 0:
                            # same size; but maybe enables more adds
                            tlist = [i for i in range(1, _N + 1) if (newmask >> (i - 1)) & 1]
                            tlist, newmask = _local_add(newmask, tlist)
                            if len(tlist) > best_size:
                                best_size = len(tlist)
                                best_mask = newmask
                                improved = True
                                break
                    if improved or time.time() > _DEADLINE - 6:
                        break
                if improved:
                    break
            if time.time() > _DEADLINE - 6:
                break

    # Final add pass again
    S = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    S, best_mask = _local_add(best_mask, S)
    best_size = len(S)

    # Polish pass: try removing each element and re-adding optimally
    if time.time() < _DEADLINE - 10:
        Slist = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
        for rem in Slist:
            if time.time() > _DEADLINE - 10:
                break
            reduced_mask = best_mask & ~(1 << (rem - 1))
            # Try to add any excluded element that fits
            improved_mask = reduced_mask
            improved = True
            while improved:
                improved = False
                for x in _CAND:
                    if time.time() > _DEADLINE - 10:
                        break
                    if not ((improved_mask >> (x - 1)) & 1) and _can_add(improved_mask, x):
                        improved_mask |= (1 << (x - 1))
                        improved = True
            if _violations_mask(improved_mask) == 0:
                im_size = _size_of(improved_mask)
                if im_size > best_size:
                    best_size = im_size
                    best_mask = improved_mask

    # Final add pass after polish
    S = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    S, best_mask = _local_add(best_mask, S)
    best_size = len(S)

    # Safety: ensure validity; fallback to guaranteed-valid greedy if needed
    if _violations_mask(best_mask) != 0:
        S, mask = _greedy_lookahead(list(range(1, _N + 1)))
        S, mask = _local_add(mask, S)
        best_mask = mask
        best_size = len(S)

    best_list = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    if not best_list:
        best_list = [2]
    return best_list


def construct():
    from itertools import combinations  # noqa: F401
    from math import gcd  # noqa: F401
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        P = _PRE
        res = list(range(1, _N + 1))
        # remove elements causing violations until valid
        while _violations_mask_mask(res) != 0 and len(res) > 1:
            worst = res[0]
            worst_cnt = -1
            for cand in res:
                others = [y for y in res if y != cand]
                cnt = 0
                for a, b in combinations(others, 2):
                    if P[cand][a] == P[cand][b] == P[a][b]:
                        cnt += 1
                if cnt > worst_cnt:
                    worst_cnt = cnt
                    worst = cand
            res.remove(worst)
        if not res:
            res = [2]
    return res


def _violations_mask_mask(lst):
    mask = 0
    for x in lst:
        mask |= 1 << (x - 1)
    return _violations_mask(mask)
