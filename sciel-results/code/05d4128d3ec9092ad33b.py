import time
import random
import math

_DEADLINE = 85.0

def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    
    # Precompute prime factors for 1..256
    pf = [set() for _ in range(257)]
    pf[1] = set()
    for n in range(2, 257):
        pf[n] = _factorize(n)

    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))

    def eval_sums(A, B):
        """Return (valid, count) for distinct sums being pairwise coprime."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if pf[slist[i]] & pf[slist[j]]:
                    return False, 0
        return True, len(slist)

    best = 0
    best_AB = [[1, 7, 13], [4, 10, 16]]

    # Focused random search with parity constraint
    # Max possible is 9 distinct sums (3x3 grid)
    for trial in range(500000):
        if time.time() - start_time > _DEADLINE:
            break
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        valid, cnt = eval_sums(A, B)
        if valid and cnt > best:
            best = cnt
            best_AB = [A, B]
            if best == 9:
                return best_AB

    # Also try even-odd
    for trial in range(500000):
        if time.time() - start_time > _DEADLINE:
            break
        A = random.sample(evens, 3)
        B = random.sample(odds, 3)
        valid, cnt = eval_sums(A, B)
        if valid and cnt > best:
            best = cnt
            best_AB = [A, B]
            if best == 9:
                return best_AB

    # Local search to try to reach 9
    if best < 9:
        A = list(best_AB[0])
        B = list(best_AB[1])
        cur = best
        for step in range(200000):
            if time.time() - start_time > _DEADLINE:
                break
            improved = False
            # Try changing one element
            for which in (0, 1):
                if improved:
                    break
                S = A if which == 0 else B
                pool = odds if S[0] % 2 == 1 else evens
                for idx in range(3):
                    if improved:
                        break
                    old = S[idx]
                    for new in random.sample(pool, min(50, len(pool))):
                        if new == old:
                            continue
                        S[idx] = new
                        if len(set(S)) < 3:
                            S[idx] = old
                            continue
                        valid, cnt = eval_sums(A, B)
                        if valid and cnt > cur:
                            cur = cnt
                            if cnt > best:
                                best = cnt
                                best_AB = [list(A), list(B)]
                            improved = True
                            if best == 9:
                                return best_AB
                            break
                        S[idx] = old
            if not improved and random.random() < 0.1:
                A = random.sample(odds, 3)
                B = random.sample(evens, 3)
                valid, cur = eval_sums(A, B)
                if not valid:
                    cur = -1

    return best_AB
