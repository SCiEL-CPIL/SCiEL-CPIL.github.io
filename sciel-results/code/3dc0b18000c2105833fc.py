import time
import math

_N = 5040
_T = 4979

def _all_divisors(m):
    out = []
    for i in range(1, int(m ** 0.5) + 1):
        if m % i == 0:
            out.append(i)
            if i != m // i:
                out.append(m // i)
    return out

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before 90 s kill

    n = 5040
    target = 4979

    divs = _all_divisors(n)
    # divisors >= target can never be used in a positive-sum representation
    pool = [d for d in divs if d <= target]
    # largest first -> greedy upper bound is tight and pruning is aggressive
    pool.sort(reverse=True)
    m = len(pool)

    # suffix sums for a feasibility upper bound
    suff = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suff[i] = suff[i + 1] + pool[i]

    # quick greedy upper bound
    rem = target
    ub = 0
    for d in pool:
        if d <= rem:
            rem -= d
            ub += 1
            if rem == 0:
                break
    if rem != 0:
        # greedy with largest-first is not guaranteed; fall back to any bound
        ub = m

    best = [ub]
    bestset = [set()]

    # iterative deepening on number of terms (A* / IDA* style)
    for depth in range(1, ub + 1):
        # lower bound on additional terms needed from state (i, rem):
        #  rem <= suff[i]  (feasibility)  AND  rem <= depth_remaining * pool[i] (max value per term)
        def feasible(i, rem, terms_left):
            if rem == 0:
                return terms_left >= 0
            if i >= m or rem < 0 or terms_left < 0:
                return False
            if rem > suff[i]:
                return False
            if terms_left > 0 and rem > terms_left * pool[i]:
                return False
            return True

        # DFS with branch and bound, choosing terms in a fixed order
        chosen = []

        def dfs(i, rem, terms_left):
            if time.time() > DEADLINE:
                return True  # signal abort
            if rem == 0:
                if terms_left >= 0:
                    cur = len(chosen)
                    if cur < best[0]:
                        best[0] = cur
                        bestset[0] = set(chosen[:])
                return False
            if i >= m or rem < 0 or terms_left < 0:
                return False
            # bound 1: cannot reach rem with what remains
            if rem > suff[i]:
                return False
            # bound 2: even taking terms_left largest available cannot reach rem
            if terms_left > 0 and rem > terms_left * pool[i]:
                return False
            # bound 3: need at least ceil(rem / pool[i]) terms
            min_need = (rem + pool[i] - 1) // pool[i]
            if min_need > terms_left:
                return False
            # bound 4: must still be able to hit exactly (skip-prune)
            # if rem not representable by distinct pool[i:] of size <= terms_left:
            #   (handled by feasibility above via suff)
            # also, cannot use more than remaining items
            if terms_left > m - i:
                terms_left = m - i

            for j in range(i, m):
                d = pool[j]
                if d > rem:
                    continue
                # prune: even taking this and (terms_left-1) more largest can't beat? no, we want exact
                if not feasible(j + 1, rem - d, terms_left - 1):
                    # since pool sorted desc, later j' > j have smaller d, so rem-d' larger,
                    # suff smaller, terms same -> still infeasible; break
                    break
                chosen.append(d)
                if dfs(j + 1, rem - d, terms_left - 1):
                    chosen.pop()
                    return True
                chosen.pop()
            return False

        if dfs(0, target, depth):
            break  # aborted by deadline
        if best[0] <= depth:
            # found a solution at this depth; deeper is worse, stop
            break

    if bestset[0]:
        return list(bestset[0])

    # fallback: return best effort greedy if any (should not happen)
    rem = target
    out = []
    for d in pool:
        if d <= rem:
            out.append(d)
            rem -= d
            if rem == 0:
                break
    return out
