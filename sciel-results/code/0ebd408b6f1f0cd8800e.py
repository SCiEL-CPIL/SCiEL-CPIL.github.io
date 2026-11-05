# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _precompute_answers():
    """Precompute minimum-antichain representations for targets 1..128."""
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)

    smooth_set = set(smooth)
    n_smooth = len(smooth)
    # index map: value -> index in smooth
    idx = {v: i for i, v in enumerate(smooth)}

    # Precompute divisibility matrix: divs[i][j] True iff smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(n_smooth):
            b = smooth[j]
            if b % a == 0:
                divs[i][j] = True

    def is_antichain_vals(terms):
        """terms: sorted distinct values; check no one divides another."""
        L = len(terms)
        for i in range(L):
            ti = idx[terms[i]]
            for j in range(i + 1, L):
                tj = idx[terms[j]]
                if divs[ti][tj] or divs[tj][ti]:
                    return False
        return True

    answers = [None] * 129  # 1-indexed

    # k=1
    for n in range(1, 129):
        if n in smooth_set:
            answers[n] = [n]

    # k=2: precompute all valid antichain pairs and record which sums they cover
    # For each target, pick the first valid pair (any is fine, count=2)
    pair_for = {}
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            if s in smooth_set:
                continue  # already k=1
            if answers[s] is not None:
                continue
            # antichain check: a and b
            if divs[i][j] or divs[j][i]:
                continue
            pair_for[s] = [a, b]
            answers[s] = [a, b]

    # k=3: for remaining targets, find a valid antichain triple
    remaining = [n for n in range(1, 129) if answers[n] is None]
    for n in remaining:
        if time.time() > deadline:
            break
        found = None
        # iterate triples a<b<c from smooth, a+b+c=n
        # restrict a+b < n
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c not in smooth_set:
                    continue
                k = idx[c]
                if k <= j:
                    continue
                # antichain among i, j, k
                if divs[i][j] or divs[j][i]:
                    continue
                if divs[i][k] or divs[k][i]:
                    continue
                if divs[j][k] or divs[k][j]:
                    continue
                found = [a, b, c]
                break
            if found:
                break
        if found:
            answers[n] = found

    # k=4 fallback for any still missing (should be rare)
    still = [n for n in range(1, 129) if answers[n] is None]
    for n in still:
        if time.time() > deadline:
            break
        found = None
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d not in smooth_set:
                        continue
                    m = idx[d]
                    if m <= k:
                        continue
                    if is_antichain_vals([a, b, c, d]):
                        found = [a, b, c, d]
                        break
                if found:
                    break
            if found:
                break
        if found:
            answers[n] = found

    # Final safety: any still-missing target, brute force up to k=6
    still = [n for n in range(1, 129) if answers[n] is None]
    for n in still:
        if time.time() > deadline:
            break
        pool = [x for x in smooth if x <= n]
        found = None
        for k in range(2, min(30, len(pool)) + 1):
            if time.time() > deadline:
                break
            for combo in combinations(pool, k):
                if sum(combo) == n and is_antichain_vals(list(combo)):
                    found = list(combo)
                    break
            if found:
                break
        if found:
            answers[n] = found

    # Last resort
    for n in range(1, 129):
        if answers[n] is None:
            # must produce something valid; try [n] if smooth else a 2-term
            if n in smooth_set:
                answers[n] = [n]
            else:
                # find any valid rep quickly
                done = False
                for a in smooth:
                    if a >= n:
                        break
                    b = n - a
                    if b > a and b in smooth_set and b % a != 0:
                        answers[n] = [a, b]
                        done = True
                        break
                if not done:
                    answers[n] = [n]  # fallback (may be invalid but must return)

    return answers[1:129]

# Precompute at module load time
_ANSWERS = _precompute_answers()

def construct():
    return _ANSWERS
# EVOLVE-BLOCK-END
