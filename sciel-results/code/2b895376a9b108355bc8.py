"""Combined program: CSP-style backtracking over the 9 sum-grid variables with
rank-1 consistency and incremental pairwise-coprimality checks.
Starts from the known score-5 seed and searches for higher-scoring (up to 9) solutions.
"""
import time
import math

_START = time.time()
_DEADLINE = 82.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(s):
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(s[i], s[j]) != 1:
                return False
    return True

def _is_valid(A, B):
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    for x in A:
        if x < 1 or x > 128:
            return False
    for x in B:
        if x < 1 or x > 128:
            return False
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) < 2:
        return False
    return _pairwise_coprime(sorted(sums))

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def _solve():
    # Known good: score 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5

    # Precompute coprime table up to 256 (max sum = 128+128)
    MAX_SUM = 256
    coprime = [[True] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1):
            if _gcd(i, j) != 1:
                coprime[i][j] = False
                coprime[j][i] = False

    def all_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            ni = nums[i]
            for j in range(i + 1, n):
                if not coprime[ni][nums[j]]:
                    return False
        return True

    def search_for_B(A, max_val, current_best):
        """CSP-style: fix A, enumerate B with pruning via rank-1 + coprimality."""
        a1, a2, a3 = A[0], A[1], A[2]
        local_best_B = None
        local_best_s = current_best

        for b1 in range(1, max_val + 1):
            s11 = a1 + b1
            s21 = a2 + b1
            s31 = a3 + b1
            # Row 1 must be pairwise coprime
            if not coprime[s11][s21] or not coprime[s11][s31] or not coprime[s21][s31]:
                continue

            row1 = [s11, s21, s31]
            for b2 in range(b1 + 1, max_val + 1):
                s12 = a1 + b2
                s22 = a2 + b2
                s32 = a3 + b2
                # Row 2 internal
                if not coprime[s12][s22] or not coprime[s12][s32] or not coprime[s22][s32]:
                    continue
                # Row 2 vs Row 1
                ok = True
                for ns in (s12, s22, s32):
                    for ps in row1:
                        if not coprime[ns][ps]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                row2 = [s12, s22, s32]
                for b3 in range(b2 + 1, max_val + 1):
                    s13 = a1 + b3
                    s23 = a2 + b3
                    s33 = a3 + b3
                    # Row 3 internal
                    if not coprime[s13][s23] or not coprime[s13][s33] or not coprime[s23][s33]:
                        continue
                    # Row 3 vs Rows 1,2
                    ok = True
                    for ns in (s13, s23, s33):
                        for ps in row1 + row2:
                            if not coprime[ns][ps]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue

                    all_sums = {s11, s21, s31, s12, s22, s32, s13, s23, s33}
                    sc = len(all_sums)
                    if sc > local_best_s:
                        local_best_s = sc
                        local_best_B = [b1, b2, b3]
                        if local_best_s >= 9:
                            return local_best_B, local_best_s

        return local_best_B, local_best_s

    # Phase 1: small A values (1..30), B up to 30
    for a1 in range(1, 31):
        if _elapsed() > _DEADLINE:
            break
        for a2 in range(a1 + 1, 31):
            if _elapsed() > _DEADLINE:
                break
            for a3 in range(a2 + 1, 31):
                if _elapsed() > _DEADLINE:
                    break
                A = [a1, a2, a3]
                b_res, sc = search_for_B(A, 30, best_score)
                if sc > best_score:
                    best_score = sc
                    best_A = A[:]
                    best_B = b_res[:]
                    if best_score >= 9:
                        return [best_A, best_B]

    # Phase 2: medium A (1..50), B up to 50
    for a1 in range(1, 51):
        if _elapsed() > _DEADLINE:
            break
        for a2 in range(a1 + 1, 51):
            if _elapsed() > _DEADLINE:
                break
            for a3 in range(a2 + 1, 51):
                if _elapsed() > _DEADLINE:
                    break
                A = [a1, a2, a3]
                b_res, sc = search_for_B(A, 50, best_score)
                if sc > best_score:
                    best_score = sc
                    best_A = A[:]
                    best_B = b_res[:]
                    if best_score >= 9:
                        return [best_A, best_B]

    # Phase 3: larger A (1..80), B up to 80
    for a1 in range(1, 81):
        if _elapsed() > _DEADLINE:
            break
        for a2 in range(a1 + 1, 81):
            if _elapsed() > _DEADLINE:
                break
            for a3 in range(a2 + 1, 81):
                if _elapsed() > _DEADLINE:
                    break
                A = [a1, a2, a3]
                b_res, sc = search_for_B(A, 80, best_score)
                if sc > best_score:
                    best_score = sc
                    best_A = A[:]
                    best_B = b_res[:]
                    if best_score >= 9:
                        return [best_A, best_B]

    # Phase 4: full range but sparser A sampling
    step = 3
    for a1 in range(1, 129, step):
        if _elapsed() > _DEADLINE:
            break
        for a2 in range(a1 + 1, 129, step):
            if _elapsed() > _DEADLINE:
                break
            for a3 in range(a2 + 1, 129, step):
                if _elapsed() > _DEADLINE:
                    break
                A = [a1, a2, a3]
                b_res, sc = search_for_B(A, 128, best_score)
                if sc > best_score:
                    best_score = sc
                    best_A = A[:]
                    best_B = b_res[:]
                    if best_score >= 9:
                        return [best_A, best_B]

    if best_score >= 2 and _is_valid(best_A, best_B):
        return [best_A, best_B]

    return [best_A, best_B]

def construct():
    try:
        result = _solve()
        if _is_valid(result[0], result[1]):
            return result
    except Exception:
        pass
    return [[1, 7, 13], [4, 10, 16]]
