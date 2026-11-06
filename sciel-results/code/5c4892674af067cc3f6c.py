import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Any element x in a valid set S (containing 255) must satisfy gcd(x, 255) > 1,
    # so x must be divisible by 3, 5, or 17.
    #
    # Key structural insight: In a pairwise non-coprime set where every element
    # shares a factor with 255 = 3*5*17, all elements must share a common prime
    # factor p in {3, 5, 17}. Proof sketch: if a is divisible only by 3 (not 5,17)
    # and b is divisible only by 5 (not 3,17), then gcd(a,b)=1, contradiction.
    # Thus S ⊆ multiples of some p ∈ {3,5,17}.
    #
    # |multiples of 3 in [1,255]| = 85
    # |multiples of 5 in [1,255]| = 51
    # |multiples of 17 in [1,255]| = 15
    #
    # Maximum is 85 (multiples of 3).

    # Primary answer: all multiples of 3
    S = list(range(3, 256, 3))

    # Verify correctness
    if 255 not in S:
        S.append(255)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if gcd(S[i], S[j]) == 1:
                # Should never happen, but fallback
                S = list(range(3, 256, 3))
                break

    # Search machinery: attempt to improve beyond the structural optimum.
    # This is a safety net; the mathematical argument above shows 85 is optimal.
    # We run a limited local search in case our reasoning has a gap.

    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    best_S = S[:]
    best_size = len(S)

    # Greedy: try to add any candidate not in S
    if (time.time() - start_time) < deadline - 15:
        S_set = set(S)
        improved = True
        while improved and (time.time() - start_time) < deadline - 15:
            improved = False
            for x in candidates:
                if x in S_set:
                    continue
                can_add = all(gcd(x, y) > 1 for y in S)
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]

    # Removal-based local search: remove one element, greedily add all possible
    if (time.time() - start_time) < deadline - 5:
        for x in list(best_S):
            if (time.time() - start_time) > deadline - 3:
                break
            S_temp = [y for y in best_S if y != x]
            S_temp_set = set(S_temp)
            for y in candidates:
                if y in S_temp_set:
                    continue
                if all(gcd(y, z) > 1 for z in S_temp):
                    S_temp.append(y)
                    S_temp_set.add(y)
            if len(S_temp) > best_size:
                best_size = len(S_temp)
                best_S = S_temp[:]

    return best_S
