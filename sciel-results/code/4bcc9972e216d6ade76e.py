import time
from math import gcd

def construct():
    # Fast construction: divisors of 480 form a valid set of size 24.
    # All pairwise lcms divide 480 <= 512.
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs

    S = get_divisors(480)  # 24 elements
    best_S = set(S)

    # Try to improve via greedy addition and swaps within time budget
    start_time = time.time()
    deadline = 50.0

    def can_add(x, S_set):
        for y in S_set:
            l = x * y // gcd(x, y)
            if l > 512:
                return False
        return True

    # Greedy pass
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 513):
            if x in best_S:
                continue
            if can_add(x, best_S):
                best_S.add(x)
                added = True

    # Swap: remove one, add up to two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best_S):
            if time.time() > deadline:
                break
            S_trial = best_S - {x}
            candidates = [y for y in range(1, 513) if y not in S_trial and can_add(y, S_trial)]
            # Try to add 2 mutually compatible candidates
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(candidates)):
                    c1, c2 = candidates[i], candidates[j]
                    if c1 * c2 // gcd(c1, c2) <= 512:
                        new_S = S_trial | {c1, c2}
                        if len(new_S) > len(best_S):
                            best_S = new_S
                            improved = True
                            break
                if improved:
                    break

    # Final greedy
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 513):
            if x in best_S:
                continue
            if can_add(x, best_S):
                best_S.add(x)
                added = True

    return list(best_S)
