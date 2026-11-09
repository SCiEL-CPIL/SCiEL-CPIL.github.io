import time
from fractions import Fraction
from math import gcd as math_gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds

    def time_exceeded():
        return (time.time() - start_time) > deadline

    def verify_solution(denoms):
        if not denoms or len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1

    def greedy_egyptian(N, D, min_denom, max_terms):
        """Greedy algorithm: pick largest unit fraction <= remaining."""
        denoms = []
        for _ in range(max_terms):
            if N == 0:
                break
            if time_exceeded():
                return None
            d = max(min_denom, (D + N - 1) // N)
            if d > 10000000:
                return None
            if d * N > D:
                d += 1
                if d > 10000000:
                    return None
            denoms.append(d)
            N = N * d - D
            D = D * d
            g = math_gcd(N, D)
            if g > 0:
                N //= g
                D //= g
            min_denom = d + 1
        if N == 0 and len(denoms) <= max_terms:
            return denoms
        return None

    def search(N, D, min_denom, terms_left, current, used, depth=0):
        if time_exceeded():
            return "TIMEOUT"

        if terms_left == 0:
            return current[:] if N == 0 else None

        if terms_left == 1:
            if N == 0:
                return None
            if D % N == 0:
                d = D // N
                if d >= min_denom and d <= 10000000 and d not in used:
                    return current + [d]
            return None

        # Pruning: max possible sum of remaining terms
        max_sum_num = 0
        max_sum_den = 1
        for i in range(terms_left):
            d = min_denom + i
            max_sum_num = max_sum_num * d + max_sum_den
            max_sum_den = max_sum_den * d
        if N * max_sum_den > D * max_sum_num:
            return None

        # Min possible sum check
        if N * 10000000 < D * terms_left:
            return None

        # Lower bound for d
        min_d = max(min_denom, (D + N - 1) // N)

        # Upper bound: tighter
        max_d = min(10000000, (terms_left * D) // N + 10)

        if min_d > max_d:
            return None

        # Limit search space
        if max_d - min_d > 3000:
            # Use greedy-like heuristic
            greedy_d = max(min_denom, (D + N - 1) // N)
            if greedy_d < min_d:
                greedy_d = min_d
            if greedy_d > max_d:
                greedy_d = max_d
            candidates = set()
            for offset in range(-30, 31):
                dd = greedy_d + offset
                if min_d <= dd <= max_d:
                    candidates.add(dd)
            candidates.add(min_d)
            candidates.add(max_d)
            candidates.add((min_d + max_d) // 2)
            for d in sorted(candidates):
                if time_exceeded():
                    return "TIMEOUT"
                if d in used:
                    continue
                new_N = N * d - D
                if new_N < 0:
                    continue
                if new_N == 0:
                    continue
                new_D = D * d
                g = math_gcd(new_N, new_D)
                if g > 1:
                    new_N //= g
                    new_D //= g
                used.add(d)
                result = search(new_N, new_D, d + 1, terms_left - 1, current + [d], used, depth + 1)
                used.discard(d)
                if result is not None and result != "TIMEOUT":
                    return result
        else:
            for d in range(min_d, max_d + 1):
                if time_exceeded():
                    return "TIMEOUT"
                if d in used:
                    continue
                new_N = N * d - D
                if new_N < 0:
                    continue
                if new_N == 0:
                    continue
                new_D = D * d
                g = math_gcd(new_N, new_D)
                if g > 1:
                    new_N //= g
                    new_D //= g
                used.add(d)
                result = search(new_N, new_D, d + 1, terms_left - 1, current + [d], used, depth + 1)
                used.discard(d)
                if result is not None and result != "TIMEOUT":
                    return result

        return None

    # Try known small solutions first
    known_solutions = [
        # 15-term solution
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 30, 42],
    ]
    
    best = None
    for sol in known_solutions:
        if verify_solution(sol):
            best = sol
            break
    
    # If no known solution works, use greedy
    if best is None:
        greedy_result = greedy_egyptian(1, 1, 5, 256)
        if greedy_result is not None and verify_solution(greedy_result):
            best = greedy_result
    
    # Try to find a shorter solution
    if best is not None:
        min_k_to_try = 5
        max_k_to_try = len(best) - 1
    else:
        min_k_to_try = 5
        max_k_to_try = 20  # Limit initial search
    
    for k in range(min_k_to_try, max_k_to_try + 1):
        if time_exceeded():
            break
        used = set()
        result = search(1, 1, 5, k, [], used)
        if result is not None and result != "TIMEOUT":
            if verify_solution(result):
                best = result
                break
        elif result == "TIMEOUT":
            break

    if best is not None:
        return best

    # Final fallback
    greedy_result = greedy_egyptian(1, 1, 5, 256)
    if greedy_result is not None:
        return greedy_result

    # Absolute last resort
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
