"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    start_time = time.time()
    DEADLINE = 85.0

    def time_ok():
        return time.time() - start_time < DEADLINE

    target = Fraction(5, 31)

    # Greedy solution for initial upper bound
    def greedy_solution(target_frac):
        r = target_frac
        values = []
        while r > 0 and len(values) < 256:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0:
                break
            values.append(d)
            r -= Fraction(1, d)
        if r == 0 and len(values) <= 256:
            return values
        return None

    greedy = greedy_solution(target)
    if greedy is None:
        # Should not happen, but fallback
        return [8, 16, 48, 240, 480, 960, 1920, 3840, 7680]

    best_solution = list(greedy)
    best_max_den = max(greedy)

    def dfs_search(remaining, last_den, current_list):
        nonlocal best_solution, best_max_den

        if not time_ok():
            return

        if remaining == 0:
            if current_list:
                max_d = max(current_list)
                if max_d < best_max_den:
                    best_max_den = max_d
                    best_solution = list(current_list)
            return

        if remaining < 0 or len(current_list) >= 256:
            return

        n = remaining.numerator
        d = remaining.denominator
        min_next = max(last_den + 1, (d + n - 1) // n)

        if min_next >= best_max_den:
            return

        # Generate candidate denominators
        greedy_choice = max(min_next, (d + n - 1) // n)
        candidates = set()
        candidates.add(greedy_choice)
        for offset in range(1, 30):
            c1 = greedy_choice - offset
            c2 = greedy_choice + offset
            if c1 > last_den and c1 < best_max_den:
                candidates.add(c1)
            if c2 < best_max_den:
                candidates.add(c2)
        for i in range(min_next, min(min_next + 15, best_max_den)):
            candidates.add(i)

        candidates = sorted(candidates)

        for di in candidates:
            if not time_ok():
                return
            if di <= last_den or di >= best_max_den:
                continue
            new_remaining = remaining - Fraction(1, di)
            if new_remaining < 0:
                continue
            if new_remaining == 0:
                new_list = current_list + [di]
                max_d = max(new_list)
                if max_d < best_max_den:
                    best_max_den = max_d
                    best_solution = list(new_list)
                continue
            dfs_search(new_remaining, di, current_list + [di])

    dfs_search(target, 0, [])

    return best_solution
# EVOLVE-BLOCK-END
