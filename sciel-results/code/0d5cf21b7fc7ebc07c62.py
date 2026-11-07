import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    N = 12
    LCM = 27720  # LCM(1,2,...,12)
    MAX_L = 144  # upper bound on f(12, m)

    def can_match(m, L):
        """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i using bitmask backtracking."""
        cands = []
        for i in range(1, N + 1):
            r = (-m) % i
            if r == 0:
                start = i
            else:
                start = r
            lst = []
            d = start
            while d <= L:
                lst.append(d)
                d += i
            cands.append(lst)

        # Sort by most constrained first (fewest candidates)
        cands.sort(key=lambda x: len(x))

        if len(cands[0]) == 0:
            return False

        # Bitmask backtracking
        def bt(k, used_mask):
            if k == N:
                return True
            lst = cands[k]
            for d in lst:
                bit = 1 << (d - 1)
                if not (used_mask & bit):
                    if bt(k + 1, used_mask | bit):
                        return True
            return False

        return bt(0, 0)

    # L-first search: find the maximum L such that some m fails to match
    # For each L from 1 upward, scan m to find one that fails
    # The answer is the m that fails at the highest L found
    best_m = 0
    best_L = 0

    for L in range(1, MAX_L + 1):
        if time.time() - start_time > deadline:
            break

        found_failing_m = None
        for m in range(LCM):
            if time.time() - start_time > deadline:
                break
            if not can_match(m, L):
                found_failing_m = m
                break

        if found_failing_m is not None:
            # This m fails at L, so f(12, m) >= L+1
            best_m = found_failing_m
            best_L = L
        else:
            # No m fails at this L, so all m match at L
            # The maximum f is L (the previous best_L where some m failed)
            break

    return int(best_m)


if __name__ == "__main__":
    construct()
