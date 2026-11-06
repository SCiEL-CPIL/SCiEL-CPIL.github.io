import time
import bisect

def construct():
    start = time.time()
    deadline = start + 80.0
    n = 256

    def can_add_fast(a, S_set, S_sorted):
        """Check if a can be added to S_set. All elements in S_set are > a."""
        max_sum = 2 * n
        for m in range(3 * a, max_sum + 1, a):
            b_min = max(a + 1, m - n)
            b_max = m // 2
            if b_min > b_max:
                continue
            left = bisect.bisect_left(S_sorted, b_min)
            right = bisect.bisect_right(S_sorted, b_max)
            for idx in range(left, right):
                b = S_sorted[idx]
                c = m - b
                if c >= b and c in S_set:
                    return False
        return True

    def is_interval_valid(m_start):
        """Check if [m_start, n] is a valid set using interval arithmetic."""
        for a in range(m_start, n + 1):
            for k in range(3, 2 * n // a + 1):
                m = k * a
                if m > 2 * n:
                    break
                # b + c = m, a < b <= c, b,c in [m_start, n]
                # b in [max(a+1, m-n, m_start), min(m//2, m-m_start, n)]
                b_lo = max(a + 1, m - n, m_start)
                b_hi = min(m // 2, m - m_start, n)
                if b_lo <= b_hi:
                    return False
        return True

    def greedy_build(m_start, deadline):
        S_list = list(range(m_start, n + 1))
        S_set = set(S_list)
        for a in range(m_start - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            if a in S_set:
                continue
            if can_add_fast(a, S_set, S_list):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
        return S_list, S_set

    best_S = set(range(171, n + 1))
    best_size = len(best_S)

    for m_start in range(170, 50, -1):
        if time.time() > deadline - 15:
            break
        if not is_interval_valid(m_start):
            continue
        S_list, S_set = greedy_build(m_start, deadline)
        if len(S_list) > best_size:
            best_S = S_set
            best_size = len(S_list)

    if time.time() < deadline - 10:
        S_list = sorted(best_S)
        S_set = set(S_list)
        for a in range(n, 0, -1):
            if time.time() > deadline - 5:
                break
            if a in S_set:
                continue
            if can_add_fast(a, S_set, S_list):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
        if len(S_list) > best_size:
            best_S = S_set
            best_size = len(S_list)

    return sorted(best_S)
