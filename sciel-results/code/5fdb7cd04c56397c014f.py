import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = 80.0  # leave margin under 90s

    # Try to find a minimal-count solution via DFS for counts 8,9,10,11
    for target_count in range(8, 12):
        if time.time() > deadline:
            break
        sol = _search(target_count, deadline)
        if sol is not None:
            return sol

    # Fallback known 11-term solution: 5,7,8,9,10,12,15,18,20,24,42
    return [5, 7, 8, 9, 10, 12, 15, 18, 20, 24, 42]


def _search(target_count, deadline):
    """
    DFS to find exactly target_count distinct denominators >= 5 summing reciprocally to 1.
    Uses float-based pruning for speed, verifies with Fraction at leaf.
    """
    path = []

    def dfs(depth, remaining_f, last_d):
        """remaining_f is a Fraction."""
        if time.time() > deadline:
            return None

        if remaining_f == 0:
            if depth == target_count:
                return list(path)
            return None

        if depth == target_count:
            return None

        slots = target_count - depth

        # Lower bound on next denominator:
        # 1/d <= remaining_f  =>  d >= ceil(1 / remaining_f)
        # Compute in float for speed
        rem_f = float(remaining_f)
        if rem_f <= 0:
            return None
        min_d = int(1.0 / rem_f) + 1
        if min_d <= last_d:
            min_d = last_d + 1

        # Upper bound via float: max sum of 'slots' terms starting at d is approx slots/d
        # We need slots/d >= rem_f => d <= slots / rem_f
        max_d_float = slots / rem_f
        max_d = int(max_d_float) + 2  # small slack

        # Also, the actual max sum condition: sum_{i=0}^{slots-1} 1/(d+i) >= rem_f
        # As d increases, this decreases. We can break when it drops below.

        d = min_d
        while d <= max_d:
            if time.time() > deadline:
                return None

            # Check 1/d <= remaining_f
            if Fraction(1, d) > remaining_f:
                d += 1
                continue

            # Check if max possible sum with 'slots' terms starting at d is >= remaining_f
            # Compute sum_{i=0}^{slots-1} 1/(d+i) in float for speed
            max_sum_f = 0.0
            for i in range(slots):
                max_sum_f += 1.0 / (d + i)
            if max_sum_f < rem_f - 1e-15:
                break

            # Try d
            path.append(d)
            new_rem = remaining_f - Fraction(1, d)
            if new_rem < 0:
                path.pop()
                d += 1
                continue

            res = dfs(depth + 1, new_rem, d)
            if res is not None:
                return res

            path.pop()
            d += 1

        return None

    return dfs(0, Fraction(1, 1), 4)


if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    s = sum(Fraction(1, d) for d in res)
    print(s)
    assert s == 1
    assert all(5 <= d <= 10000000 for d in res)
    assert len(set(res)) == len(res)
