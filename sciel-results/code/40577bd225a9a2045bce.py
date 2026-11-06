import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break

        found = None
        t_start = time.time()
        remaining = deadline - t_start
        # Allocate more time to harder (larger n) cases
        n_budget = min(0.35, remaining / max(1, 256 - idx))

        x_min = n // 4 + 1
        x_max = 3 * n // 4

        # Build candidate x list: try x=n first, then nearby, then rest
        candidate_xs = []
        if x_min <= n <= x_max:
            candidate_xs.append(n)
        for dx in range(1, 80):
            for x in [n - dx, n + dx]:
                if x_min <= x <= x_max and x not in candidate_xs:
                    candidate_xs.append(x)
        for x in range(x_min, x_max + 1):
            if x not in candidate_xs:
                candidate_xs.append(x)

        for x in candidate_xs:
            if time.time() > deadline or (time.time() - t_start) > n_budget:
                break

            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x

            # y must satisfy: 1/y + 1/z = A/B with x < y < z
            # y is in (B/A, 2B/A) and y > x
            y_min = B // A + 1
            y_max = (2 * B - 1) // A
            if y_min <= x:
                y_min = x + 1
            if y_min > y_max:
                continue

            # Iterate over y in the valid range
            # Use k substitution: y = (k*A - B + B) / A... actually let's use direct y
            # z = B*y / (A*y - B), must be integer and > y
            for y in range(y_min, y_max + 1):
                D = A * y - B
                if D <= 0:
                    continue
                Num = B * y
                if Num % D != 0:
                    continue
                z = Num // D
                if z <= y:
                    continue
                if z > DENOM_BOUND:
                    continue
                if y > DENOM_BOUND:
                    continue
                # All distinct since x < y < z
                found = [x, y, z]
                break

            if found:
                break

        results[idx] = found

    return results
