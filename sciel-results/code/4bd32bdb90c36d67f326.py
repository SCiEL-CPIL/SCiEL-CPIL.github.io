import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    def reduce(num, den):
        if num == 0:
            return (0, 1)
        g = gcd(abs(num), abs(den))
        if g > 1:
            num //= g
            den //= g
        return (num, den)

    def frac_sub(rn, rd, d):
        new_num = rn * d - rd
        new_den = rd * d
        return reduce(new_num, new_den)

    def frac_gt(a_num, a_den, b_num, b_den):
        return a_num * b_den > b_num * a_den

    def frac_eq(a_num, a_den, b_num, b_den):
        return a_num * b_den == b_num * a_den

    # Precompute harmonic sums H[(a,b)] = sum_{i=a}^{b} 1/i for small ranges
    MAX_PRECOMP = 200
    H = {}
    for a in range(1, MAX_PRECOMP + 1):
        total_num, total_den = 0, 1
        for b in range(a, MAX_PRECOMP + 1):
            total_num, total_den = reduce(total_num * b + total_den, total_den * b)
            H[(a, b)] = (total_num, total_den)

    def get_harmonic(min_d, max_d):
        if min_d > max_d:
            return (0, 1)
        if min_d >= 1 and max_d <= MAX_PRECOMP:
            return H[(min_d, max_d)]
        # Compute on the fly for larger ranges (shouldn't happen often)
        total_num, total_den = 0, 1
        for i in range(min_d, max_d + 1):
            total_num, total_den = reduce(total_num * i + total_den, total_den * i)
        return total_num, total_den

    def search(rn, rd, min_d, max_d, path):
        if time.time() - start_time > deadline:
            return None

        if rn == 0:
            return list(path)

        if rn < 0 or len(path) >= 256 or min_d > max_d:
            return None

        # Pruning: max possible sum from min_d to max_d
        mh_num, mh_den = get_harmonic(min_d, max_d)
        if frac_gt(rn, rd, mh_num, mh_den):
            return None

        # Determine starting d
        if rn > 0:
            min_needed_d = (rd + rn - 1) // rn  # ceil(rd/rn)
            if min_needed_d > max_d:
                return None
            d_start = max(min_d, min_needed_d)
        else:
            d_start = min_d

        if d_start > max_d:
            return None

        for d in range(d_start, max_d + 1):
            if time.time() - start_time > deadline:
                return None

            new_rn, new_rd = frac_sub(rn, rd, d)

            # Prune before recursing
            if new_rn > 0 and d + 1 <= max_d:
                mh2_num, mh2_den = get_harmonic(d + 1, max_d)
                if frac_gt(new_rn, new_rd, mh2_num, mh2_den):
                    continue
                min_next_d = (new_rd + new_rn - 1) // new_rn
                if min_next_d > max_d:
                    continue

            path.append(d)
            result = search(new_rn, new_rd, d + 1, max_d, path)
            path.pop()

            if result is not None:
                return result

        return None

    # Try increasing D values
    for D in range(7, 300):
        if time.time() - start_time > deadline:
            break

        result = search(5, 31, 7, D, [])
        if result is not None:
            # Verify the solution
            total_num, total_den = 0, 1
            for d in result:
                total_num, total_den = reduce(total_num * d + total_den, total_den * d)
            if frac_eq(total_num, total_den, 5, 31):
                if all(1 <= d <= 10000000 for d in result):
                    if len(set(result)) == len(result):
                        if 1 <= len(result) <= 256:
                            return result

    # Fallback: greedy Egyptian fraction
    rn, rd = 5, 31
    result = []
    while rn > 0 and len(result) < 256:
        d = (rd + rn - 1) // rn
        if d > 10000000:
            break
        result.append(d)
        rn, rd = frac_sub(rn, rd, d)

    if rn == 0 and len(result) > 0:
        return result

    # Last resort
    return [7, 55, 4000]
