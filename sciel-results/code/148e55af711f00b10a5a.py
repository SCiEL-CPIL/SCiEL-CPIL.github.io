import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_val = float('inf')
    best_list = None

    # Brute force for small ranges [211, M]
    # Check k=1 to k=5 for M up to 235
    max_M = 235
    nums = list(range(211, max_M + 1))
    n = len(nums)

    # k=1
    for x in nums:
        P = x * 210
        s = math.isqrt(P)
        if s * s == P:
            val = x - 210
            if val < best_val:
                best_val = val
                best_list = [x]

    # k=2
    for i in range(n):
        for j in range(i + 1, n):
            P = nums[i] * nums[j] * 210
            s = math.isqrt(P)
            if s * s == P:
                val = nums[j] - 210
                if val < best_val:
                    best_val = val
                    best_list = [nums[i], nums[j]]

    # k=3
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                P = nums[i] * nums[j] * nums[k] * 210
                s = math.isqrt(P)
                if s * s == P:
                    val = nums[k] - 210
                    if val < best_val:
                        best_val = val
                        best_list = [nums[i], nums[j], nums[k]]

    # k=4
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    P = nums[i] * nums[j] * nums[k] * nums[l] * 210
                    s = math.isqrt(P)
                    if s * s == P:
                        val = nums[l] - 210
                        if val < best_val:
                            best_val = val
                            best_list = [nums[i], nums[j], nums[k], nums[l]]

    # k=5
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    for m in range(l + 1, n):
                        P = nums[i] * nums[j] * nums[k] * nums[l] * nums[m] * 210
                        s = math.isqrt(P)
                        if s * s == P:
                            val = nums[m] - 210
                            if val < best_val:
                                best_val = val
                                best_list = [nums[i], nums[j], nums[k], nums[l], nums[m]]

    if best_val < 21:
        return best_list

    # k=2 formula search: x1 * x2 = 210 * s^2
    # Need x1 > 210, x2 > 210, x1 != x2
    # Minimize max(x1, x2) - 210
    for s in range(15, 500):
        if time.time() > deadline:
            break
        prod = 210 * s * s
        sqrt_p = math.isqrt(prod)
        # Search near sqrt for balanced factor pairs
        lo = max(211, sqrt_p - 200)
        hi = min(prod // 211, sqrt_p + 200)
        for x1 in range(lo, hi + 1):
            if prod % x1 == 0:
                x2 = prod // x1
                if x2 > 210 and x2 != x1:
                    val = max(x1, x2) - 210
                    if val < best_val:
                        best_val = val
                        best_list = [x1, x2]
                        if val <= 5:
                            return best_list

    # k=3 formula search: x1*x2*x3 = 210 * s^2
    # Try small s values
    for s in range(15, 100):
        if time.time() > deadline:
            break
        prod = 210 * s * s
        # Need three factors > 210 whose product is prod
        # x1 >= 211, x2 >= 212, x3 >= 213 (roughly)
        # x1 * x2 * x3 = prod
        # x1 from 211 to prod^(1/3)
        cube_root = int(round(prod ** (1/3)))
        max_x1 = min(prod // (211 * 212), cube_root + 50)
        for x1 in range(211, max_x1 + 1):
            if prod % x1 != 0:
                continue
            rem = prod // x1
            # x2 * x3 = rem, x2 > 210, x3 > 210
            sqrt_rem = math.isqrt(rem)
            for x2 in range(max(211, sqrt_rem - 100), min(rem // 211, sqrt_rem + 100) + 1):
                if rem % x2 == 0:
                    x3 = rem // x2
                    if x3 > 210 and x3 != x2 and x3 != x1 and x2 != x1:
                        val = max(x1, x2, x3) - 210
                        if val < best_val:
                            best_val = val
                            best_list = [x1, x2, x3]
                            if val <= 5:
                                return best_list

    if best_list is None:
        return [840]

    return best_list
