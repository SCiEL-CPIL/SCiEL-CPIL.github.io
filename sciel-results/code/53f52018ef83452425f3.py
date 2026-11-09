import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # Target parity: odd exponent for primes 2, 3, 5, 7; even for all others.
    # We need product of selected * 210 to be a perfect square.
    # So XOR of parities of selected numbers must equal parity of 210.

    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i * i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Target parity: 210 = 2*3*5*7
    target_primes = [2, 3, 5, 7]
    target_mask = 0
    for p in target_primes:
        if p in prime_to_idx:
            target_mask |= (1 << prime_to_idx[p])

    def get_parity_mask(x):
        """Return bitmask of prime indices with odd exponent in x."""
        mask = 0
        temp = x
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    mask |= (1 << i)
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute parities for numbers 211 to max_val
    parities = [0] * (max_val + 1)
    for x in range(211, max_val + 1):
        parities[x] = get_parity_mask(x)

    # Build parity_map: map from parity mask to sorted list of numbers
    from collections import defaultdict
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)

    best_result = [840]  # fallback: 840*210 = 176400 = 420^2
    best_raw = 630
    current_best_max = 840

    # Search for size 1: x such that x*210 is a perfect square
    # x = 210 * k^2, need 211 <= x <= max_val
    for k in range(1, 20):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    best_result = [x]
                    current_best_max = x

    # Search for size 2: x*y*210 is a perfect square
    # parity(x) XOR parity(y) = target_mask
    # For each x, find y with parity = parity(x) XOR target_mask, y != x, y > 210
    # To minimize max(x,y), we want both small.

    for x in range(211, current_best_max):
        if time.time() > deadline:
            break
        px = parities[x]
        py = px ^ target_mask
        if py in parity_map:
            # Find smallest y in parity_map[py] such that y != x and y > 210
            for y in parity_map[py]:
                if y == x:
                    continue
                if y <= 210:
                    continue
                max_xy = max(x, y)
                if max_xy < current_best_max:
                    # Verify
                    prod = x * y * 210
                    if _is_square(prod):
                        current_best_max = max_xy
                        best_raw = max_xy - 210
                        best_result = [x, y]
                        break  # Found best y for this x, move to next x
                else:
                    break  # y is too large, no point continuing for this x

    # Search for size 3: x*y*z*210 is a perfect square
    # parity(x) XOR parity(y) XOR parity(z) = target_mask
    # Fix x and y, find z with parity = target_mask XOR parity(x) XOR parity(y)
    # To keep max small, use small x and y.

    if best_raw > 50 and time.time() < deadline:
        # Use small numbers for x and y
        small_numbers = list(range(211, min(400, max_val + 1)))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                needed_parity = target_mask ^ parities[x] ^ parities[y]
                if needed_parity in parity_map:
                    # Find smallest z with that parity, z != x, z != y, z > 210
                    for z in parity_map[needed_parity]:
                        if z == x or z == y or z <= 210:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz < current_best_max:
                            prod = x * y * z * 210
                            if _is_square(prod):
                                current_best_max = max_xyz
                                best_raw = max_xyz - 210
                                best_result = [x, y, z]
                                break
                    # Continue to find potentially better solutions

    # Search for size 4: x*y*z*w*210 is a perfect square
    if best_raw > 30 and time.time() < deadline:
        # Use small numbers for x, y, z
        small_numbers = list(range(211, min(300, max_val + 1)))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                for k in range(j + 1, len(small_numbers)):
                    if time.time() > deadline:
                        break
                    z = small_numbers[k]
                    needed_parity = target_mask ^ parities[x] ^ parities[y] ^ parities[z]
                    if needed_parity in parity_map:
                        # Find smallest w with that parity, w != x, w != y, w != z, w > 210
                        for w in parity_map[needed_parity]:
                            if w == x or w == y or w == z or w <= 210:
                                continue
                            max_xyzw = max(x, y, z, w)
                            if max_xyzw < current_best_max:
                                prod = x * y * z * w * 210
                                if _is_square(prod):
                                    current_best_max = max_xyzw
                                    best_raw = max_xyzw - 210
                                    best_result = [x, y, z, w]
                                    break
                    # Continue to find potentially better solutions

    # Verify the best result
    def verify(solution):
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) < 1 or len(solution) > max_size:
            return False
        if len(set(solution)) != len(solution):
            return False
        for x in solution:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x <= n or x > max_val:
                return False
        prod = n
        for x in solution:
            prod *= x
        r = math.isqrt(prod)
        if r * r != prod:
            return False
        return True

    if not verify(best_result):
        # Fallback to known good solution
        best_result = [840]
        best_raw = 630

    # Return as list of Python ints
    return [int(x) for x in best_result]
