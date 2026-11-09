# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008  # values are 1..1008

    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse: 1 and 1008
    lo_bound = 2
    hi_bound = 1007

    # is_L[x] is True if x < inv[x] (x is the smaller element of the pair)
    # is_R[x] is True if x > inv[x] (x is the larger element of the pair)
    is_L = [False] * (MOD + 1)
    for x in range(lo_bound, hi_bound + 1):
        if x < inv[x]:
            is_L[x] = True
        # else x > inv[x] (since x != inv[x] for x in 2..1007)

    best_start = lo_bound
    best_len = 0

    for s in range(lo_bound, hi_bound + 1):
        e_max = hi_bound
        # Iterate x from s upwards, updating the maximum valid end
        for x in range(s, hi_bound + 1):
            if x > e_max:
                break
            
            if is_L[x]:
                # Constraint: e < inv[x] => e <= inv[x] - 1
                limit = inv[x] - 1
                if limit < e_max:
                    e_max = limit
            else:
                # Constraint: s > inv[x]
                # If inv[x] >= s, then x cannot be in the interval
                # So e must be < x
                if inv[x] >= s:
                    if x - 1 < e_max:
                        e_max = x - 1
            
            if e_max < s:
                break

        if e_max >= s:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Time check
    if time.time() - start_time > DEADLINE:
        pass

    # Verify the solution is valid
    end = best_start + best_len - 1
    valid = True
    for x in range(best_start, end + 1):
        for y in range(best_start, end + 1):
            if (x * y) % MOD == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: brute force search for any valid interval (should not happen)
        for s in range(lo_bound, hi_bound + 1):
            for e in range(hi_bound, s - 1, -1):
                ok = True
                for x in range(s, e + 1):
                    if (x * x) % MOD == 1:
                        ok = False
                        break
                    if inv[x] >= s and inv[x] <= e:
                        ok = False
                        break
                if ok:
                    return [s, e - s + 1]

    return [best_start, best_len]
# EVOLVE-BLOCK-END
