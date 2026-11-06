import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 % MOD == 1 => x == 1 or x == MOD-1 = 1008
    self_inverse = set([1, 1008])

    # Two-pointer sweep: for each right endpoint R, find the maximal valid window [L, R]
    best_start = 1
    best_length = 0

    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break

        # Check if R itself is self-inverse
        if R in self_inverse:
            # Any window containing R is invalid; reset L past R
            L = R + 1
            continue

        # Check if inv[R] is in [L, R-1]
        # If so, we must push L past inv[R] to avoid the conflict pair
        if L <= inv[R] < R:
            L = inv[R] + 1

        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L

    # Verify the result with a direct check
    if best_length > 0:
        s_final = best_start
        e_final = s_final + best_length - 1
        valid = True
        for x in range(s_final, e_final + 1):
            if x in self_inverse:
                valid = False
                break
            if s_final <= inv[x] <= e_final:
                valid = False
                break
            if not valid:
                break

        if not valid:
            # Fallback: direct enumeration
            best_start = 1
            best_length = 0
            for s in range(1, N + 1):
                if time.time() > deadline:
                    break
                if s in self_inverse:
                    continue
                e_max = s
                for e in range(s + 1, N + 1):
                    if e in self_inverse:
                        break
                    if s <= inv[e] <= e:
                        break
                    e_max = e
                length = e_max - s + 1
                if length > best_length:
                    best_length = length
                    best_start = s

    if best_length == 0:
        for i in range(2, N + 1):
            if i not in self_inverse:
                return [i, 1]
        return [2, 1]

    return [best_start, best_length]
