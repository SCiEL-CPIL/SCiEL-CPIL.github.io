import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements (x^2 == 1 mod 1009): {1, 1008}
    self_inv = set()
    for x in range(1, MOD):
        if x * x % MOD == 1:
            self_inv.add(x)

    # For each x in [2, 1007] (non-self-inverse), the pair is {x, inv[x]}
    # a = min(x, inv[x]), b = max(x, inv[x])
    # For a fixed start S, E_max(S) = min{b : a >= S} - 1, capped at 1007

    # b_a[a] = max(a, inv[a]) for a in [2, 1007], or MOD if a is self-inverse
    b_a = [MOD] * (MOD + 1)
    for a in range(2, MOD):  # 2 to 1008
        if a in self_inv:
            b_a[a] = MOD
        else:
            b_a[a] = max(a, inv[a])

    # Suffix minimum: suff_min[S] = min{b_a[a] : a >= S}
    suff_min = [MOD] * (MOD + 2)
    suff_min[MOD] = MOD
    for a in range(MOD - 1, 1, -1):
        suff_min[a] = min(b_a[a], suff_min[a + 1])

    best_start = 2
    best_len = 0

    for S in range(2, MOD):  # S from 2 to 1007
        if S in self_inv:
            continue
        e_max = suff_min[S] - 1
        if e_max > 1007:
            e_max = 1007
        if e_max < S:
            continue
        current_len = e_max - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S

        if time.time() > DEADLINE:
            break

    # Verify the result
    S = best_start
    E = S + best_len - 1
    valid = True
    for x in range(S, E + 1):
        if x in self_inv:
            valid = False
            break
        if S <= inv[x] <= E:
            valid = False
            break

    if not valid:
        # Fallback: brute force
        for S2 in range(2, MOD):
            if S2 in self_inv:
                continue
            E2 = S2 - 1
            while E2 < 1007:
                new_E = E2 + 1
                if new_E in self_inv:
                    break
                partner = inv[new_E]
                if S2 <= partner <= E2:
                    break
                E2 = new_E
            if E2 >= S2:
                cl = E2 - S2 + 1
                if cl > best_len:
                    best_len = cl
                    best_start = S2
            if time.time() > DEADLINE:
                break

    return [best_start, best_len]
