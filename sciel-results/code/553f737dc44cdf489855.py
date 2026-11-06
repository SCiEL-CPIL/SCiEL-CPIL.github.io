import time


def construct():
    p = 1009
    n = 1008
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Identify self-inverse residues (x*x % p == 1) and partner pairs
    is_self_inv = [False] * (p + 1)
    partner = [None] * (p + 1)
    for x in range(1, p):
        ix = inv[x]
        if ix == x:
            is_self_inv[x] = True
        else:
            partner[x] = ix

    # Two-pointer sliding window:
    # For each right endpoint R, find the smallest L such that [L, R] is valid.
    # A window is invalid if it contains a self-inverse element or a full inverse pair.
    best_start = 2
    best_len = 1

    L = 1
    present_count = [0] * (p + 1)
    conflict_count = 0

    for R in range(1, p):
        # Add R to the window
        present_count[R] += 1
        if is_self_inv[R]:
            conflict_count += 1
        else:
            pr = partner[R]
            if pr is not None and present_count[pr] > 0:
                conflict_count += 1

        # Shrink L from the left until the window is valid
        while conflict_count > 0 and L <= R:
            if is_self_inv[L]:
                conflict_count -= 1
            else:
                pr = partner[L]
                if pr is not None and present_count[pr] > 0:
                    conflict_count -= 1
            present_count[L] -= 1
            L += 1

        # Now [L, R] is the longest valid window ending at R
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L

        # Time guard (should never trigger for p=1009)
        if time.time() > deadline:
            break

    # Verify the result with a direct check
    end = best_start + best_len - 1
    valid = True
    if best_start < 1 or end > 1008 or best_len <= 0:
        valid = False
    else:
        present = set(range(best_start, end + 1))
        for x in present:
            if (x * x) % p == 1:
                valid = False
                break
            if inv[x] in present:
                valid = False
                break

    if valid:
        return [best_start, best_len]

    # Fallback: return a trivially valid 1-element interval
    return [2, 1]
