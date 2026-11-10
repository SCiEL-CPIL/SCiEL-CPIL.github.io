import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def square_nonzero_count(coeffs):
        n = len(coeffs)
        nz = [(i, coeffs[i]) for i in range(n) if coeffs[i] != 0]
        sq = {}
        for i, ci in nz:
            for j, cj in nz:
                d = i + j
                sq[d] = sq.get(d, 0) + ci * cj
        return sum(1 for v in sq.values() if v != 0)

    def is_valid(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return sum(1 for c in coeffs if c != 0) == 12

    def make_polynomial(support, signs):
        if not support:
            return [1]
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for exp, sign in zip(support, signs):
            coeffs[exp] = sign
        return coeffs

    # Initial: alternating
    best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    best_score = square_nonzero_count(best_coeffs)

    def local_search(coeffs, time_limit):
        cur = coeffs[:]
        cur_score = square_nonzero_count(cur)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            nz_idx = [i for i in range(len(cur)) if cur[i] != 0]
            random.shuffle(nz_idx)
            for idx in nz_idx:
                old = cur[idx]
                cur[idx] = -old
                new_s = square_nonzero_count(cur)
                if new_s < cur_score:
                    cur_score = new_s
                    improved = True
                else:
                    cur[idx] = old
        return cur, cur_score

    # Local search on initial
    cur, cs = local_search(best_coeffs, time.time() + 5)
    if cs < best_score:
        best_score = cs
        best_coeffs = cur[:]

    # Random restarts
    while time.time() < deadline:
        support = sorted(random.sample(range(0, 41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        rc = make_polynomial(support, signs)
        if not is_valid(rc):
            continue
        rs = square_nonzero_count(rc)
        if rs < best_score:
            best_score = rs
            best_coeffs = rc[:]
            continue
        # Quick local search
        remaining = deadline - time.time()
        if remaining < 3:
            break
        cur, cs = local_search(rc, time.time() + min(2.0, remaining - 1))
        if cs < best_score:
            best_score = cs
            best_coeffs = cur[:]

    if not is_valid(best_coeffs):
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    return best_coeffs
# EVOLVE-BLOCK-END
