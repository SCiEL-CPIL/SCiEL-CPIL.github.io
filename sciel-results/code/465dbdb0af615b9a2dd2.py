"""Combined approach: systematic window search + efficient backtracking + local polish."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)

    # Initial naive solution
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]

    def check_window(W, L):
        """Check if window [L, L+W] has a valid distinct assignment."""
        if L < 1 or L + W > MAX_V:
            return None

        candidate_offsets = []
        for d in divisors:
            p = L % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None
            offsets = list(range(first_offset, W + 1, d))
            candidate_offsets.append(offsets)

        # Sort by most constrained first
        order = sorted(range(n), key=lambda i: len(candidate_offsets[i]))
        cand_sets = [set(candidate_offsets[i]) for i in range(n)]

        assignment = [None] * n
        used = set()

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            avail = [j for j in candidate_offsets[i] if j not in used]
            if not avail:
                return False
            # Try positions that conflict with fewer remaining variables first
            def conflicts(pos):
                c = 0
                for k in range(idx + 1, n):
                    if pos in cand_sets[order[k]]:
                        c += 1
                return c
            avail.sort(key=conflicts)
            for j in avail:
                used.add(j)
                assignment[i] = L + j
                if backtrack(idx + 1):
                    return True
                used.discard(j)
                assignment[i] = None
            return False

        if backtrack(0):
            return assignment[:]
        return None

    # Systematic search over W from theoretical minimum
    # Theoretical minimum is 9 (10 distinct integers need span >= 9)
    for W in range(9, 60):
        if time.time() > deadline:
            break

        # Due to LCM periodicity, we only need to check L in [1, min(LCM, MAX_V - W)]
        # But we can be smarter: for each divisor d, L mod d determines feasibility
        # Let's sample L values strategically

        max_L = min(LCM, MAX_V - W)

        # First try: check L values that are multiples of small numbers
        # to increase density of multiples in the window
        found = False
        # Try L = 1, then stride through
        stride = max(1, max_L // 3000)  # Check at most ~3000 L values
        for L in range(1, max_L + 1, stride):
            if time.time() > deadline:
                break

            # Quick check: all divisors must have a multiple in [L, L+W]
            valid = True
            for d in divisors:
                p = L % d
                first_offset = (d - p) % d
                if first_offset > W:
                    valid = False
                    break
            if not valid:
                continue

            result = check_window(W, L)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break

        if found and best_val <= W:
            # Can't do better than W for this W
            if best_val == W:
                break

    # Local search polish: try to tighten the span further
    if best_assign is not None:
        current_sol = best_assign[:]
        current_cost = max(current_sol) - min(current_sol)
        best_sol = current_sol[:]
        best_cost = current_cost

        # Multiple restarts with simulated annealing
        for sa_run in range(5):
            if time.time() > deadline:
                break

            if sa_run == 0:
                current_sol = best_sol[:]
                current_cost = best_cost
            else:
                # Perturb from best
                current_sol = best_sol[:]
                for _ in range(15):
                    idx = random.randint(0, n - 1)
                    d = divisors[idx]
                    delta = random.randint(-30, 30)
                    new_val = current_sol[idx] + delta * d
                    if 1 <= new_val <= MAX_V and all(current_sol[j] != new_val for j in range(n) if j != idx):
                        current_sol[idx] = new_val
                current_cost = max(current_sol) - min(current_sol)

            temp = 50.0
            cooling = 0.997

            while temp > 0.01 and time.time() < deadline:
                idx = random.randint(0, n - 1)
                d = divisors[idx]
                old_val = current_sol[idx]

                # Try nearby multiples
                for delta_k in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta_k * d
                    if new_val < 1 or new_val > MAX_V:
                        continue
                    if any(current_sol[j] == new_val for j in range(n) if j != idx):
                        continue

                    new_sol = current_sol[:]
                    new_sol[idx] = new_val
                    new_cost = max(new_sol) - min(new_sol)
                    delta_cost = new_cost - current_cost

                    if delta_cost < 0 or random.random() < min(1.0, __import__('math').exp(-delta_cost / max(temp, 0.01))):
                        current_sol = new_sol
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_sol = current_sol[:]
                        break  # Only try one move per iteration

                temp *= cooling

        if best_cost < best_val:
            best_val = best_cost
            best_assign = best_sol

    if best_assign is None:
        best_assign = divisors[:]

    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)

    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n

    return result

# EVOLVE-BLOCK-END
