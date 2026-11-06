import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    N = 512
    # Precompute weights
    w = {a: Fraction(1, N - a) for a in range(1, N)}

    def score(s):
        return sum(w[a] for a in s)

    # Prime factor bitmask for fast coprime checks
    # For each a in 1..511, compute set of prime factors
    primes_under_512 = []
    for p in range(2, 512):
        is_prime = True
        for d in range(2, int(math.isqrt(p)) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            primes_under_512.append(p)

    # Map each prime to an index
    prime_to_idx = {p: i for i, p in enumerate(primes_under_512)}

    # For each a, compute bitmask of its prime factors
    def factor_mask(a):
        mask = 0
        n = a
        for i, p in enumerate(primes_under_512):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            # n is prime
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        return mask

    masks = {a: factor_mask(a) for a in range(1, N)}

    def is_coprime(a, b):
        return (masks[a] & masks[b]) == 0

    # Greedy initialization: pick largest coprime set (by weight descending)
    chosen = set()
    for a in range(511, 0, -1):
        if all(is_coprime(a, b) for b in chosen):
            chosen.add(a)

    best = set(chosen)
    best_score = score(chosen)

    # Try MILP for exact solution using scipy
    milp_success = False
    if elapsed() < deadline - 30:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np

            # Build conflict graph: nodes 0..510 represent a=1..511
            # x[a-1] = 1 if we select a
            # Constraints: for each pair (a,b) with gcd(a,b)>1, x[a-1] + x[b-1] <= 1
            # Objective: maximize sum w[a] * x[a-1]

            # Build list of conflicting pairs
            conflicts = []
            for a in range(1, N):
                for b in range(a + 1, N):
                    if not is_coprime(a, b):
                        conflicts.append((a - 1, b - 1))

            n_vars = N - 1  # 511 variables

            # Objective: minimize -sum(w[a] * x[a-1])
            # Convert Fraction weights to float for MILP
            obj_coeffs = np.array([-float(w[a]) for a in range(1, N)])

            # Constraints: x[a-1] + x[b-1] <= 1 for each conflict
            if conflicts:
                # Build sparse-like constraint matrix
                # Number of constraints = len(conflicts)
                n_cons = len(conflicts)
                # Use dense matrix for simplicity (511x511x~conflicts could be large)
                # Actually, let's use a different approach: build row by row
                # Each row has two non-zero entries

                # For scipy milp, we need A_ub, b_ub where A_ub @ x <= b_ub
                # This could be very large. Let's estimate: number of conflicting pairs
                # For 511 numbers, worst case ~511*510/2 = ~130k constraints
                # That's manageable

                A_rows = []
                for (i, j) in conflicts:
                    row = [0.0] * n_vars
                    row[i] = 1.0
                    row[j] = 1.0
                    A_rows.append(row)

                A_ub = np.array(A_rows)
                b_ub = np.ones(n_cons)
            else:
                A_ub = None
                b_ub = None

            # Bounds: 0 <= x <= 1
            lb = np.zeros(n_vars)
            ub = np.ones(n_vars)

            # Integrate all variables
            integrality = np.ones(n_vars)

            constraints = [LinearConstraint(A_ub, None, b_ub)] if A_ub is not None else []

            result = milp(
                c=obj_coeffs,
                constraints=constraints,
                integrality=integrality,
                bounds=Bounds(lb, ub),
                options={"time_limit": deadline - elapsed() - 5}
            )

            if result.status == 0 or result.status == 1:  # optimal or near-optimal
                x = result.x
                milp_set = set()
                for idx in range(n_vars):
                    if x[idx] > 0.5:
                        a_val = idx + 1
                        milp_set.add(a_val)

                # Verify pairwise coprime
                valid = True
                milp_list = list(milp_set)
                for i in range(len(milp_list)):
                    for j in range(i + 1, len(milp_list)):
                        if not is_coprime(milp_list[i], milp_list[j]):
                            valid = False
                            break
                    if not valid:
                        break

                if valid and len(milp_set) > 0:
                    milp_score = score(milp_set)
                    if milp_score > best_score:
                        best = milp_set
                        best_score = milp_score
                        milp_success = True

        except Exception as e:
            milp_success = False

    # Local search polish (1-swap moves)
    if elapsed() < deadline - 5:
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            cur = set(best)
            best_gain = Fraction(0)
            best_move = None

            # 1-swap: remove x in cur, add y not in cur
            for x in list(cur):
                rest = cur - {x}
                for y in range(511, 0, -1):
                    if y in cur:
                        continue
                    if y <= x:
                        continue
                    if all(is_coprime(y, z) for z in rest):
                        gain = w[y] - w[x]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (x, y)

            if best_move is not None:
                x, y = best_move
                best.remove(x)
                best.add(y)
                best_score = score(best)
                improved = True

    # Simulated annealing polish
    if elapsed() < deadline - 8:
        current = set(best)
        current_score = best_score
        T = Fraction(1, 1000)
        T_min = Fraction(1, 1000000)
        cooling = 0.995
        while T > T_min and elapsed() < deadline - 3:
            # random 1-swap
            cur_list = list(current)
            x = random.choice(cur_list)
            rest = current - {x}
            # pick random y coprime to rest
            cands = [y for y in range(1, 512)
                     if y not in current and all(is_coprime(y, z) for z in rest)]
            if not cands:
                T *= cooling
                continue
            y = random.choice(cands)
            new_set = set(current)
            new_set.remove(x)
            new_set.add(y)
            new_score = score(new_set)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(float(delta) / float(T)):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
            T *= cooling

    return list(best)
