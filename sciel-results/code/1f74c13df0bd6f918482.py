"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 30
    best_val = None
    best_coeffs = None
    best_frac = None

    # Helper to evaluate a coefficient vector
    def evaluate(coeffs):
        # coeffs is list of 30 ints in {-1, 0, 1}
        # Sum: coeffs[0]/1 + coeffs[1]/2 + ... + coeffs[29]/30
        # Must be nonzero
        num = 0
        den = 1
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            term = Fraction(c, i + 1)
            num, den = num * (i + 1) + c * den, den * (i + 1)
            # Simplify
            from math import gcd
            g = gcd(abs(num), den)
            if g > 1:
                num //= g
                den //= g
        if num == 0:
            return None
        return abs(num) / den, num, den

    # Initial seed: the trivial solution [0]*28 + [1, -1]
    seed = [0] * 28 + [1, -1]
    val, num, den = evaluate(seed)
    if val is not None:
        best_val = val
        best_coeffs = seed[:]
        best_frac = (num, den)

    # Random local search
    rng = random.Random(42)
    iterations = 0
    while time.time() < deadline - 2.0:
        # Start from current best or a random point
        if iterations < 100 or rng.random() < 0.1:
            # Random restart: sparse random vector
            coeffs = [0] * N
            # Ensure at least 2 non-zero entries to have a chance of small sum
            n_nonzero = rng.randint(2, min(10, N))
            indices = rng.sample(range(N), n_nonzero)
            for idx in indices:
                coeffs[idx] = rng.choice([-1, 1])
        else:
            coeffs = best_coeffs[:]

        # Local improvement: try flipping one coefficient
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                for new_val in [-1, 0, 1]:
                    if new_val == coeffs[i]:
                        continue
                    coeffs[i] = new_val
                    res = evaluate(coeffs)
                    if res is not None:
                        v, n, d = res
                        if best_val is None or v < best_val:
                            best_val = v
                            best_coeffs = coeffs[:]
                            best_frac = (n, d)
                            improved = True
                            break
                if improved:
                    break
            # Reset to best if we made an improvement
            if improved:
                coeffs = best_coeffs[:]

        iterations += 1

    return best_coeffs
# EVOLVE-BLOCK-END
