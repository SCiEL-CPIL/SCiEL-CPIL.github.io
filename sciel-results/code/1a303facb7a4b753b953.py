"""Program for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    n = 30
    
    # The seed gives [0]*28 + [1, -1], which means:
    # c_29 = 1, c_30 = -1, others 0
    # S = 1/29 - 1/30 = (30-29)/(29*30) = 1/870
    # So the seed value is 1/870 ≈ 0.001149...
    
    # To beat this, we need |S| < 1/870 with S != 0.
    
    # We'll use a meet-in-the-middle approach.
    # Split the 30 coefficients into two groups of 15.
    # For each group, compute all possible sums (3^15 = 14,348,907 possibilities).
    # Actually, 3^15 is about 14 million, which might be too slow in Python with Fractions.
    # Let's try a smaller split or a more efficient approach.
    
    # Alternative: Use the fact that we're minimizing |S| where S = sum c_i / i.
    # This is equivalent to finding a subset of {1/i} with signs that sums to a small nonzero value.
    
    # Let's try a smarter local search or simulated annealing approach.
    # Or we can try exhaustive search over small support sizes with early termination.
    
    # First, let's establish the baseline: seed gives 1/870.
    seed_coeffs = [0] * 28 + [1, -1]
    seed_sum = Fraction(1, 29) - Fraction(1, 30)  # = 1/870
    best_abs = abs(seed_sum)
    best_coeffs = seed_coeffs[:]
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        return s
    
    # Strategy: Systematic search over increasing support sizes
    # Support size 1: min |1/i| = 1/30 > 1/870, so no improvement possible
    # Support size 2: Already analyzed, best is 1/870 (the seed)
    
    # Support size 3:
    # C(30,3) * 3^3 = 4060 * 27 = 109,620 possibilities
    # This is feasible.
    
    found_better = False
    for triple in combinations(range(n), 3):
        if time.time() > deadline:
            break
        i, j, k = triple
        di, dj, dk = i + 1, j + 1, k + 1
        for a, b, c in product((-1, 1), repeat=3):
            s = Fraction(a, di) + Fraction(b, dj) + Fraction(c, dk)
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs:
                    best_abs = abs_s
                    best_coeffs = [0] * n
                    best_coeffs[i] = a
                    best_coeffs[j] = b
                    best_coeffs[k] = c
                    found_better = True
    
    if found_better:
        return best_coeffs
    
    # Support size 4:
    # C(30,4) * 3^4 = 27405 * 81 = 2,219,805 possibilities
    # This is also feasible.
    
    found_better = False
    for quad in combinations(range(n), 4):
        if time.time() > deadline:
            break
        i, j, k, l = quad
        di, dj, dk, dl = i + 1, j + 1, k + 1, l + 1
        for a, b, c, d in product((-1, 1), repeat=4):
            s = (Fraction(a, di) + Fraction(b, dj) + 
                 Fraction(c, dk) + Fraction(d, dl))
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs:
                    best_abs = abs_s
                    best_coeffs = [0] * n
                    best_coeffs[i] = a
                    best_coeffs[j] = b
                    best_coeffs[k] = c
                    best_coeffs[l] = d
                    found_better = True
    
    if found_better:
        return best_coeffs
    
    # Support size 5:
    # C(30,5) * 3^5 = 142506 * 243 = 34,628,958 possibilities
    # This might be too slow with Fractions. Let's try a different approach.
    
    # Use meet-in-the-middle for larger supports.
    # Split 30 variables into two sets of 15.
    # For each set, enumerate all 3^15 = 14,348,907 assignments.
    # For each assignment, compute the sum.
    # Then find pairs of sums from the two sets that sum to a small nonzero value.
    
    # This is still quite large. Let's try a more targeted approach.
    
    # Greedy local search with multiple restarts.
    
    import random
    random.seed(42)
    
    def local_search(coeffs, max_iter=10000):
        """Perform local search from initial configuration."""
        current_sum = compute_sum(coeffs)
        best_abs_local = abs(current_sum) if current_sum != 0 else float('inf')
        best_coeffs_local = coeffs[:]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Try changing one coefficient
            i = random.randint(0, n - 1)
            old_val = coeffs[i]
            new_val = random.choice([v for v in (-1, 0, 1) if v != old_val])
            coeffs[i] = new_val
            s = compute_sum(coeffs)
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs_local:
                    best_abs_local = abs_s
                    best_coeffs_local = coeffs[:]
            else:
                # If sum is zero, this is invalid, but we can still use it as an intermediate
                pass
            # With some probability, revert (simulated annealing style)
            if random.random() < 0.1:
                coeffs[i] = old_val
        
        return best_coeffs_local, best_abs_local
    
    # Run local search from the seed and several random initializations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            init_coeffs = seed_coeffs[:]
        else:
            init_coeffs = [random.choice((-1, 0, 1)) for _ in range(n)]
            # Ensure the sum is not zero initially (if possible)
            s = compute_sum(init_coeffs)
            if s == 0:
                init_coeffs[random.randint(0, n-1)] = random.choice((-1, 1))
        
        best_local, best_local_abs = local_search(init_coeffs, max_iter=5000)
        
        if best_local_abs < best_abs:
            best_abs = best_local_abs
            best_coeffs = best_local
    
    # Return the best configuration found
    return best_coeffs

# EVOLVE-BLOCK-END
