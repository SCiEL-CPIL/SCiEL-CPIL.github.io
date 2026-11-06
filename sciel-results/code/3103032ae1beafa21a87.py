import time
import random
import math
from itertools import product

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85.0

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    # Sumset with equal summands allowed
    sums = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            sums.add(ai + A_sorted[j])
    
    # Product set with equal factors allowed
    products = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            products.add(ai * A_sorted[j])
    
    return max(len(sums), len(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _rational_gp(start, p, q):
    """
    Generate a geometric progression with rational ratio p/q.
    A = {start * (p/q)^i for i=0..19}, rounded to nearest integer.
    Returns None if any element exceeds MAX_VAL or if not all distinct.
    """
    if q == 0:
        return None
    
    A = []
    current = start
    for i in range(20):
        A.append(current)
        # Next term: current * p / q
        next_val = current * p
        # Check if divisible by q, or just do integer division
        if next_val % q != 0:
            # Round to nearest integer for the next step
            # But we need to track the exact rational value
            # Let's use exact rational arithmetic
            pass
        
        # Use exact rational: current * p / q
        # We'll store as float for comparison but compute exactly
        current = (current * p) / q
        current_int = int(round(current))
        
        if current_int > MAX_VAL:
            return None
        if current_int < 1:
            return None
    
    # Convert to integers
    A_int = [int(round(x)) for x in A]
    
    # Check distinctness
    if len(set(A_int)) != 20:
        return None
    
    # Check bounds
    if min(A_int) < 1 or max(A_int) > MAX_VAL:
        return None
    
    return A_int

def _rational_gp_exact(start, p, q):
    """
    Generate geometric progression with exact rational arithmetic.
    A_i = start * p^i / q^i, rounded to nearest integer.
    """
    if q == 0 or q < 1 or p < 1:
        return None
    
    A = []
    for i in range(20):
        # A_i = start * (p/q)^i
        # Compute numerator and denominator
        num = start * (p ** i)
        den = q ** i
        val = num / den
        val_int = int(round(val))
        
        if val_int < 1 or val_int > MAX_VAL:
            return None
        A.append(val_int)
    
    if len(set(A)) != 20:
        return None
    
    return A

def _deterministic_search():
    """
    Grid search over rational ratios p/q and starting values.
    Rational ratios create multiplicative collisions in A*A.
    """
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Search over rational ratios p/q where p > q (ratio > 1)
    # and small integers to keep numbers manageable
    # Also try ratios close to 1 for compact sumset
    
    # Strategy 1: Ratios slightly above 1 (compact both sets)
    # Strategy 2: Ratios that are perfect powers (more collisions)
    
    ratio_candidates = []
    
    # Small ratios close to 1
    for q in range(1, 15):
        for p in range(q + 1, q + 15):
            ratio = p / q
            if ratio > 1.5:  # Don't want too large ratios
                continue
            ratio_candidates.append((p, q))
    
    # Ratios that are perfect powers (e.g., 4/2=2, 9/3=3, 8/4=2)
    for base in range(2, 10):
        for exp in range(1, 4):
            val = base ** exp
            if val <= 50:
                ratio_candidates.append((val, 1))
    
    # Remove duplicates
    seen = set()
    unique_ratios = []
    for p, q in ratio_candidates:
        # Reduce fraction
        g = math.gcd(p, q)
        rp, rq = p // g, q // g
        key = (rp, rq)
        if key not in seen:
            seen.add(key)
            unique_ratios.append(key)
    
    # Sort by ratio value for systematic search
    unique_ratios.sort(key=lambda x: x[0] / x[1])
    
    # Starting values to try
    # For small ratios, start can be small
    # For larger ratios, start must be small to fit in 10000
    start_candidates = list(range(1, 50))
    
    # Add some larger starts for very small ratios
    for s in range(50, 200, 10):
        start_candidates.append(s)
    
    # Time budget per (ratio, start) evaluation
    # _objective takes ~1-2ms for n=20
    # We have ~80 seconds, so we can evaluate ~40000-80000 combinations
    
    evaluated = 0
    for p, q in unique_ratios:
        if time.time() - start_time > TIME_LIMIT - 3.0:
            break
        
        for start in start_candidates:
            if time.time() - start_time > TIME_LIMIT - 3.0:
                break
            
            A = _rational_gp_exact(start, p, q)
            if A is None:
                continue
            
            evaluated += 1
            obj = _objective(A)
            
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
    
    # Also try some manually crafted sets
    # Powers of 2: {1, 2, 4, 8, ..., 2^19} - too large
    # Powers of 2 but truncated: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
    # That's only 13 elements. Need 20.
    
    # Try mixed: some powers of 2, some powers of 3
    # {1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96}
    # This is numbers of form 2^a * 3^b with a+b <= some limit
    
    # Generate numbers of form 2^a * 3^b
    def gen_2_3_powers():
        S = set()
        for a in range(0, 15):
            for b in range(0, 15):
                val = (2 ** a) * (3 ** b)
                if val <= MAX_VAL:
                    S.add(val)
        return sorted(S)
    
    # Take first 20 from 2^a * 3^b
    two_three = gen_2_3_powers()
    if len(two_three) >= 20:
        A_candidate = two_three[:20]
        obj = _objective(A_candidate)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A_candidate)
    
    # Try numbers of form 2^a * 5^b
    def gen_2_5_powers():
        S = set()
        for a in range(0, 15):
            for b in range(0, 15):
                val = (2 ** a) * (5 ** b)
                if val <= MAX_VAL:
                    S.add(val)
        return sorted(S)
    
    two_five = gen_2_5_powers()
    if len(two_five) >= 20:
        A_candidate = two_five[:20]
        obj = _objective(A_candidate)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A_candidate)
    
    # Try numbers with many small prime factors
    def gen_smooth_numbers():
        """Generate 5-smooth numbers (only primes 2,3,5)."""
        S = set()
        for a in range(0, 12):
            for b in range(0, 12):
                for c in range(0, 12):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= MAX_VAL:
                        S.add(val)
        return sorted(S)
    
    smooth = gen_smooth_numbers()
    if len(smooth) >= 20:
        A_candidate = smooth[:20]
        obj = _objective(A_candidate)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A_candidate)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def _local_refine(A, time_budget, rng):
    """Quick local refinement of a good starting point."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try small perturbations
        improved = False
        for delta in [-3, -2, -1, 1, 2, 3]:
            new_val = old_val + delta
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A and new_val != old_val:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                improved = True
                break
            else:
                A[idx] = old_val
        
        if not improved:
            # Try larger perturbation
            new_val = rng.randint(1, MAX_VAL)
            if new_val in A and new_val != old_val:
                continue
            A[idx] = new_val
            new_obj = _objective(A)
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
            else:
                A[idx] = old_val
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    
    try:
        # Deterministic grid search over rational GP
        best_A, best_obj = _deterministic_search()
        
        # Use remaining time for local refinement
        remaining = TIME_LIMIT - 3.0 - (time.time() - start_time)
        if remaining > 1:
            rng = random.Random(42)
            refined_A, refined_obj = _local_refine(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        
        if not _is_valid(best_A):
            best_A = list(range(1, 21))
            
        return sorted(best_A)
        
    except Exception:
        # Fallback
        return list(range(1, 21))

# EVOLVE-BLOCK-END
