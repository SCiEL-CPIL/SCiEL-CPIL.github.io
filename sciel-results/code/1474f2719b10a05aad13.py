import math
import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Calculate LCM of all divisors
    L = 1
    for d in DIVISORS:
        L = _lcm(L, d)
        
    best_range = float('inf')
    best_s = None
    best_vals = None
    
    # Iterate over all possible start offsets s in [0, L)
    # We can potentially optimize by checking only s where min(v_i) is small,
    # but L is small enough to check all.
    
    for s in range(1, L):  # Skip s=0 as it yields all zeros (conflict)
        # Check deadline
        if time.time() > deadline:
            break
            
        vals = []
        for d in DIVISORS:
            # Smallest multiple of d >= s
            # v = s + ((-s) % d)
            v = s + ((-s) % d)
            vals.append(v)
            
        # Check if all values are distinct
        if len(set(vals)) == len(vals):
            current_range = max(vals) - min(vals)
            if current_range < best_range:
                best_range = current_range
                best_s = s
                best_vals = vals[:]
                
    # Fallback if no conflict-free solution found (should not happen)
    if best_vals is None:
        # Use s=1 as a safe default
        s = 1
        best_vals = [s + ((-s) % d) for d in DIVISORS]
        # Resolve conflicts by picking next multiples if necessary
        # This is a simple heuristic fallback
        used = set()
        final_vals = []
        for i, d in enumerate(DIVISORS):
            v = best_vals[i]
            while v in used:
                v += d
            used.add(v)
            final_vals.append(v)
        best_vals = final_vals

    # Ensure values are positive and <= MAX_VAL
    # Since s >= 1, vals >= 1.
    # Max val < s + L <= L + L = 2L = 60060 < 1000000.
    
    # Convert to standard Python ints
    return [int(v) for v in best_vals]
# EVOLVE-BLOCK-END
