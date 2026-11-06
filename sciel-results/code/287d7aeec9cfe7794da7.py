import time
import random
from math import gcd

def _evaluate(coeffs):
    """Compute the signed harmonic sum as (num, den) in reduced form."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def _abs_val(num, den):
    return abs(num) / den

def construct():
    deadline = time.time() + 80
    best_val = float('inf')
    best_coeffs = [0] * 28 + [1, -1] # Default: 1/29 - 1/30 = 1/870
    
    def try_update(coeffs):
        nonlocal best_val, best_coeffs
        num, den = _evaluate(coeffs)
        if num == 0:
            return
        val = _abs_val(num, den)
        if val < best_val:
            best_val = val
            best_coeffs = coeffs[:]

    def local_search(start_coeffs):
        coeffs = start_coeffs[:]
        current_num, current_den = _evaluate(coeffs)
        
        if current_num == 0:
            # Try to kick out of zero by picking a random non-zero move
            # Pick a random index and set to 1 or -1
            idx = random.randint(0, 29)
            coeffs[idx] = 1 if coeffs[idx] != 1 else -1
            current_num, current_den = _evaluate(coeffs)
            if current_num == 0:
                # Still zero? Unlikely but possible. Try another.
                idx2 = random.randint(0, 29)
                if idx2 != idx:
                    coeffs[idx2] = 1 if coeffs[idx2] != 1 else -1
                    current_num, current_den = _evaluate(coeffs)
                else:
                    coeffs[idx] = -1 if coeffs[idx] == 1 else 1
                    current_num, current_den = _evaluate(coeffs)
            
            if current_num == 0:
                return # Can't escape zero easily, skip

        current_val = _abs_val(current_num, current_den)
        try_update(coeffs)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                original_c = coeffs[i]
                # Try changing to other two values
                candidates = [c for c in [-1, 0, 1] if c != original_c]
                random.shuffle(candidates) # Randomize order to escape symmetry?
                
                for new_c in candidates:
                    delta_c = new_c - original_c
                    if delta_c == 0: continue
                    
                    # Update sum: current + delta_c / (i+1)
                    d = i + 1
                    g = gcd(current_den, d)
                    new_den = current_den // g * d
                    new_num = current_num * (d // g) + delta_c * (new_den // d)
                    
                    if new_num == 0:
                        # Don't accept zero as improvement over non-zero
                        continue
                    
                    new_val = _abs_val(new_num, new_den)
                    if new_val < current_val:
                        # Accept
                        coeffs[i] = new_c
                        current_num = new_num
                        current_den = new_den
                        current_val = new_val
                        improved = True
                        try_update(coeffs)
                        break # Restart for loop
                    
                if improved:
                    break # Restart while loop

    # 1. Greedy Initialization
    g_coeffs = [0] * 30
    g_num = 0
    g_den = 1
    for i in range(30):
        best_c = 0
        best_val = float('inf')
        for c in [-1, 0, 1]:
            d = i + 1
            g = gcd(g_den, d)
            new_den = g_den // g * d
            new_num = g_num * (d // g) + c * (new_den // d)
            if new_num == 0:
                v = float('inf')
            else:
                v = abs(new_num) / new_den
            if v < best_val:
                best_val = v
                best_c = c
        g_coeffs[i] = best_c
        d = i + 1
        g = gcd(g_den, d)
        g_den = g_den // g * d
        g_num = g_num * (d // g) + best_c * (g_den // d)
        
    local_search(g_coeffs)
    
    # 2. Random Restarts
    while time.time() < deadline - 2: # Leave 2s margin
        r_coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        local_search(r_coeffs)

    return best_coeffs

# EVOLVE-BLOCK-END
