"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # We need to find 30 coefficients in {-1, 0, 1} such that
    # sum(c_i / i) is nonzero and has minimal absolute value.
    
    # Strategy: Use a more systematic search.
    # 1. Start with a good heuristic: try to cancel terms.
    # 2. Use simulated annealing or a stronger local search.
    # 3. Also, try to construct specific small values.
    
    # First, let's compute all possible small values we can get with subsets.
    # But 3^30 is too large.
    
    # Let's use a better local search with multiple restarts and a more aggressive move set.
    # Moves: single flip, double flip, triple flip.
    
    def evaluate(coeffs):
        total = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                total += Fraction(c, i)
        return total
    
    best_coeffs = None
    best_abs = Fraction(1, 1)
    
    # Try to find a very small value by construction.
    # For example, 1/2 - 1/3 = 1/6.
    # 1/2 - 1/3 - 1/7 = 1/6 - 1/7 = 1/42.
    # 1/2 - 1/3 - 1/7 + 1/43 = 1/42 - 1/43 = 1/(42*43).
    # This pattern can be extended.
    
    # Let's try to build such a sequence.
    # We want sum_{i in S} c_i / i to be small.
    
    # Start with a random configuration and do a thorough local search.
    # Use a deterministic seed for reproducibility.
    random.seed(12345)
    
    # Try several different starting points.
    starts = []
    # Start 1: The seed's starting point
    starts.append([0] * 28 + [1, -1])
    # Start 2: All zeros except a few
    starts.append([1, -1] + [0] * 28)
    starts.append([1, 0, -1] + [0] * 27)
    starts.append([1, 0, 0, -1] + [0] * 26)
    # Start 3: Random
    for _ in range(20):
        starts.append([random.choice([-1, 0, 1]) for _ in range(30)])
    
    for start in starts:
        coeffs = start[:]
        current = evaluate(coeffs)
        current_abs = abs(current)
        
        if current == 0:
            # Perturb if zero
            idx = random.randint(0, 29)
            coeffs[idx] = 1 if coeffs[idx] != 1 else -1
            current = evaluate(coeffs)
            current_abs = abs(current)
        
        if best_abs is None or current_abs < best_abs:
            best_abs = current_abs
            best_coeffs = coeffs[:]
        
        # Local search with single, double, and triple moves
        improved = True
        iterations = 0
        while improved and iterations < 500:
            improved = False
            iterations += 1
            
            # Single moves
            for i in range(30):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_total = evaluate(coeffs)
                    new_abs = abs(new_total)
                    if new_total != 0 and (best_abs is None or new_abs < best_abs):
                        best_abs = new_abs
                        best_coeffs = coeffs[:]
                        improved = True
                    else:
                        coeffs[i] = old_val
            
            # Double moves
            for i in range(30):
                for j in range(i + 1, 30):
                    old_i, old_j = coeffs[i], coeffs[j]
                    for new_i in [-1, 0, 1]:
                        for new_j in [-1, 0, 1]:
                            if new_i == old_i and new_j == old_j:
                                continue
                            coeffs[i] = new_i
                            coeffs[j] = new_j
                            new_total = evaluate(coeffs)
                            new_abs = abs(new_total)
                            if new_total != 0 and (best_abs is None or new_abs < best_abs):
                                best_abs = new_abs
                                best_coeffs = coeffs[:]
                                improved = True
                            else:
                                coeffs[i] = old_i
                                coeffs[j] = old_j
            
            # Triple moves (limited to keep runtime down)
            if iterations % 10 == 0:
                for i in range(30):
                    for j in range(i + 1, 30):
                        for k in range(j + 1, 30):
                            old_i, old_j, old_k = coeffs[i], coeffs[j], coeffs[k]
                            for new_i in [-1, 0, 1]:
                                for new_j in [-1, 0, 1]:
                                    for new_k in [-1, 0, 1]:
                                        if new_i == old_i and new_j == old_j and new_k == old_k:
                                            continue
                                        coeffs[i] = new_i
                                        coeffs[j] = new_j
                                        coeffs[k] = new_k
                                        new_total = evaluate(coeffs)
                                        new_abs = abs(new_total)
                                        if new_total != 0 and (best_abs is None or new_abs < best_abs):
                                            best_abs = new_abs
                                            best_coeffs = coeffs[:]
                                            improved = True
                                        else:
                                            coeffs[i] = old_i
                                            coeffs[j] = old_j
                                            coeffs[k] = old_k
    
    # Verify best is nonzero
    final_val = evaluate(best_coeffs)
    if final_val == 0:
        # Fallback
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
