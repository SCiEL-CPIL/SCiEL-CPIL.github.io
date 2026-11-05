import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    terms = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_abs_sum = None
    best_coeffs = None
    
    def calc_sum(coeffs):
        s = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * terms[i]
        return s
        
    def local_search(init_coeffs):
        coeffs = init_coeffs[:]
        current_sum = calc_sum(coeffs)
        current_abs = abs(current_sum)
        
        # 1-flip local search
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if coeffs[i] == 0:
                    # Try 1
                    new_sum = current_sum + terms[i]
                    if abs(new_sum) < current_abs:
                        current_abs = abs(new_sum)
                        current_sum = new_sum
                        coeffs[i] = 1
                        improved = True
                        break
                    # Try -1
                    new_sum = current_sum - terms[i]
                    if abs(new_sum) < current_abs:
                        current_abs = abs(new_sum)
                        current_sum = new_sum
                        coeffs[i] = -1
                        improved = True
                        break
                else:
                    # Flip to 0
                    new_sum = current_sum - coeffs[i] * terms[i]
                    if abs(new_sum) < current_abs:
                        current_abs = abs(new_sum)
                        current_sum = new_sum
                        coeffs[i] = 0
                        improved = True
                        break
                    # Flip to opposite
                    new_sum = current_sum - 2 * coeffs[i] * terms[i]
                    if abs(new_sum) < current_abs:
                        current_abs = abs(new_sum)
                        current_sum = new_sum
                        coeffs[i] = -coeffs[i]
                        improved = True
                        break
                        
        # 2-flip moves to escape local minima
        # Try a limited number of random 2-flips
        for _ in range(10):
            if time.time() > deadline:
                break
            i, j = random.sample(range(n), 2)
            # Try all combinations of coeffs[i], coeffs[j]
            orig_i, orig_j = coeffs[i], coeffs[j]
            for new_i in [-1, 0, 1]:
                for new_j in [-1, 0, 1]:
                    if new_i == orig_i and new_j == orig_j:
                        continue
                    # Calculate new sum
                    delta = (new_i - orig_i) * terms[i] + (new_j - orig_j) * terms[j]
                    new_sum = current_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        current_sum = new_sum
                        coeffs[i] = new_i
                        coeffs[j] = new_j
                        # Restart 1-flip search
                        local_search(coeffs) # Recursive? No, better to loop
                        # Break out of 2-flip loop and restart 1-flip
                        # We'll just return and let the outer loop handle it?
                        # No, we should continue searching from the new state.
                        # Let's just update and break to restart 1-flip
                        # We can't easily restart the while loop from inside.
                        # Let's just return the current best from this restart.
                        return coeffs, current_abs
        return coeffs, current_abs

    while time.time() < deadline:
        init_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        coeffs, abs_sum = local_search(init_coeffs)
        
        if abs_sum != 0:
            if best_abs_sum is None or abs_sum < best_abs_sum:
                best_abs_sum = abs_sum
                best_coeffs = coeffs
        else:
            # If we hit 0, we need the closest non-zero.
            # We can try flipping one term to get the smallest non-zero.
            # This is a local search starting from 0.
            # But we want non-zero.
            # We can just keep searching.
            pass

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
