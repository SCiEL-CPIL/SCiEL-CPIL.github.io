"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    # We restrict to arithmetic progression supports 0..11 for simplicity and optimality of support size.
    # P(x) = sum_{i=0}^{11} a_i x^i
    # P(x)^2 = sum_{k=0}^{22} c_k x^k
    # c_k = sum_{i+j=k} a_i a_j
    
    def count_nonzero_square(coeffs):
        """Calculate the number of non-zero coefficients in the square of the polynomial."""
        # coeffs is a list of 12 integers
        # Square has degree 22
        square_coeffs = [0] * 23
        for i in range(12):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, 12):
                aj = coeffs[j]
                if aj == 0:
                    continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    # Simulated Annealing / Random Restart Local Search
    # State: list of 12 integers in [-64, 64] \ {0}
    
    best_count = 23  # Max possible is 23
    best_poly = None
    
    # Initialize with a few good starting points
    # 1. All 1s
    # 2. Random integers
    # 3. Alternating signs with random magnitudes
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial state
        # Use random integers, biased towards small values to allow easier cancellation?
        # Or uniform? Uniform is safer.
        # Ensure non-zero
        state = [random.randint(1, 64) * random.choice([-1, 1]) for _ in range(12)]
        
        current_count = count_nonzero_square(state)
        
        if current_count < best_count:
            best_count = current_count
            best_poly = state[:]
            if best_count <= 5: # Theoretical lower bound is likely higher, but 5 is very good
                break

        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        # We want to minimize count.
        # Neighborhood: pick a random index, change value to a random new value in [-64, 64] \ {0}
        
        while T > T_min:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            idx = random.randint(0, 11)
            old_val = state[idx]
            new_val = random.randint(1, 64) * random.choice([-1, 1])
            if new_val == old_val:
                continue
                
            state[idx] = new_val
            new_count = count_nonzero_square(state)
            
            delta = new_count - current_count
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_poly = state[:]
                    if best_count <= 5:
                        break
            else:
                # Revert
                state[idx] = old_val
                
            T *= cooling
            
        if best_count <= 5:
            break

    # If we haven't found a good one, try a more targeted search with small integers
    # Sometimes small integers cancel more easily.
    if best_count > 10:
        for r in range(100):
            if time.time() > deadline:
                break
            state = [random.randint(1, 10) * random.choice([-1, 1]) for _ in range(12)]
            current_count = count_nonzero_square(state)
            if current_count < best_count:
                best_count = current_count
                best_poly = state[:]
            
            # Local search on small integers
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    for new_val in range(-10, 11):
                        if new_val == 0 or new_val == state[i]:
                            continue
                        old_val = state[i]
                        state[i] = new_val
                        new_count = count_nonzero_square(state)
                        if new_count < best_count:
                            best_count = new_count
                            best_poly = state[:]
                            improved = True
                        else:
                            state[i] = old_val
                if best_count <= 5:
                    break

    # Convert best_poly to the required format
    # The task requires:
    # 1. Coefficient list in ascending degree.
    # 2. Length <= 41.
    # 3. Nonzero final coefficient.
    # 4. Entries integers in [-64, 64].
    # 5. Exactly 12 nonzero terms in the ORIGINAL polynomial.
    
    # Our best_poly is a list of 12 coefficients for x^0 to x^11.
    # We need to return this list.
    # Check constraints:
    # - Are there exactly 12 nonzero terms? Yes, we ensured non-zero in generation.
    # - Are entries in [-64, 64]? Yes.
    # - Is the final coefficient nonzero? Yes.
    # - Length is 12, which is <= 41.
    
    # However, we must ensure that the "square" minimization was on the EXACT integer square.
    # Our count_nonzero_square does exactly that.
    
    # One edge case: What if best_poly is None?
    if best_poly is None:
        # Fallback: return a simple valid polynomial
        # 1 + x + ... + x^11
        return [1] * 12
    
    # Ensure the list is trimmed/adjusted if necessary, though our construction is tight.
    # The problem says "Return coefficient list in ascending degree".
    # If we used x^0 to x^11, the list is [a0, a1, ..., a11].
    
    # Double check: "exactly 12 nonzero terms".
    # Our state always has 12 non-zero integers.
    
    return best_poly

# EVOLVE-BLOCK-END
