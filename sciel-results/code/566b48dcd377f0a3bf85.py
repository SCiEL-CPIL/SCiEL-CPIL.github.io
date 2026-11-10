import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for a set of integers A."""
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))

    # Initial candidate: Arithmetic Progression 1..20
    best_A = list(range(1, 21))
    best_score = evaluate(best_A)
    
    # Simulated Annealing
    T = 100.0
    cooling_rate = 0.9995
    current_A = list(best_A)
    current_score = best_score
    
    max_iterations = 1000000  # Upper bound, will be cut by time
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Generate a new candidate
        new_A = current_A[:]
        
        # Strategy: Replace one random element with a random integer
        # This allows escaping local minima in the AP structure
        idx = random.randint(0, 19)
        new_val = random.randint(1, 10000)
        
        # Ensure distinctness
        if new_val in new_A:
            # Try a few times to find a valid distinct value, or just skip
            # To keep speed high, we just pick another random one
            # Or we can pick from a list of valid candidates
            # For simplicity, if collision, retry logic is implicit in next iter, 
            # but let's try to pick a unique one quickly
            attempts = 0
            while new_val in new_A and attempts < 10:
                new_val = random.randint(1, 10000)
                attempts += 1
            if new_val in new_A:
                continue
        
        new_A[idx] = new_val
        new_score = evaluate(new_A)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Deterioration: accept with probability exp(-delta/T)
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6

    return best_A

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To adhere to the "Keep the function names" instruction, I will place the logic 
# inside the EVOLVE block or ensure the global construct is the one used.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `construct`.
# I will put the implementation in the EVOLVE block as requested by the structure 
# "Initial program ... SCiEL evolves the EVOLVE-BLOCK".
# However, the instruction says "Rewrite the program from scratch... Output the COMPLETE program".
# I will provide the complete program with the logic in the EVOLVE block.
# EVOLVE-BLOCK-END
