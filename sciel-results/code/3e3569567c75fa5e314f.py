"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import random
from collections import deque

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def _evaluate(coeffs):
    """Return (numerator, denominator) of the signed sum."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1  # term is c / (i+1)
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def construct():
    deadline = time.time() + 55  # safety margin before 90s kill
    n = 30
    L = _lcm_range(n)  # lcm(1..30)
    weights = [L // (i + 1) for i in range(n)]
    
    # Trivial solution: 1/29 - 1/30 = 1/(29*30) = 1/870
    # N = L/29 - L/30 = L * (30-29)/(29*30) = L/870
    # But let's just compute: best_coeffs = [0]*28 + [1, -1]
    best_coeffs = [0] * 28 + [1, -1]
    best_N = sum(c * w for c, w in zip(best_coeffs, weights))
    best_abs_N = abs(best_N)
    
    # Meet-in-the-Middle Strategy
    # Split indices 0..29 into two sets of 15
    # Set A: indices 0..14
    # Set B: indices 15..29
    
    # Enumerate all 3^15 = 14,348,907 combinations for each set
    # This is feasible in Python if optimized, but might be tight.
    # 3^15 is ~14.3 million. 
    # Generating 14.3M tuples and sorting them might take a few seconds.
    # Let's try to be efficient.
    
    half = n // 2 # 15
    indices_A = list(range(half))
    indices_B = list(range(half, n))
    
    # Generate all possible sums for Set A
    # We need to map sum -> list of coefficients (or just one coefficient set that achieves it)
    # Actually, for MITM to find min |N|, we need:
    # N = sum_A + sum_B
    # We want min |sum_A + sum_B| != 0
    # This is equivalent to finding sum_A and sum_B such that sum_A is close to -sum_B.
    
    # Store sums for B in a dictionary: sum_val -> coeffs_B
    # If multiple coeffs give same sum, we just need one.
    # Wait, if sum_A + sum_B = 0, that's invalid (zero sum).
    # So we want the smallest non-zero |sum_A + sum_B|.
    
    # Algorithm:
    # 1. Generate all (sum_B, coeffs_B) for Set B. Store in dict `map_B`.
    #    If a sum appears multiple times, we don't need to store all, just one example.
    # 2. Generate all (sum_A, coeffs_A) for Set A.
    # 3. For each sum_A, we want to find sum_B in map_B such that |sum_A + sum_B| is minimized and non-zero.
    #    This is a "closest sum" problem.
    #    To do this efficiently:
    #    a. Extract all unique sum_B values, sort them.
    #    b. For each sum_A, perform binary search in sorted sums_B for -sum_A.
    #    c. Check the neighbors in the sorted list to find the closest non-zero sum.
    
    # Optimization: 
    # 3^15 is 14,348,907. 
    # Storing 14M entries in a Python dict is memory heavy (~1GB+?) and slow.
    # Let's check memory. 14M keys. Python overhead is high.
    # Maybe we can reduce the search space?
    # Or use a different split? No, 15/15 is optimal for MITM.
    
    # Alternative: 
    # Since we only care about the MINIMUM non-zero absolute value, and the weights are large,
    # the sums will be spread out.
    # Let's try to generate the lists and sort them.
    
    # To speed up generation, we can use a recursive generator or iterative base-3 counting.
    # Iterative is faster in Python if we avoid recursion overhead.
    
    def generate_sums(indices, weights):
        """Generate all possible sums and corresponding coefficients for a subset of indices."""
        # Number of combinations is 3^len(indices)
        # We will store (sum_val, coeffs_tuple)
        # To save memory, we can just store sum_val in a set, and reconstruct coeffs later?
        # No, we need the coeffs to return the final answer.
        # But we can store a mapping: sum_val -> coeffs_list
        # If we have collisions, we keep the first one.
        
        k = len(indices)
        # Precompute powers? No, just iterate.
        # 3^15 is too large for a simple list of tuples in standard memory limits if we're not careful?
        # 14M tuples of (int, list) is huge.
        # Let's store:
        # sums_B: sorted list of unique sums
        # coeffs_map: dict mapping sum -> coeffs (only for unique sums? No, multiple coeffs can give same sum)
        # Actually, for the "closest sum" step, we only need the set of unique sums from B.
        # Then, once we find the best (sum_A, sum_B) pair, we need to retrieve the coeffs.
        # So we need a way to retrieve coeffs for a specific sum.
        # We can store `sum_to_coeffs` dict.
        
        # Let's try to be memory efficient.
        # 14M entries.
        # Let's try it. If it's too slow, we fallback to local search.
        
        map_sum_to_coeffs = {}
        
        # Iterative generation
        # coeffs is a list of length k, values in {-1, 0, 1}
        # We can represent the state as an integer in base 3.
        # 0 -> -1, 1 -> 0, 2 -> 1
        
        # This might be slow in pure Python.
        # Let's use a recursive approach with pruning? No, no pruning possible for MITM.
        
        # Let's try a different approach: 
        # Since 3^15 is borderline, maybe we can use a smaller split? 
        # 10/20? 3^10 = 59049 (small). 3^20 = 3.4e9 (too big).
        # So 15/15 is the only feasible MITM split.
        
        # Let's optimize the inner loop.
        # We can generate the sums for B, store unique sums in a list, and a dict for coeffs.
        
        # To speed up generation, we can use numpy? 
        # "NumPy and SciPy are available."
        # Yes! Use NumPy to vectorize the sum generation?
        # Generating all 3^15 combinations as a 14M x 15 array of ints is 14M * 15 * 4 bytes = ~800MB. 
        # That might be too much memory.
        
        # Let's stick to pure Python but optimize.
        # Or, use a "Gray code" style update?
        
        # Actually, let's look at the weights.
        # weights[i] = L / (i+1)
        # These are large numbers.
        
        # Let's try a hybrid:
        # Use MITM for the first 15, but maybe we don't need to store ALL coeffs.
        # We just need to find the min |N|.
        
        # Let's try to run the MITM generation.
        # If it takes > 30s, we abort and use local search.
        
        start_time = time.time()
        
        # Generate Set B (indices 15..29)
        # We'll store unique sums and one coefficient vector for each.
        sums_B = {}
        
        # Helper to generate
        def gen_sums(indices, weights, start_idx=0, current_sum=0, current_coeffs=None):
            if current_coeffs is None:
                current_coeffs = [0] * len(indices)
            if start_idx == len(indices):
                # Store
                # If current_sum is already in dict, we don't need to update (any coeffs work)
                if current_sum not in sums_B:
                    sums_B[current_sum] = current_coeffs[:]
                return
            
            # Option 1: -1
            current_coeffs[start_idx] = -1
            gen_sums(indices, weights, start_idx + 1, current_sum - weights[start_idx], current_coeffs)
            
            # Option 2: 0
            current_coeffs[start_idx] = 0
            gen_sums(indices, weights, start_idx + 1, current_sum, current_coeffs)
            
            # Option 3: 1
            current_coeffs[start_idx] = 1
            gen_sums(indices, weights, start_idx + 1, current_sum + weights[start_idx], current_coeffs)
            
        # Recursion depth 15 is fine.
        # But 14M calls might be slow.
        # Let's use an iterative approach with a stack or just base-3 iteration.
        
        # Iterative base-3 is faster.
        # Total combos: 3^15
        # We can precompute the weights for the subset.
        
        weights_B = [weights[i] for i in indices_B]
        k_B = len(weights_B)
        
        # We will iterate through 0 to 3^k_B - 1
        # Convert to base 3 digits to get coefficients
        # This is O(3^k * k). 14M * 15 = 215M operations. Might be ~10-20s in Python.
        
        # Let's try to optimize by updating the sum incrementally?
        # No, base conversion is simple.
        
        # Let's try a different tactic: 
        # The problem is small enough that we can just do a smart local search if MITM is too slow.
        # But MITM is guaranteed to find the global optimum.
        # Let's attempt MITM with a time check.
        
        # To make it faster, we can use `itertools.product`?
        # itertools.product([-1,0,1], repeat=15)
        # This generates tuples. Summing them is the cost.
        
        # Let's try to generate the sums for B.
        # We'll use a list for sums and a dict for coeffs.
        
        # Actually, let's just use the local search from the current program but improved.
        # The current program scored 2848. 
        # The inspiration 3 (Karmarkar-Karp) scored 0.06 (bad).
        # The inspiration 1 scored 2591.
        
        # Let's try a **Simulated Annealing** approach with a very fast evaluation.
        # The bottleneck in the current program is likely the Python overhead of local search.
        # We can vectorize the evaluation?
        
        # Let's try a **Genetic Algorithm** or **Simulated Annealing** with NumPy.
        
        # NumPy approach:
        # Pop of size 1000.
        # Each individual is a vector of 30 ints in {-1,0,1}.
        # Compute sums for all individuals at once using matrix multiplication.
        
        import numpy as np
        
        # Weights as numpy array
        w_np = np.array(weights, dtype=np.int64)
        
        # Initialize population
        pop_size = 1000
        # Random init
        # Values in {-1, 0, 1}
        # Use random integers 0,1,2 then map
        pop = np.random.randint(0, 3, size=(pop_size, n))
        # Map 0->-1, 1->0, 2->1
        # pop = pop - 1 gives {-1, 0, 1}
        pop = pop - 1
        
        # Compute sums
        # sums = pop @ w_np
        sums = pop @ w_np
        
        # Find best non-zero
        # We want min |sum| where sum != 0
        abs_sums = np.abs(sums)
        # Mask out zeros
        # We need to be careful: if sum is 0, it's invalid.
        # We can set abs_sums for zero sums to a large value.
        abs_sums[sums == 0] = np.iinfo(np.int64).max
        
        # Find index of min
        best_idx = np.argmin(abs_sums)
        best_sum = sums[best_idx]
        best_coeffs_np = pop[best_idx]
        
        if abs(best_sum) < best_abs_N:
            best_N = best_sum
            best_abs_N = abs(best_sum)
            best_coeffs = best_coeffs_np.tolist()
        
        # Simulated Annealing
        # T = 1000
        # T_min = 1
        # alpha = 0.99
        # Max iterations: until deadline
        
        T = 1000.0
        T_min = 1.0
        alpha = 0.99
        step_size = 1 # probability of changing a gene? No, just random restarts or flips.
        
        # We will do a simple SA loop.
        # At each step, pick a random individual, mutate it (flip a few genes), evaluate.
        # To speed up, we can mutate a whole row and recompute only that row's sum?
        # Or mutate a batch.
        
        # Let's mutate one individual at a time to keep it simple and fast enough.
        # 1000 individuals, 60s.
        # We can do ~1000-5000 iterations per second in Python if simple.
        
        # Better: Mutate a batch of individuals at each step.
        
        current_best_idx = best_idx
        current_best_sum = best_sum
        
        # To allow escaping local minima, we need to accept worse moves.
        # But we track the global best.
        
        # Let's run SA for the remaining time.
        while time.time() < deadline - 5:
            T *= alpha
            if T < T_min:
                T = T_min
                
            # Pick a random batch of individuals to mutate
            batch_size = 100
            indices_to_mutate = np.random.choice(pop_size, batch_size, replace=False)
            
            # For each, pick a random gene to flip/change
            # We can vectorize the mutation?
            # Pick random gene index for each individual in the batch
            gene_indices = np.random.randint(0, n, size=batch_size)
            
            # Get current values
            # old_vals = pop[indices_to_mutate, gene_indices]
            
            # Generate new values in {-1, 0, 1}
            # We can just pick a random new value different from old?
            # Or just random from {-1, 0, 1}.
            
            # Let's try: new_val = random choice from {-1, 0, 1}
            # We can use numpy random to generate these.
            new_vals = np.random.randint(0, 3, size=batch_size) - 1
            
            # Update pop
            pop[indices_to_mutate, gene_indices] = new_vals
            
            # Recompute sums for these individuals
            # sums[indices_to_mutate] = pop[indices_to_mutate] @ w_np
            # This is a batch matrix-vector product.
            new_sums = pop[indices_to_mutate] @ w_np
            sums[indices_to_mutate] = new_sums
            
            # Update best
            # Check if any of these new sums are better
            # We need to check |sum| < current_best_abs and sum != 0
            # Also, if we find a zero sum, we should avoid it? 
            # The task says "nonzero". 
            # If we hit 0, it's invalid. 
            # In SA, if we hit 0, we treat it as "very bad" (high cost) so we don't accept it as best.
            
            abs_new_sums = np.abs(new_sums)
            # Mask zeros
            abs_new_sums[new_sums == 0] = np.iinfo(np.int64).max
            
            min_idx_in_batch = np.argmin(abs_new_sums)
            min_abs_val = abs_new_sums[min_idx_in_batch]
            
            if min_abs_val < abs(best_N):
                # Update global best
                best_N = new_sums[min_idx_in_batch]
                best_abs_N = min_abs_val
                best_coeffs = pop[indices_to_mutate[min_idx_in_batch]].tolist()
                
                # Also update the current best in the population for SA reference?
                # Not strictly necessary if we track global best.
                
                # Optional: If we found a very small value, we can restart SA around it?
                # For now, just continue.
                
                if best_abs_N == 1:
                    break
            
            # SA Acceptance: 
            # We should accept worse moves with probability exp(-delta/T)
            # But since we are doing batch updates and only tracking global best, 
            # standard SA logic (accept/reject per move) is lost if we just overwrite.
            # To do proper SA, we should keep a "current" state and "best" state.
            
            # Let's refine: 
            # Keep `current_pop` and `best_pop`.
            # Actually, the current approach is more like "Random Restart Hill Climbing" with a population.
            # It might get stuck.
            
            # To improve, let's add a "restart" mechanism if no improvement for a while.
            
        # Final check
        # Ensure best_N is nonzero
        if best_N == 0:
            # Fallback to trivial
            return [0] * 28 + [1, -1]
            
        # Convert to num, den
        num = best_N
        den = L
        g = gcd(abs(num), den)
        if g:
            num //= g
            den //= g
            
        if den < 0:
            num = -num
            den = -den
            
        return best_coeffs

# EVOLVE-BLOCK-END
