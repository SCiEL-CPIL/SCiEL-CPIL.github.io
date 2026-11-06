import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 128
    # A is a subset of [0, 128]
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    
    # Fast evaluation using FFT-based convolution
    # For unordered pairs with a<=b:
    #   unordered(t) = (ordered_count(t) + diag(t)) / 2
    # where diag(t) = 1 if t even and t//2 in A
    
    def evaluate(A_set):
        """Return the maximum multiplicity over t in [128,256], and coverage info."""
        v = np.zeros(129, dtype=np.float64)
        for x in A_set:
            v[x] = 1
        # FFT-based convolution for speed
        N = 1 << (257 - 1).bit_length()  # next power of 2 >= 257
        V = np.fft.rfft(v, N)
        conv = np.fft.irfft(V * V, N)
        conv = np.round(conv).astype(np.int32)
        
        max_mult = 0
        covered = True
        for t in range(128, 257):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    # Start with a dense set and try to sparsify
    A = set(range(0, 129))  # all elements
    score, covered = evaluate(A)
    best_A = set(A)
    best_score = score
    best_covered = covered
    
    # Simulated annealing with faster evaluation
    import random
    random.seed(42)
    
    T = 30.0
    T_min = 0.05
    cooling = 0.998
    
    current_A = set(A)
    current_score, current_covered = evaluate(current_A)
    
    iteration = 0
    while time.time() < deadline and T > T_min:
        iteration += 1
        # Generate a move: add or remove a random element
        # Bias towards removal since we start dense
        if random.random() < 0.7 and len(current_A) > 1:
            # Remove
            x = random.choice(list(current_A))
            new_A = current_A - {x}
        else:
            # Add
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A = current_A | {x}
        
        new_score, new_covered = evaluate(new_A)
        
        # Score function: penalize lack of coverage heavily
        if new_covered:
            new_cost = new_score
        else:
            new_cost = new_score + 500  # heavy penalty for uncovered
        
        if current_covered:
            current_cost = current_score
        else:
            current_cost = current_score + 500
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_A = new_A
            current_score = new_score
            current_covered = new_covered
            
            if current_covered and (not best_covered or current_score < best_score):
                best_A = set(current_A)
                best_score = current_score
                best_covered = True
        
        T *= cooling
    
    # Local search refinement: try all single removals from best
    if best_covered:
        for x in list(best_A):
            if len(best_A) <= 1:
                break
            new_A = best_A - {x}
            s, c = evaluate(new_A)
            if c and s < best_score:
                best_A = new_A
                best_score = s
    
    # Ensure best is valid
    if not best_covered:
        # Fallback to a known valid solution
        best_A = set(range(64, 129))
        s, c = evaluate(best_A)
        if not c:
            best_A = set(range(0, 129))
    
    return sorted(best_A)
