import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial parameters
    M = 2000
    max_M = 100000
    
    # Initial residues: try to cover early numbers
    # Heuristic: r_p = 0 or small
    residues = [0] * n_primes
    
    # Initialize counts
    counts = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            counts[p:M+1:p] += 1
        else:
            counts[r:M+1:p] += 1
            
    def get_L(counts, M):
        # Find first x in 1..M where count < 2
        # counts is 0-indexed, so counts[1] is count for 1
        # We want the largest L such that 1..L are covered.
        # So we look for the first index i >= 1 where counts[i] < 2.
        # L = i - 1.
        # If no such index, L = M.
        
        # Efficient check: 
        # Create a boolean array where count >= 2
        # Find the first False after index 0?
        # Actually, we just need to scan. 
        # For speed, we can use np.where
        uncovered = np.where(counts[1:M+1] < 2)[0]
        if len(uncovered) > 0:
            return uncovered[0] # Since indices in np.where are relative to slice 1:M+1?
            # Wait. counts[1:M+1] has indices 0..M-1 corresponding to values 1..M.
            # So if uncovered[0] is k, the value is k+1.
            # So L = (k+1) - 1 = k.
            # Let's verify.
            # If counts[1] < 2, uncovered[0]=0. L=0. Correct.
            # If counts[1]>=2, counts[2]<2, uncovered[0]=1. L=1. Correct.
            return int(uncovered[0])
        return M

    best_L = get_L(counts, M)
    best_residues = residues.copy()
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.999
    
    # Track stagnation
    last_improvement_time = start_time
    stagnation_limit = 5.0
    
    while time.time() < deadline:
        # Check for time
        if time.time() > deadline - 5:
            break
            
        # Pick a random prime to change
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        
        # Pick a new residue
        # Bias towards changing to 0 or small numbers? 
        # Or just uniform.
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply change to counts
        # Remove old
        if old_r == 0:
            counts[p:M+1:p] -= 1
        else:
            counts[old_r:M+1:p] -= 1
            
        # Add new
        if new_r == 0:
            counts[p:M+1:p] += 1
        else:
            counts[new_r:M+1:p] += 1
            
        residues[idx] = new_r
        
        new_L = get_L(counts, M)
        
        # Acceptance criterion
        # If new_L > best_L, accept and update best.
        # If new_L == best_L, accept with some probability or if it helps future expansion?
        # Actually, we want to maximize L.
        # If new_L < best_L, it's worse.
        # But we are doing SA on the "state", not just L.
        # The objective is L.
        # If we accept worse L, we might escape local minima.
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues.copy()
            last_improvement_time = time.time()
            
            # Expand M if we hit the boundary
            if best_L == M:
                # Increase M
                old_M = M
                M = min(M * 2, max_M)
                # Extend counts array
                new_counts = np.zeros(M + 1, dtype=np.int32)
                # Copy old counts
                new_counts[:old_M+1] = counts
                # We need to recompute counts for the new range?
                # No, the existing counts are correct for 1..old_M.
                # But we need counts for old_M+1 .. M.
                # It is easier to recompute all counts from scratch when M changes.
                # Since M doubles, this is O(M log log M).
                counts = np.zeros(M + 1, dtype=np.int32)
                for i, p_prime in enumerate(primes):
                    r = residues[i]
                    if r == 0:
                        counts[p_prime:M+1:p_prime] += 1
                    else:
                        counts[r:M+1:p_prime] += 1
                # Re-evaluate L with new M
                best_L = get_L(counts, M)
                # It might have increased further
                if best_L > best_L: # This logic is redundant, just update best_L
                    pass
                # Update best_residues if it improved? 
                # The residues are the same, so best_residues is still valid.
                # But best_L might be higher now.
                # We should update best_L to the new calculated value.
                # And if it's better than previous best, it is.
                
        else:
            # Calculate delta
            delta = new_L - best_L
            # If delta < 0, accept with probability exp(delta / T)
            # Note: delta is negative.
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    # Accept worse move
                    pass
                else:
                    # Revert
                    residues[idx] = old_r
                    if new_r == 0:
                        counts[p:M+1:p] -= 1
                    else:
                        counts[new_r:M+1:p] -= 1
                    if old_r == 0:
                        counts[p:M+1:p] += 1
                    else:
                        counts[old_r:M+1:p] += 1
            else:
                # delta == 0, accept
                pass
        
        # Cool down
        T *= cooling
        
        # Stagnation check
        if time.time() - last_improvement_time > stagnation_limit:
            # Restart from best
            residues = best_residues.copy()
            # Recompute counts for current M
            counts = np.zeros(M + 1, dtype=np.int32)
            for i, p_prime in enumerate(primes):
                r = residues[i]
                if r == 0:
                    counts[p_prime:M+1:p_prime] += 1
                else:
                    counts[r:M+1:p_prime] += 1
            best_L = get_L(counts, M)
            
            # Increase temperature to explore more
            T = 50.0
            last_improvement_time = time.time()
            
            # Optional: Perturb best solution
            for _ in range(5):
                j = random.randint(0, n_primes - 1)
                p_j = primes[j]
                old_r = residues[j]
                new_r = random.randint(0, p_j - 1)
                if new_r == old_r: continue
                residues[j] = new_r
                if old_r == 0:
                    counts[p_j:M+1:p_j] -= 1
                else:
                    counts[old_r:M+1:p_j] -= 1
                if new_r == 0:
                    counts[p_j:M+1:p_j] += 1
                else:
                    counts[new_r:M+1:p_j] += 1
                    
    return best_residues
# EVOLVE-BLOCK-END
