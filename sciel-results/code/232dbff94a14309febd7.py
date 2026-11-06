import time
import random
import math
import numpy as np

N = 51975

def get_divisors(n):
    divs = []
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

MODULI = get_divisors(N)
K = len(MODULI)

def calculate_score(counts):
    return int(np.sum(counts == 0))

def update_counts(residues):
    counts = np.zeros(N, dtype=np.int32)
    for m, r in zip(MODULI, residues):
        counts[r::m] += 1
    return counts

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initial Solution: Greedy Ascending
    # Start with smallest moduli to cover more ground quickly
    # Or just random init and hill climb?
    # Let's try a few random inits and pick the best after a quick hill climb.
    
    best_residues = None
    best_score = N + 1
    best_counts = None
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Random initialization
        residues = [random.randint(0, m - 1) for m in MODULI]
        counts = update_counts(residues)
        current_score = calculate_score(counts)
        
        # Hill Climbing
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Shuffle moduli to avoid bias
            indices = list(range(K))
            random.shuffle(indices)
            
            for idx in indices:
                m = MODULI[idx]
                r_old = residues[idx]
                
                best_r = r_old
                best_delta = 0  # We want to minimize delta, so 0 is no change
                
                # Evaluate all possible new residues
                # Optimization: If m is large, N/m is small, so iteration is fast.
                # If m is small, N/m is large, but m is small.
                
                # We can vectorize the gain calculation for all r' at once?
                # gain[r'] = sum(counts[r'::m] == 1) - sum(counts[r'::m] == 0) ... wait.
                # Delta for moving r_old -> r_new:
                # Remove r_old: +count of 1s in r_old slice
                # Add r_new: -count of 0s in r_new slice (before removal? No, sets are disjoint)
                
                # Let's compute delta for each r'
                # delta(r') = np.sum(counts[r_old::m] == 1) - np.sum(counts[r'::m] == 0)
                # Note: counts[r_old::m] == 1 is constant for all r'.
                # Let base_gain = np.sum(counts[r_old::m] == 1)
                # Then delta(r') = base_gain - np.sum(counts[r'::m] == 0)
                
                base_gain = np.sum(counts[r_old::m] == 1)
                
                # We need to find r' that minimizes delta, i.e., maximizes np.sum(counts[r'::m] == 0)
                # subject to r' != r_old
                
                best_r_prime = r_old
                max_zeros = -1
                
                # Iterate r'
                # For small m, this loop is small. For large m, the slice is small.
                for r_prime in range(m):
                    if r_prime == r_old:
                        continue
                    zeros = np.sum(counts[r_prime::m] == 0)
                    if zeros > max_zeros:
                        max_zeros = zeros
                        best_r_prime = r_prime
                
                if best_r_prime != r_old:
                    delta = base_gain - max_zeros
                    if delta < 0:
                        # Apply move
                        counts[r_old::m] -= 1
                        counts[best_r_prime::m] += 1
                        residues[idx] = best_r_prime
                        current_score += delta
                        improved = True
                        # If score improves, we might want to restart the while loop
                        # or just continue. Continuing is fine for Hill Climbing.
                        if current_score < best_score:
                            best_score = current_score
                            best_residues = residues[:]
                            best_counts = counts.copy() # Copy for safety? No, we update in place.
                            # Actually, we don't need to copy best_counts if we just track best_residues and recalc if needed.
                            # But we need best_counts for perturbation? No, we recalc counts from residues.
        
        # After hill climbing, if this is the best, save it
        if current_score < best_score:
            best_score = current_score
            best_residues = residues[:]
        
        # Perturbation for next restart (if any)
        # Actually, the next restart is independent random init.
        # We can also do ILS: perturb the current best.
        
        # Let's do a simple ILS strategy:
        # If we have time, perturb the current best and hill climb again.
        
    # Final Polish: Run a more exhaustive local search on the best solution found
    if best_residues is not None and time.time() - start_time < time_limit - 2:
        residues = best_residues[:]
        counts = update_counts(residues)
        current_score = calculate_score(counts)
        
        # Repeat hill climbing a few times with different orders
        for _ in range(5):
            if time.time() - start_time > time_limit - 1:
                break
            improved = True
            while improved and (time.time() - start_time < time_limit - 1):
                improved = False
                indices = list(range(K))
                random.shuffle(indices)
                for idx in indices:
                    m = MODULI[idx]
                    r_old = residues[idx]
                    base_gain = np.sum(counts[r_old::m] == 1)
                    best_r_prime = r_old
                    max_zeros = -1
                    for r_prime in range(m):
                        if r_prime == r_old:
                            continue
                        zeros = np.sum(counts[r_prime::m] == 0)
                        if zeros > max_zeros:
                            max_zeros = zeros
                            best_r_prime = r_prime
                    if best_r_prime != r_old:
                        delta = base_gain - max_zeros
                        if delta < 0:
                            counts[r_old::m] -= 1
                            counts[best_r_prime::m] += 1
                            residues[idx] = best_r_prime
                            current_score += delta
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_residues = residues[:]
    
    classes = [[m, r] for m, r in zip(MODULI, best_residues)]
    return classes
