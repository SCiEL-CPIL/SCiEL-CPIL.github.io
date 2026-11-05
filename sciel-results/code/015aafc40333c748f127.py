import time
import numpy as np
import random

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _find_L_from_counts(counts, max_L):
    """
    Find largest L such that counts[1..L] >= 2.
    counts is a numpy array of length max_L+1.
    """
    # We care about indices 1 to max_L
    # np.where returns indices where condition is true
    # We want the first index where counts < 2
    # If none, return max_L
    # Note: counts[0] is unused/irrelevant for the score definition 1..L
    below_two = np.where(counts[1:max_L+1] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        # below_two[0] is the 0-based index in the slice, so add 1
        return below_two[0] + 1

def construct():
    start_time = time.time()
    # 150s limit, keep 10s safety margin
    deadline = start_time + 140
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Precompute positions for each prime and each residue?
    # Memory: sum(p_i * (max_L/p_i)) = 168 * max_L = 16.8M ints. 
    # 16.8M * 4 bytes = ~67 MB. This is acceptable.
    # positions[i][r] = array of indices in 1..max_L that are r mod p_i
    # This allows O(1) access to the delta vectors.
    
    # However, storing 168 * 1000 arrays might have overhead.
    # Let's store them in a list of lists.
    # positions[i] is a list of length primes[i], where positions[i][r] is the np array.
    
    positions = []
    for p in primes:
        pos_list = []
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                pos_list.append(np.array([], dtype=np.int32))
            else:
                pos_list.append(np.arange(start, max_L + 1, p, dtype=np.int32))
        positions.append(pos_list)
        
    # Initial solution: Random or simple heuristic
    # Let's start with a random solution to avoid bias, or the seed's initial.
    # Seed initial: ((i + 2) // 2) % p
    # Let's try a few random starts and pick the best, or just start with one and anneal.
    # Given the time, a single good anneal run with many restarts is better.
    
    best_residues_overall = None
    best_L_overall = -1
    
    # Number of restarts
    # Each restart will run SA for a certain amount of time or iterations.
    # Let's do multiple short SA runs.
    
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize residues
        # Heuristic: start with the seed's initial or random
        # Let's use random for diversity
        residues = np.random.randint(0, primes, size=n_primes)
        
        # Compute initial counts
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            pos = positions[i][r]
            if len(pos) > 0:
                counts[pos] += 1
                
        current_L = _find_L_from_counts(counts, max_L)
        
        if current_L > best_L_overall:
            best_L_overall = current_L
            best_residues_overall = residues.copy()
            
        # Simulated Annealing
        # Parameters
        # We want to maximize L.
        # Objective function is step-like, so we need to look at the "depth" of the minimum.
        # A better objective for SA: Minimize the number of "uncovered" (count < 2) points, 
        # or minimize the sum of (2 - count) for count < 2.
        # Let's use the number of points with count < 2 as the "energy" to minimize.
        # Actually, L is the first point with count < 2. 
        # If we minimize the count of points with count < 2, we might not increase L if the first bad point remains.
        # A better metric for SA acceptance: 
        # Accept if new_L > old_L.
        # Accept if new_L == old_L and (some random condition or slight improvement in secondary metric).
        # Accept if new_L < old_L with probability exp(-delta / T).
        
        # To make SA effective, we should define a smooth objective.
        # Let's define Energy = sum_{x=1}^{max_L} max(0, 2 - counts[x])
        # This penalizes all under-covered points. Minimizing this tends to push the first under-covered point further out.
        
        # Calculate initial energy
        energy = np.sum(np.maximum(0, 2 - counts[1:max_L+1]))
        
        T = 100.0 # Initial temperature
        T_min = 0.01
        cooling_rate = 0.999 # Adjust based on time
        
        # We don't know how many steps we can take. 
        # Let's run until time runs out for this restart.
        restart_deadline = start_time + (restart + 1) * time_per_restart
        
        while time.time() < restart_deadline:
            # Pick a random prime
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            
            # Pick a new random residue
            new_r = np.random.randint(p)
            if new_r == old_r:
                continue
                
            # Calculate delta
            old_pos = positions[i][old_r]
            new_pos = positions[i][new_r]
            
            # Update counts temporarily? 
            # We need new_L and new_Energy.
            # counts_new = counts - 1 at old_pos + 1 at new_pos
            
            # To avoid copying the whole array, we can compute the change in L and Energy locally?
            # L change: 
            # The positions affected are old_pos and new_pos.
            # The set of affected indices is A = union(old_pos, new_pos).
            # We can check the minimum index in A where the count drops below 2?
            # This is tricky. 
            # Given max_L=100000, copying the array (400KB) is fast.
            # Let's copy.
            
            counts_new = counts.copy()
            if len(old_pos) > 0:
                counts_new[old_pos] -= 1
            if len(new_pos) > 0:
                counts_new[new_pos] += 1
            
            new_L = _find_L_from_counts(counts_new, max_L)
            new_energy = np.sum(np.maximum(0, 2 - counts_new[1:max_L+1]))
            
            delta_E = new_energy - energy
            delta_L = new_L - current_L
            
            # Acceptance criteria
            accept = False
            if delta_L > 0:
                accept = True
            elif delta_L == 0:
                # Accept if energy decreases or with small probability
                if new_energy < energy:
                    accept = True
                else:
                    # Random walk to escape plateaus
                    if random.random() < 0.1:
                        accept = True
            else:
                # delta_L < 0
                # Accept with probability exp(-delta_E / T)
                # If delta_E is positive (worse energy), probability is low.
                # If delta_E is negative (better energy, but L dropped?), that's unlikely if L is the first bad spot.
                # Usually dropping L means the first bad spot moved earlier, so energy likely increased.
                if delta_E < 0:
                    accept = True
                else:
                    if T > 0:
                        prob = np.exp(-delta_E / T)
                        if random.random() < prob:
                            accept = True
            
            if accept:
                counts = counts_new
                residues[i] = new_r
                current_L = new_L
                energy = new_energy
                
                if current_L > best_L_overall:
                    best_L_overall = current_L
                    best_residues_overall = residues.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Optional: If we get stuck for a long time, reheat?
            # For simplicity, just let it cool.
            
    # Final local search (Hill Climbing) on the best solution found
    if best_residues_overall is not None:
        residues = best_residues_overall.copy()
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            pos = positions[i][r]
            if len(pos) > 0:
                counts[pos] += 1
        current_L = _find_L_from_counts(counts, max_L)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n_primes):
                if time.time() > deadline - 2:
                    break
                p = primes[i]
                old_r = residues[i]
                best_new_r = old_r
                best_new_L = current_L
                
                # Try all residues for this prime
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    old_pos = positions[i][old_r]
                    new_pos = positions[i][new_r]
                    
                    counts_new = counts.copy()
                    if len(old_pos) > 0:
                        counts_new[old_pos] -= 1
                    if len(new_pos) > 0:
                        counts_new[new_pos] += 1
                    
                    new_L = _find_L_from_counts(counts_new, max_L)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_r = new_r
                
                if best_new_r != old_r:
                    # Apply change
                    old_pos = positions[i][old_r]
                    new_pos = positions[i][best_new_r]
                    if len(old_pos) > 0:
                        counts[old_pos] -= 1
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                    residues[i] = best_new_r
                    current_L = best_new_L
                    improved = True
                    if current_L > best_L_overall:
                        best_L_overall = current_L
                        best_residues_overall = residues.copy()

    return list(best_residues_overall)

