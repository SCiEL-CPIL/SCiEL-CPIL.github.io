"""
Strategy: Multi-phase construction with improved conflict-aware greedy and smooth local search.
Phase 1: Conflict-aware greedy gap-closure with marginal damage scoring.
Phase 2: Smooth surrogate local search (soft-min objective) to escape local minima.
Phase 3: High-precision final polish with exact local optimization.
"""
import time
import numpy as np
from math import log, exp

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We'll track coverage up to a reasonable limit
    max_check = 5000
    
    # residues[i] is the residue for primes[i]
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        r = new_r if delta == 1 else old_r
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            counts[start::p] += delta

    def get_prefix_score():
        """
        Find the largest L such that counts[1..L] >= 2.
        """
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    # Initialize: set all residues to 0 and compute counts
    for i in range(n_primes):
        update_counts(i, 0, 0, 1)
        
    best_score = get_prefix_score()
    best_residues = residues.copy()

    def marginal_damage(i, old_r, new_r, current_score):
        """
        Estimate the marginal damage of changing prime i from old_r to new_r.
        Returns a score where lower is better.
        """
        p = primes[i]
        
        # Damage from removing old coverage
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
            
        old_damage = 0
        if old_start <= current_score:
            old_positions = np.arange(old_start, current_score + 1, p)
            # Count how many positions would drop below 2
            old_damage = np.sum(counts[old_positions] == 2)
        
        # Gain from adding new coverage
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
            
        new_gain = 0
        if new_start <= current_score:
            new_positions = np.arange(new_start, current_score + 1, p)
            # Count how many positions would go from 1 to 2 (or 0 to 1, etc.)
            new_gain = np.sum((counts[new_positions] == 1))
        
        # Also consider positions beyond current_score that might be affected
        # (these are potential future gains)
        future_horizon = min(current_score + 200, max_check)
        
        if new_r == 0:
            new_start_f = p
        else:
            new_start_f = new_r
            
        if new_start_f <= future_horizon and new_start_f > current_score:
            future_positions = np.arange(new_start_f, future_horizon + 1, p)
            future_gain = np.sum((counts[future_positions] == 1))
            new_gain += future_gain * 0.5  # Weight future gains less
        
        if old_r == 0:
            old_start_f = p
        else:
            old_start_f = old_r
            
        if old_start_f <= future_horizon and old_start_f > current_score:
            future_old_positions = np.arange(old_start_f, future_horizon + 1, p)
            future_loss = np.sum((counts[future_old_positions] == 2))
            old_damage += future_loss * 0.5
        
        return old_damage - new_gain

    # --- Phase 1: Conflict-Aware Greedy Construction ---
    # Iteratively fix the first gap (count < 2) with damage-aware selection
    
    max_repair_steps = 10000
    step = 0
    
    while time.time() < deadline - 40.0 and step < max_repair_steps:
        step += 1
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        
        # Find the best prime to cover x_target using marginal damage scoring
        best_i = -1
        best_damage = float('inf')
        best_new_r = 0
        
        for i in range(n_primes - 1, -1, -1):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
            
            # Check if removing coverage from old_r breaks any prefix point
            if old_r == 0:
                start_pos = p
            else:
                start_pos = old_r
                
            if start_pos <= current_score:
                positions = np.arange(start_pos, current_score + 1, p)
                if np.any(counts[positions] == 2):
                    continue  # This move would break the prefix
            
            # Calculate marginal damage
            damage = marginal_damage(i, old_r, new_r, current_score)
            
            if damage < best_damage:
                best_damage = damage
                best_i = i
                best_new_r = new_r
        
        if best_i >= 0:
            i = best_i
            p = primes[i]
            old_r = residues[i]
            new_r = best_new_r
            update_counts(i, old_r, new_r, -1)
            residues[i] = new_r
            update_counts(i, new_r, new_r, 1)
        else:
            # Stuck! Try a random kick to escape local minimum
            kicked = False
            for _ in range(30):
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r == old_r:
                    continue
                    
                # Check if this move breaks the prefix
                if old_r == 0:
                    start_pos = p
                else:
                    start_pos = old_r
                    
                valid = True
                if start_pos <= current_score:
                    positions = np.arange(start_pos, current_score + 1, p)
                    if np.any(counts[positions] == 2):
                        valid = False
                
                if valid:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                    kicked = True
                    break
            
            if not kicked:
                # Try a more aggressive kick: change a prime and accept temporary damage
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r != old_r:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                else:
                    break

    # --- Phase 2: Smooth Surrogate Local Search ---
    # Use a soft-min objective to guide moves toward improving coverage
    # The surrogate objective is: sum of soft-min(counts[x], 2) for x in prefix
    
    def soft_min_surrogate(counts_arr, beta=2.0):
        """
        Compute a smooth surrogate for the number of positions with count < 2.
        Uses soft-min: for each position, if count < 2, contribute (2 - count) softened.
        """
        # For positions with count < 2, we want to minimize the deficit
        # Use a smooth approximation: deficit = 2 - count, softened with exp
        below = counts_arr < 2
        if not np.any(below):
            return 0.0
        deficits = np.maximum(2 - counts_arr, 0)
        # Soft sum: log-sum-exp of exp(deficit * beta)
        max_deficit = np.max(deficits)
        if max_deficit == 0:
            return 0.0
        log_sum = np.log(np.sum(np.exp((deficits - max_deficit) * beta))) + max_deficit * beta
        return log_sum / beta
    
    temperature = 100.0
    cooling = 0.998
    min_temp = 0.5
    surrogate_beta = 2.0
    
    # Track current surrogate value
    current_surrogate = soft_min_surrogate(counts[1:max_check+1], surrogate_beta)
    
    while time.time() < deadline - 15.0 and temperature > min_temp:
        # Pick a random prime
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
        
        # Make the move
        update_counts(i, old_r, new_r, -1)
        residues[i] = new_r
        update_counts(i, new_r, new_r, 1)
        
        new_score = get_prefix_score()
        new_surrogate = soft_min_surrogate(counts[1:max_check+1], surrogate_beta)
        
        # Accept if better score, or with probability based on surrogate improvement
        if new_score > best_score:
            best_score = new_score
            best_residues = residues.copy()
            current_surrogate = new_surrogate
        elif new_score == best_score and new_surrogate < current_surrogate:
            # Same score but better surrogate: accept
            current_surrogate = new_surrogate
        else:
            # Compute acceptance probability based on surrogate change
            surrogate_diff = current_surrogate - new_surrogate  # positive if improvement
            if surrogate_diff > 0:
                accept_prob = 1.0
            else:
                accept_prob = exp(surrogate_diff / temperature)
            
            if np.random.random() < accept_prob:
                current_surrogate = new_surrogate
            else:
                # Revert
                update_counts(i, new_r, old_r, -1)
                residues[i] = old_r
                update_counts(i, old_r, old_r, 1)
        
        temperature *= cooling

    # --- Phase 3: High-Precision Final Polish ---
    # Exact local optimization: try all single-prime changes and accept any improvement
    polish_iters = 5000
    for it in range(polish_iters):
        if time.time() > deadline - 3.0:
            break
            
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        # Try to find any improving move
        improved = False
        
        # First, try to fix the first gap
        if current_score < max_check:
            x_target = current_score + 1
            for i in range(n_primes - 1, -1, -1):
                p = primes[i]
                old_r = residues[i]
                new_r = x_target % p
                
                if old_r == new_r:
                    continue
                
                if old_r == 0:
                    start_pos = p
                else:
                    start_pos = old_r
                    
                if start_pos > current_score:
                    valid = True
                else:
                    positions = np.arange(start_pos, current_score + 1, p)
                    if np.any(counts[positions] == 2):
                        valid = False
                    else:
                        valid = True
                
                if valid:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                    improved = True
                    break
        
        if not improved:
            # Try random improvements: change a prime and see if score improves
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)
            
            if new_r == old_r:
                continue
            
            # Make the move temporarily
            update_counts(i, old_r, new_r, -1)
            residues[i] = new_r
            update_counts(i, new_r, new_r, 1)
            
            new_score = get_prefix_score()
            
            if new_score > current_score:
                improved = True
            else:
                # Revert
                update_counts(i, new_r, old_r, -1)
                residues[i] = old_r
                update_counts(i, old_r, old_r, 1)
        
        if not improved and it > 100:
            # No improvements found in several iterations; break
            if it > 200:
                break

    # Final check
    final_score = get_prefix_score()
    if final_score > best_score:
        best_score = final_score
        best_residues = residues.copy()

    # Ensure we return the best found
    return [int(r) for r in best_residues]
