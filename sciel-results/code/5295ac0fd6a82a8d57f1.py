import time
import random
import math

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def get_prime_factors_filtered(n):
    """Get prime factors > 5 only"""
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        if p > 5:
            factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_new_primes(k_vals):
    """
    Counts the number of distinct prime factors of (1 + k_i + k_j) 
    for i < j, excluding 2, 3, 5.
    """
    new_primes = set()
    n = len(k_vals)
    for i in range(n):
        for j in range(i + 1, n):
            s = 1 + k_vals[i] + k_vals[j]
            p_factors = get_prime_factors_filtered(s)
            new_primes.update(p_factors)
    return len(new_primes)

def incremental_cost(k_vals, idx, old_val, new_val):
    """
    Compute cost change when replacing k_vals[idx] from old_val to new_val.
    Returns (new_total_cost, removed_primes, added_primes)
    """
    n = len(k_vals)
    # Collect all prime factors from pairs involving idx
    old_primes = set()
    new_primes = set()
    
    for j in range(n):
        if j == idx:
            continue
        # Old sum
        s_old = 1 + old_val + k_vals[j]
        old_primes.update(get_prime_factors_filtered(s_old))
        # New sum
        s_new = 1 + new_val + k_vals[j]
        new_primes.update(get_prime_factors_filtered(s_new))
    
    # Primes that are only in old (removed)
    removed = old_primes - new_primes
    # Primes that are only in new (added)
    added = new_primes - old_primes
    
    return removed, added

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 48.0  # Safety margin within 90s limit

    # We want 12 distinct k_i >= 0
    # a_i = 15 + 30 * k_i
    # Max a_i = 10000 => 15 + 30*k_i <= 10000 => 30*k_i <= 9985 => k_i <= 332
    
    best_k = None
    best_cost = float('inf')
    
    # Try multiple random initial solutions
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
            
        # Generate random initial solution with bias toward small values
        k_vals = set()
        while len(k_vals) < 12:
            if random.random() < 0.6:
                k_vals.add(random.randint(0, 30))
            else:
                k_vals.add(random.randint(0, 100))
        k_vals = list(k_vals)
        
        current_cost = count_new_primes(k_vals)
        if current_cost < best_cost:
            best_cost = current_cost
            best_k = k_vals[:]
    
    if best_k is None:
        best_k = list(range(12))
        best_cost = count_new_primes(best_k)
    
    # Use the best initial solution for SA
    k_vals = best_k[:]
    current_cost = best_cost
    
    T = 3.0
    cooling_rate = 0.998
    
    # Track recent best to detect stagnation
    recent_best = current_cost
    stagnation_counter = 0
    max_stagnation = 2000
    
    # Local Search / Simulated Annealing
    while time.time() < deadline:
        # Check for stagnation
        if current_cost < recent_best:
            recent_best = current_cost
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if stagnation_counter > max_stagnation:
            # Restart with perturbation
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            for _ in range(3):
                idx = random.randint(0, 11)
                old_val = k_vals[idx]
                new_val = random.randint(0, 80)
                if new_val not in k_vals or new_val == old_val:
                    k_vals[idx] = new_val
                    break
            current_cost = count_new_primes(k_vals)
            recent_best = current_cost
            stagnation_counter = 0
            T = 2.0
        
        # Pick a random index to modify
        idx = random.randint(0, 11)
        old_val = k_vals[idx]
        
        # Generate a new value with bias toward small values
        r = random.random()
        if r < 0.15:
            # Large jump
            new_val = random.randint(0, 120)
        elif r < 0.4:
            # Medium jump
            new_val = max(0, min(332, old_val + random.randint(-15, 15)))
        else:
            # Small step
            new_val = max(0, min(332, old_val + random.randint(-3, 3)))
        
        # Check distinctness
        if new_val in k_vals and new_val != old_val:
            continue

        # Apply move
        k_vals[idx] = new_val
        
        # Evaluate using incremental cost for speed
        removed, added = incremental_cost(k_vals, idx, old_val, new_val)
        
        # Compute net change: primes in added that aren't in removed, minus primes in removed that aren't in added
        # But we need the actual count, so let's just do full evaluation for accuracy
        # Actually, let's use the incremental approach properly:
        # We need to know which primes are unique to the pairs involving idx
        
        # For simplicity and correctness, do full evaluation
        new_cost = count_new_primes(k_vals)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_k = k_vals[:]
        else:
            # Simulated Annealing
            delta = new_cost - current_cost
            if delta == 0:
                # Accept moves with same cost to explore
                if random.random() < 0.6:
                    current_cost = new_cost
                else:
                    k_vals[idx] = old_val
            else:
                # Use a more aggressive acceptance for small deltas
                if delta <= 1 and random.random() < 0.3:
                    current_cost = new_cost
                elif random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Reject move
                    k_vals[idx] = old_val
        
        T *= cooling_rate
        if T < 0.05:
            T = 1.5  # Warm restart

    # Final polish: try local moves around the best solution
    if time.time() < deadline - 2:
        polish_k = best_k[:]
        polish_cost = best_cost
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(12):
                old_val = polish_k[idx]
                # Try nearby values
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val < 0 or new_val > 332:
                        continue
                    if new_val in polish_k and new_val != old_val:
                        continue
                    polish_k[idx] = new_val
                    new_cost = count_new_primes(polish_k)
                    if new_cost < polish_cost:
                        polish_cost = new_cost
                        improved = True
                        break
                    else:
                        polish_k[idx] = old_val
                if time.time() > deadline - 1:
                    break
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_k = polish_k[:]

    # Construct final array
    result = [15 + 30 * k for k in best_k]
    
    # Verify distinctness and bounds
    # k are distinct, so result are distinct.
    # Max k is 332, max result 15 + 30*332 = 9975 <= 10000.
    # Min result 15 >= 1.
    
    return result
# EVOLVE-BLOCK-END
