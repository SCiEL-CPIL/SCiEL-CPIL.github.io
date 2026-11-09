import time
import random
import numpy as np
from fractions import Fraction
import sys

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes_list = _get_primes(1000)
    L_cap = 100000 # Cap for evaluation
    
    # Pre-compute prime indices and reciprocal values for fast budget checks
    # We use float for speed during search, but verify with Fraction at the end.
    # The budget constraint is sum(1/p) <= 1.
    
    # Strategy:
    # 1. Initialize with a set of primes that fits the budget.
    #    Primes >= 13 sum to approx 0.92. This is a safe starting point.
    #    We can also try including smaller primes if we drop others, but 
    #    smaller primes cover more dense numbers, so they are valuable for L.
    #    Let's start with primes 11..997? Sum(1/p for p in 11..997) ~ 0.95.
    #    Let's check sum for primes >= 11.
    
    candidate_primes = [p for p in primes_list if p >= 11]
    initial_sum = sum(Fraction(1, p) for p in candidate_primes)
    
    # If initial sum is too high, we might need to drop the largest primes (smallest reciprocals) 
    # or start with a higher cutoff.
    # Actually, for maximizing L, we want small primes (dense coverage) but they cost more budget.
    # The "Erdos-Ruzsa" construction typically uses a specific set.
    # Let's use a greedy initialization:
    # Start with empty set, add primes in increasing order of 1/p (i.e., decreasing p) 
    # until we can't add more? No, that gives large primes.
    # We want to cover 1..L. Small primes are crucial for covering small numbers densely.
    # However, 1/2 + 1/3 + 1/5 + ... quickly exceeds 1.
    # The known best coverings for C=1 use primes roughly in the range [11, 1000].
    
    # Let's use the set of primes p where sum(1/p) <= 1.
    # A good heuristic: use primes p >= 13.
    # Sum(1/p for p in 13..997) is approx 0.92.
    # We have ~0.08 budget left. We can add a few smaller primes?
    # If we add 11, cost is 1/11 ~ 0.09. Total ~ 1.01 > 1.
    # If we add 7, cost 1/7 ~ 0.14. Too much.
    # So primes >= 13 is a tight, valid set.
    
    # Let's try to optimize the set of primes itself as part of the search?
    # Changing a prime p to q changes the cost by 1/q - 1/p and the coverage pattern.
    # This is a hard combinatorial problem.
    # Let's stick to a fixed set of primes for the residue optimization, 
    # but allow swapping primes in the SA if it helps L and fits budget.
    
    # Initial Set: Primes 13 to 997.
    active_primes = [p for p in primes_list if 13 <= p <= 997]
    
    # Check budget
    current_sum_frac = sum(Fraction(1, p) for p in active_primes)
    if current_sum_frac > 1:
        # Fallback: remove largest primes until sum <= 1
        active_primes.sort(reverse=True)
        while current_sum_frac > 1 and active_primes:
            p = active_primes.pop()
            current_sum_frac -= Fraction(1, p)
    
    # For speed, we will track the sum as a float during search, 
    # but we must ensure the final result is exact.
    # Since we only swap primes or change residues (which don't affect sum),
    # the sum only changes when we swap primes.
    # We will maintain `current_sum_float` and `current_sum_frac` is only updated on swap.
    
    current_sum_float = sum(1.0 / p for p in active_primes)
    
    # --- NumPy Coverage Engine ---
    # We maintain a boolean array `covered` of size L_cap + 1.
    # To update efficiently, we can't just do `covered[start::p] = True` because 
    # we need to handle removals (when changing a residue, the old positions become uncovered 
    # unless covered by another class).
    # So we need a count array `counts` of size L_cap + 1 (int8 or int16 is enough).
    
    # Re-initializing the count array from scratch on every move is O(K * L/p).
    # K ~ 100, L/p ~ 1000. 100k ops. In NumPy, this is very fast.
    # However, doing it 100,000 times might be slow if Python overhead dominates.
    # Let's try an incremental approach with NumPy.
    # `counts[start_old::p_old] -= 1`
    # `counts[start_new::p_new] += 1`
    # This is O(L/p_old + L/p_new). Average p ~ 500. L/p ~ 200.
    # 200 operations in NumPy is extremely fast.
    
    counts = np.zeros(L_cap + 1, dtype=np.int8)
    
    def update_counts(primes, residues):
        """Rebuild counts from scratch. Used for init and restarts."""
        nonlocal counts
        counts.fill(0)
        for p, r in zip(primes, residues):
            start = r if r >= 1 else r + p
            if start <= L_cap:
                counts[start::p] += 1
    
    def get_L(counts):
        """Find the first index where count is 0."""
        # Find first zero in counts[1:]
        # np.nonzero(counts[1:] == 0)[0]
        # If empty, L is L_cap.
        zeros = np.nonzero(counts[1:] == 0)[0]
        if len(zeros) == 0:
            return L_cap
        return zeros[0] # This is the 0-based index in the slice, which corresponds to 1-based index in original?
        # Slice is counts[1:]. Index 0 in slice is counts[1].
        # So if zeros[0] is 0, counts[1] is 0, so L=0.
        # If zeros[0] is 5, counts[6] is 0, so L=5.
        # So L = zeros[0].
        
    def apply_residue_change(idx, new_r):
        """Change residue of prime at idx to new_r. Returns new L."""
        nonlocal counts
        p = active_primes[idx]
        old_r = residues[idx]
        
        if old_r == new_r:
            return get_L(counts)
        
        # Remove old
        start_old = old_r if old_r >= 1 else old_r + p
        if start_old <= L_cap:
            counts[start_old::p] -= 1
            
        # Add new
        start_new = new_r if new_r >= 1 else new_r + p
        if start_new <= L_cap:
            counts[start_new::p] += 1
            
        residues[idx] = new_r
        return get_L(counts)

    def apply_prime_swap(idx, new_p, new_r):
        """Swap prime at idx to new_p with residue new_r. Returns new L and sum change."""
        nonlocal counts, active_primes, current_sum_float
        old_p = active_primes[idx]
        old_r = residues[idx]
        
        if old_p == new_p:
            return get_L(counts), 0.0
            
        # Remove old
        start_old = old_r if old_r >= 1 else old_r + old_p
        if start_old <= L_cap:
            counts[start_old::old_p] -= 1
            
        # Add new
        start_new = new_r if new_r >= 1 else new_r + new_p
        if start_new <= L_cap:
            counts[start_new::new_p] += 1
            
        # Update state
        active_primes[idx] = new_p
        residues[idx] = new_r
        
        sum_change = (1.0 / new_p) - (1.0 / old_p)
        current_sum_float += sum_change
        
        return get_L(counts), sum_change

    # --- Multi-Restart Simulated Annealing ---
    
    best_L_global = 0
    best_primes_global = []
    best_residues_global = []
    best_sum_frac_global = Fraction(0)
    
    # We will run several restarts.
    # Each restart initializes residues randomly or greedily, then runs SA.
    
    num_restarts = 50 # We have time for many restarts if each is short
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize residues
        # Option 1: Random
        # Option 2: Greedy (cover 1, then 2, etc.)
        # Let's use a mix.
        
        active_primes = [p for p in primes_list if 13 <= p <= 997]
        current_sum_float = sum(1.0 / p for p in active_primes)
        current_sum_frac = sum(Fraction(1, p) for p in active_primes)
        
        if current_sum_frac > 1:
             # This shouldn't happen with 13..997
             pass
             
        residues = [random.randint(0, p-1) for p in active_primes]
        
        update_counts(active_primes, residues)
        current_L = get_L(counts)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_primes_global = active_primes[:]
            best_residues_global = residues[:]
            best_sum_frac_global = current_sum_frac
            
        # SA Parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        max_steps = 20000 # Steps per restart
        
        for step in range(max_steps):
            if time.time() > deadline - 5: # Leave 5s for final check
                break
                
            # Pick a move
            r_move = random.random()
            
            if r_move < 0.8:
                # Change residue
                idx = random.randint(0, len(active_primes) - 1)
                p = active_primes[idx]
                old_r = residues[idx]
                
                # Smart residue selection:
                # Target the first gap
                gap = current_L + 1
                if gap <= L_cap:
                    r_candidate = gap % p
                    # With some probability, use the smart one, otherwise random
                    if random.random() < 0.5:
                        new_r = r_candidate
                    else:
                        new_r = random.randint(0, p - 1)
                else:
                    new_r = random.randint(0, p - 1)
                    
                new_L = apply_residue_change(idx, new_r)
                delta_L = new_L - current_L
                
                if delta_L >= 0:
                    current_L = new_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_primes_global = active_primes[:]
                        best_residues_global = residues[:]
                        best_sum_frac_global = current_sum_frac
                else:
                    # Accept with prob exp(delta_L / T)
                    # delta_L is negative
                    accept_prob = np.exp(delta_L / T)
                    if random.random() < accept_prob:
                        current_L = new_L
                    else:
                        # Revert
                        apply_residue_change(idx, old_r)
                        
            else:
                # Swap prime
                # Only if we have budget headroom or swapping to a larger prime (cheaper)
                idx = random.randint(0, len(active_primes) - 1)
                old_p = active_primes[idx]
                old_r = residues[idx]
                
                # Choose a new prime
                # To maintain distinctness, we should ideally pick from available primes.
                # For simplicity, pick a random prime from the list.
                # If it's already in the set, we might create a duplicate.
                # To avoid duplicates, we can check.
                # Since K is ~100 and total primes ~168, duplicates are likely if random.
                # Let's pick from a pool of primes not currently in active_primes.
                # This is O(K) to check.
                
                # Optimization: Pre-compute a list of all primes.
                # active_set = set(active_primes)
                # available = [p for p in primes_list if p not in active_set]
                # This is slow if done every step.
                
                # Alternative: Just pick a random prime. If it's in the set, skip or handle swap.
                # If we swap p_old with p_new and p_new is already at idx2, we now have two p_new.
                # This is invalid.
                # So we must ensure p_new is not in active_primes.
                
                # Let's maintain a set of active primes for O(1) lookup.
                active_prime_set = set(active_primes)
                available_primes = [p for p in primes_list if p not in active_prime_set]
                
                if not available_primes:
                    continue
                    
                new_p = random.choice(available_primes)
                
                # Budget check
                new_sum = current_sum_float + (1.0 / new_p) - (1.0 / old_p)
                if new_sum > 1.0:
                    continue
                    
                # Residue: keep same residue value? Or random?
                # new_r = old_r % new_p
                new_r = random.randint(0, new_p - 1)
                
                new_L, _ = apply_prime_swap(idx, new_p, new_r)
                delta_L = new_L - current_L
                
                # Update active_prime_set
                active_prime_set.remove(old_p)
                active_prime_set.add(new_p)
                
                if delta_L >= 0:
                    current_L = new_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_primes_global = active_primes[:]
                        best_residues_global = residues[:]
                        # Recalculate exact sum for best
                        best_sum_frac_global = sum(Fraction(1, p) for p in best_primes_global)
                else:
                    accept_prob = np.exp(delta_L / T)
                    if random.random() < accept_prob:
                        current_L = new_L
                    else:
                        # Revert
                        apply_prime_swap(idx, old_p, old_r)
                        active_prime_set.remove(new_p)
                        active_prime_set.add(old_p)
                        
            T *= cooling
            if T < T_min:
                T = T_min

    # Final Verification and Return
    # We must ensure the best result is valid (distinct primes, sum <= 1).
    
    if not best_primes_global:
        # Fallback: return a valid small set
        return [[13, 1], [17, 1], [19, 1]]
        
    # Check distinctness
    if len(set(best_primes_global)) != len(best_primes_global):
        # This shouldn't happen with our swap logic, but just in case
        # Remove duplicates
        seen = set()
        unique_primes = []
        unique_residues = []
        for p, r in zip(best_primes_global, best_residues_global):
            if p not in seen:
                seen.add(p)
                unique_primes.append(p)
                unique_residues.append(r)
        best_primes_global = unique_primes
        best_residues_global = unique_residues
        
    # Check budget
    final_sum = sum(Fraction(1, p) for p in best_primes_global)
    if final_sum > 1:
        # If invalid, we need to fix it.
        # Simplest: remove primes with largest 1/p (smallest p) until sum <= 1?
        # Or just return a known valid set.
        # Let's try to trim.
        # Sort by p ascending (largest cost first)
        pairs = sorted(zip(best_primes_global, best_residues_global), key=lambda x: x[0])
        while final_sum > 1 and pairs:
            p, r = pairs.pop(0) # Remove smallest prime (largest cost)
            final_sum -= Fraction(1, p)
            best_primes_global = [x[0] for x in pairs]
            best_residues_global = [x[1] for x in pairs]
            
    return [[p, r] for p, r in zip(best_primes_global, best_residues_global)]

# Keep the function signature as requested
# The function is `construct`

if __name__ == "__main__":
    # For local testing
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Result: {res[:10]}...")
    # Verify
    primes = [p for p, r in res]
    residues = [r for p, r in res]
    s = sum(Fraction(1, p) for p in primes)
    print(f"Sum: {s}, Valid: {s <= 1}")
    # Check L
    L = 0
    for i in range(1, 10000):
        covered = False
        for p, r in res:
            if i % p == r:
                covered = True
                break
        if covered:
            L = i
        else:
            break
    print(f"L: {L}")
