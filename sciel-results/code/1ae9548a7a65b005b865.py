import numpy as np
import time
import random

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_size = 0
    best_points = []
    
    x_indices = np.arange(RANGE)
    
    # Precompute residues for all primes up to N
    # primes_residues[p] = x_indices % p
    primes_residues = {}
    for p in PRIMES:
        if p <= RANGE:
            primes_residues[p] = x_indices % p

    def _greedy_random(x_indices, PRIMES, seed=None):
        """Randomized greedy residue removal."""
        if seed is not None:
            random.seed(seed)
        
        active = np.ones(RANGE, dtype=bool)
        current_size = RANGE
        
        for p in PRIMES:
            if p > current_size:
                break
            
            # Count active elements in each residue class mod p
            # Using precomputed residues if available, else compute
            if p in primes_residues:
                residues = primes_residues[p][active]
            else:
                residues = x_indices[active] % p
                
            if len(residues) == 0:
                break
            
            # Count occurrences of each residue
            counts = np.bincount(residues, minlength=p)
            min_count = np.min(counts)
            
            if min_count == 0:
                continue
            
            candidate_classes = np.where(counts == min_count)[0]
            r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
            
            # Deactivate elements in this class
            # active = active & (x_indices % p != r)
            # Optimization: use precomputed residue
            if p in primes_residues:
                mask = (primes_residues[p] == r)
            else:
                mask = (x_indices % p) == r
            
            current_size -= min_count
            active = active & ~mask
            
            if current_size == 0:
                break
        
        return active

    def _fast_local_repair(active, x_indices, PRIMES, max_adds=2000):
        """
        Fast local repair: try to add back removed elements while maintaining admissibility.
        This is a hill-climbing step that increases size if possible.
        """
        active = active.copy()
        current_size = int(np.sum(active))
        
        # Precompute counts for current active set for relevant primes
        # We only care about primes p <= current_size + 1 (since adding 1 might make p relevant if p == current_size+1)
        # Actually, if we add an element, size becomes current_size + 1.
        # So we need to check primes p <= current_size + 1.
        
        # Let's maintain counts for primes p <= current_size + 5 (buffer)
        max_p_check = current_size + 10
        relevant_primes = [p for p in PRIMES if p <= max_p_check]
        
        prime_counts = {}
        for p in relevant_primes:
            if p in primes_residues:
                res = primes_residues[p][active]
            else:
                res = x_indices[active] % p
            prime_counts[p] = np.bincount(res, minlength=p)
        
        removed = np.where(~active)[0]
        random.shuffle(removed.tolist())
        
        added = 0
        for x in removed.tolist():
            if added >= max_adds:
                break
            
            can_add = True
            new_size = current_size + 1
            
            # Check all primes p <= new_size
            for p in PRIMES:
                if p > new_size:
                    break
                
                if p not in prime_counts:
                    # Compute counts for this prime if not cached
                    if p in primes_residues:
                        res = primes_residues[p][active]
                    else:
                        res = x_indices[active] % p
                    prime_counts[p] = np.bincount(res, minlength=p)
                
                counts_p = prime_counts[p]
                res = x % p
                old_count = counts_p[res]
                
                # If we add x, count for res becomes old_count + 1.
                # The condition for admissibility is that NOT all residue classes are covered.
                # i.e., min(counts) must be 0.
                # Before adding, min(counts) is 0 (since it's admissible).
                # Let zero_count = number of residue classes with count 0.
                # If we add to a class with count > 0, zero_count doesn't change. min remains 0.
                # If we add to a class with count == 0, zero_count decreases by 1.
                # If zero_count becomes 0, then all classes are covered -> not admissible.
                
                if old_count == 0:
                    # This was an empty class.
                    # How many empty classes are there?
                    # We need to know if this is the LAST empty class.
                    # If it is the last one, adding it makes all classes non-empty -> invalid.
                    zero_count = np.sum(counts_p == 0)
                    if zero_count == 1:
                        can_add = False
                        break
                # If old_count > 0, adding keeps it > 0, so empty classes remain empty.
                # So it's always safe?
                # Wait, what if new_size > p?
                # The condition is: for every prime p <= len(tuple), some residue class is missed.
                # If p <= current_size, it was admissible, so some class was empty.
                # Adding an element to a non-empty class doesn't fill the empty ones.
                # Adding an element to an empty class fills it.
                # So the only risk is filling the last empty class.
                
            if can_add:
                active[x] = True
                current_size = new_size
                added += 1
                
                # Update counts for relevant primes
                # For primes p <= new_size:
                for p in PRIMES:
                    if p > new_size:
                        break
                    if p in prime_counts:
                        res = x % p
                        prime_counts[p][res] += 1
                    # If p was not in prime_counts, we don't strictly need to update it yet
                    # unless it becomes relevant in the next iteration?
                    # But we check p <= new_size in the next iteration's can_add loop.
                    # So if p was > old_size but <= new_size, it wasn't in prime_counts.
                    # We will compute it on the fly in the next iteration if needed.
                    # But for efficiency, we could add it here.
                    # However, adding to prime_counts dict dynamically is fine.
        
        return active

    # Main loop: Simulated Annealing / Random Restart with Local Repair
    # The "state" is the active mask.
    # Objective: Maximize size.
    # Moves: 
    # 1. Greedy random construction from scratch.
    # 2. Local repair on an existing state.
    # 3. Mutation: Remove a random element, then try to repair? (Maybe too complex)
    
    # Strategy:
    # 1. Run several greedy random constructions.
    # 2. For the best few, run local repair.
    # 3. Iterate.
    
    states = []
    
    # Initial batch of greedy constructions
    num_initial = 50
    for i in range(num_initial):
        if time.time() > deadline - 10:
            break
        active = _greedy_random(x_indices, PRIMES, seed=i)
        size = int(np.sum(active))
        states.append((active, size))
        
        if size > best_size:
            best_size = size
            best_points = np.where(active)[0].tolist()
    
    # Sort by size descending
    states.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top K states for further refinement
    K = 10
    states = states[:K]
    
    # Iterative refinement
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        new_states = []
        
        for active, size in states:
            # 1. Local Repair
            if time.time() > deadline - 5:
                break
            
            repaired_active = _fast_local_repair(active, x_indices, PRIMES, max_adds=500)
            repaired_size = int(np.sum(repaired_active))
            
            if repaired_size > best_size:
                best_size = repaired_size
                best_points = np.where(repaired_active)[0].tolist()
            
            new_states.append((repaired_active, repaired_size))
            
            # 2. Generate new random states to diversify
            # Sometimes restart from scratch is better
            if len(new_states) < 2 * K:
                if time.time() > deadline - 5:
                    break
                new_active = _greedy_random(x_indices, PRIMES, seed=random.randint(0, 1000000))
                new_size = int(np.sum(new_active))
                new_states.append((new_active, new_size))
                
                if new_size > best_size:
                    best_size = new_size
                    best_points = np.where(new_active)[0].tolist()
        
        if not new_states:
            break
            
        # Prune and sort
        new_states.sort(key=lambda x: x[1], reverse=True)
        states = new_states[:K]
        
        # If the best size hasn't improved in a few iterations, we might be stuck.
        # But with random restarts, it should eventually find something better or we run out of time.
        
        # Optional: Aggressive mutation
        # Pick the best state, remove a random element, repair?
        # This might help escape local optima in the repair phase.
        if states:
            best_active, best_s = states[0]
            if best_s < RANGE and time.time() < deadline - 5:
                # Try to "un-stick" by removing a random active element and repairing
                # This is a form of simulated annealing step (downhill move)
                active_indices = np.where(best_active)[0]
                if len(active_indices) > 0:
                    idx_to_remove = active_indices[random.randint(0, len(active_indices)-1)]
                    mutated_active = best_active.copy()
                    mutated_active[idx_to_remove] = False
                    mutated_size = best_s - 1
                    
                    # Repair the mutated state
                    repaired_mutated = _fast_local_repair(mutated_active, x_indices, PRIMES, max_adds=500)
                    rm_size = int(np.sum(repaired_mutated))
                    
                    if rm_size > best_size:
                        best_size = rm_size
                        best_points = np.where(repaired_mutated)[0].tolist()
                    
                    # Add to states if it's decent
                    if rm_size > best_s - 10: # Keep if it's close
                         states.append((repaired_mutated, rm_size))
                         states.sort(key=lambda x: x[1], reverse=True)
                         states = states[:K]

    return best_points
