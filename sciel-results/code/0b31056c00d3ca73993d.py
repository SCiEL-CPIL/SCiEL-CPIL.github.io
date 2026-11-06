# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_list = primes
    primes_arr = np.array(primes_list, dtype=np.int64)

    # Evaluation bound
    EVAL_BOUND = 10000

    def compute_counts(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        return int(below[0])

    def evaluate(residues):
        counts = compute_counts(residues)
        return get_L_from_counts(counts)

    def evaluate_local(residues, L_current, window=200):
        """Evaluate up to L_current + window for faster feedback during search."""
        bound = min(L_current + window, EVAL_BOUND)
        counts = np.zeros(bound + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= bound:
                idx = np.arange(first, bound + 1, p)
                counts[idx] += 1
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return bound
        return int(below[0])

    def greedy_fill_gaps_smart(current, current_L, max_steps=3000):
        """
        Smart gap filling: at each gap, evaluate multiple candidate primes
        by their local impact, then commit to the best one.
        """
        L = current_L
        steps = 0
        
        while L < EVAL_BOUND and steps < max_steps:
            if time.time() > deadline - 2.0:
                break
            
            gap_pos = L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            # Find primes that don't cover gap_pos
            candidates = []
            for i, p in enumerate(primes_list):
                if current[i] != gap_pos % p:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Sample candidates for efficiency
            if len(candidates) > 50:
                sampled = random.sample(candidates, 50)
            else:
                sampled = candidates
            
            best_candidate = None
            best_new_L = L
            
            # Try each candidate and see which extends L the most
            for i in sampled:
                if time.time() > deadline - 2.0:
                    break
                
                p = primes_list[i]
                old_r = current[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                current[i] = new_r
                new_L = evaluate_local(current, L, window=100)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_candidate = i
                    best_old_r = old_r
                
                current[i] = old_r
            
            if best_candidate is None:
                break
            
            # Commit to the best candidate
            p = primes_list[best_candidate]
            current[best_candidate] = gap_pos % p
            L = evaluate_local(current, L, window=200)
            
            if L > current_L:
                current_L = L
                steps += 1
            else:
                break
        
        return current, current_L

    def local_search_improve(current, current_L, max_iters=100):
        """
        Local search: try small perturbations around the gap to find better solutions.
        """
        L = current_L
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            if L >= EVAL_BOUND:
                break
            
            gap_pos = L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            # Try changing 1-2 residues near the gap
            improved = False
            num_changes = random.choice([1, 1, 2])
            
            for attempt in range(20):
                if time.time() > deadline - 2.0:
                    break
                
                trial = current[:]
                changed_indices = []
                
                for _ in range(num_changes):
                    # Prefer primes that don't currently cover the gap
                    candidates = [i for i in range(n_primes) if trial[i] != gap_pos % primes_list[i]]
                    if not candidates:
                        break
                    idx = random.choice(candidates)
                    p = primes_list[idx]
                    old_val = trial[idx]
                    trial[idx] = random.randint(0, p - 1)
                    changed_indices.append((idx, old_val))
                
                if not changed_indices:
                    continue
                
                new_L = evaluate(trial)
                
                if new_L > L:
                    current = trial
                    L = new_L
                    improved = True
                    no_improve = 0
                    break
                else:
                    # Revert
                    for idx, old_val in changed_indices:
                        current[idx] = old_val
            
            if not improved:
                no_improve += 1
                if no_improve >= 10:
                    break
        
        return current, L

    # Initial best
    best_L = 0
    best_residues = [0] * n_primes

    random.seed(42)

    # Phase 1: Quick random restarts to find a decent starting point
    quick_restart_limit = start_time + 8.0
    while time.time() < quick_restart_limit:
        current = [random.randint(0, p - 1) for p in primes_list]
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Phase 2: Smart greedy gap filling from best
    if best_L > 0:
        current = best_residues[:]
        current_L, current = greedy_fill_gaps_smart(current, current_L, max_steps=2000)
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Phase 3: Iterative improvement
    while time.time() < deadline - 10.0:
        # Start from best and perturb
        if best_L > 0 and random.random() < 0.7:
            current = best_residues[:]
            # Perturb a small fraction of residues
            num_perturb = max(1, n_primes // 15)
            for _ in range(num_perturb):
                idx = random.randint(0, n_primes - 1)
                p = primes_list[idx]
                current[idx] = random.randint(0, p - 1)
        else:
            current = [random.randint(0, p - 1) for p in primes_list]
        
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
        
        # Smart greedy fill from this point
        if current_L < EVAL_BOUND and time.time() < deadline - 5.0:
            current_L, current = greedy_fill_gaps_smart(current, current_L, max_steps=1500)
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
        
        # Local search refinement
        if best_L < EVAL_BOUND and time.time() < deadline - 3.0:
            current_L, current = local_search_improve(current, best_L, max_iters=50)
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
        
        # If we've reached the target, stop
        if best_L >= 1000:
            break
    
    # Final polish: one more greedy pass
    if best_L > 0 and time.time() < deadline - 2.0:
        current = best_residues[:]
        current_L, current = greedy_fill_gaps_smart(current, current_L, max_steps=3000)
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Ensure Python integers
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
