# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (150s limit, leave buffer)

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

    def evaluate_local(residues, focus_start, focus_end):
        """Evaluate only in a local range for faster feedback."""
        if focus_start < 1:
            focus_start = 1
        if focus_end > EVAL_BOUND:
            focus_end = EVAL_BOUND
        if focus_start > focus_end:
            return 0
            
        counts = np.zeros(focus_end + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= focus_start and first <= focus_end:
                idx = np.arange(first, focus_end + 1, p)
                counts[idx] += 1
            elif first < focus_start:
                # Need to find the first occurrence >= focus_start
                # k = ceil((focus_start - first) / p)
                # new_first = first + k * p
                diff = focus_start - first
                k = (diff + p - 1) // p
                new_first = first + k * p
                if new_first <= focus_end:
                    idx = np.arange(new_first, focus_end + 1, p)
                    counts[idx] += 1
        
        # Check if all positions in [focus_start, focus_end] have count >= 2
        # But we also need to ensure the prefix up to focus_start-1 is still valid
        # For local eval, we just check the local window
        local_counts = counts[focus_start:focus_end+1]
        if np.all(local_counts >= 2):
            return focus_end
        else:
            below = np.where(local_counts < 2)[0]
            return focus_start + int(below[0]) - 1

    def greedy_fill_gaps(residues, current_L, max_steps=500):
        """
        Greedily try to extend current_L by filling gaps.
        Returns (new_L, new_residues)
        """
        counts = compute_counts(residues)
        L = current_L
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            if L >= EVAL_BOUND:
                break
            
            gap_pos = L + 1
            
            # Find primes that don't cover gap_pos
            candidates = []
            for i, p in enumerate(primes_list):
                if residues[i] != gap_pos % p:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Score candidates: prefer primes with larger p (less frequent, so changing them affects fewer positions)
            # Also prefer primes where the new residue doesn't create new gaps elsewhere
            scored_candidates = []
            for i in candidates:
                p = primes_list[i]
                new_r = gap_pos % p
                old_r = residues[i]
                
                # Heuristic: larger primes are "safer" to change
                # But we also want to check if this change helps nearby positions
                score = -p  # Prefer larger primes (negative so we sort ascending)
                
                # Check impact on positions near gap_pos
                # If we change residue for prime p, positions that were covered by old_r but not new_r lose coverage
                # Positions that were not covered by old_r but are covered by new_r gain coverage
                
                # Simple heuristic: prefer changes that don't break coverage at positions we care about
                # For now, just use prime size
                scored_candidates.append((score, i))
            
            scored_candidates.sort()
            
            improved = False
            # Try top candidates
            for score, i in scored_candidates[:30]:
                p = primes_list[i]
                old_r = residues[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                residues[i] = new_r
                new_counts = compute_counts(residues)
                new_L = get_L_from_counts(new_counts)
                
                if new_L > L:
                    L = new_L
                    counts = new_counts
                    improved = True
                    break
                else:
                    residues[i] = old_r
            
            if not improved:
                break
        
        return L, residues

    # Initial best
    best_L = 0
    best_residues = [0] * n_primes

    # Strategy 1: Start with a good initial guess
    # For each prime, choose residue to cover position 1
    # x = r_p (mod p), so r_p = 1 % p = 1 (for p > 1)
    # This gives count[1] = 168, which is >= 2
    # But we need to be smarter to extend further
    
    # Let's try a structured initialization:
    # For small primes, choose residues to cover early positions
    # For large primes, they cover fewer positions, so be careful
    
    # Random restarts with Greedy Repair
    random.seed(42)
    
    # Phase 1: Quick random restarts to find a decent starting point
    quick_restart_limit = start_time + 10.0
    
    while time.time() < quick_restart_limit:
        current = [random.randint(0, p - 1) for p in primes_list]
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Phase 2: Greedy gap filling from best
    if best_L > 0:
        current = best_residues[:]
        current_L, current = greedy_fill_gaps(current, current_L, max_steps=1000)
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Phase 3: Iterative improvement with local search
    while time.time() < deadline - 5.0:
        # Start from best and perturb
        if best_L > 0:
            current = best_residues[:]
            # Perturb a small fraction of residues
            num_perturb = max(1, n_primes // 20)
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
        
        # Greedy fill from this point
        if current_L < EVAL_BOUND:
            current_L, current = greedy_fill_gaps(current, current_L, max_steps=500)
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
        
        # If we've reached the target, stop
        if best_L >= 1000:
            break
    
    # Final polish: one more greedy pass
    if best_L > 0 and time.time() < deadline - 2.0:
        current = best_residues[:]
        current_L, current = greedy_fill_gaps(current, current_L, max_steps=2000)
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
    
    # Ensure Python integers
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
