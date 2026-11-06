"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [i for i in range(n + 1) if sieve[i]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has count >= 2."""
    L = min(cap, 50000)
    counts = np.zeros(L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= L:
            counts[start::p] += 1
    if L > 0 and counts[1] < 2:
        return 0
    mask = counts[1:] < 2
    if not mask.any():
        return L
    first_bad = np.argmax(mask) + 1
    return first_bad

def _score(residues, primes, cap=100000):
    return _compute_L(residues, primes, cap)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)  # 168
    
    # Greedy set-cover approach:
    # Process primes in increasing order. For each prime, choose the residue
    # that maximizes the number of currently single-covered positions in [1, current_L]
    # that would become double-covered. This systematically builds up coverage.
    
    # Start with a reasonable search window
    max_L = 5000
    
    # Initialize: all residues = 0 (covers multiples of p)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Build counts array incrementally
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # Greedy: process primes in order of increasing p
    # For each prime, try all residues and pick the one that best helps
    # cover single-covered positions while not breaking any double-covered ones
    
    # First, let's try a smarter initialization:
    # Set residues so that small numbers get covered by many primes
    # For position k in 1..max_L, it's covered by prime p if k ≡ r_p (mod p)
    # i.e., r_p = k % p (with 0 meaning p divides k)
    
    # Strategy: For each prime p, choose r_p to cover as many "need coverage" 
    # positions as possible. A position needs coverage if it's currently covered 
    # by only 0 or 1 primes.
    
    # We'll iterate: for each prime (in order), pick the best residue
    # "best" = maximizes number of positions in [1, max_L] that go from 
    # count=1 to count=2, minus a penalty for positions that go from 
    # count>=2 to count=3 (wasted) or count=0 to count=1 (insufficient)
    
    # Actually, let's think differently. We want every position in [1,L] to have
    # count >= 2. The bottleneck is the first position with count < 2.
    
    # New approach: 
    # 1. Start with all r_p = 1 (position 1 covered by all primes)
    # 2. For each prime p (in increasing order), try to find a residue that
    #    extends the doubly-covered prefix
    # 3. Use local search to improve
    
    # Let me try a completely different structural approach:
    # Simulated annealing with a better objective function
    
    # Objective: maximize L (the doubly-covered prefix length)
    # But also penalize positions near the boundary that are at risk
    
    # Let's use a weighted objective:
    # score = sum_{k=1}^{max_L} w_k * I(count[k] >= 2)
    # where w_k = 1 for k <= current_L, w_k = 0.1 for k > current_L
    
    # Actually, let's just do a focused greedy + local search
    
    # Phase 1: Greedy initialization
    # For each prime p in increasing order, choose r_p to maximize coverage
    # of positions that are currently under-covered
    
    # Start: all r_p = 0
    # counts[k] = number of primes p such that k ≡ 0 (mod p), i.e., p | k
    # This gives good coverage for composite numbers but poor for primes
    
    # Better start: r_p = 1 for all p
    # Then position 1 is covered by all 168 primes
    # Position k is covered by primes p where k ≡ 1 (mod p)
    
    # Let me try: r_p = 1 for all, then greedily adjust
    
    residues[:] = 1
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = 1
        if r > 0:
            start = r
        else:
            start = p
        if start <= max_L:
            counts[start::p] += 1
    
    def get_L(counts_arr):
        if counts_arr[1] < 2:
            return 0
        mask = counts_arr[1:] < 2
        if not mask.any():
            return max_L
        return int(np.argmax(mask)) + 1
    
    current_L = get_L(counts)
    best_L = current_L
    best_res = residues.copy()
    
    # Phase 2: Greedy improvement
    # For each prime (in random order), try to find a better residue
    # that extends the prefix
    
    rng = np.random.default_rng(12345)
    
    # Precompute: for each prime p and each possible residue r,
    # which positions in [1, max_L] are covered
    # This is just the pattern r, r+p, r+2p, ...
    
    # For efficiency, we'll try a limited set of candidate residues for each prime
    # Key insight: for prime p, residue r covers positions r, r+p, r+2p, ...
    # To help cover position k, we need r ≡ k (mod p), i.e., r = k % p
    # (with the convention that 0 means p divides k)
    
    # So for a given set of "critical" positions (those with count < 2),
    # the best residue for prime p is the one that matches the most critical positions
    
    # Let's implement this:
    
    def find_best_residue(p, counts_arr, critical_mask):
        """Find the residue for prime p that covers the most critical positions."""
        # Critical positions are those where count < 2
        # For residue r, it covers positions r, r+p, r+2p, ...
        # We want to maximize the number of critical positions covered
        
        # For each critical position k, the required residue is k % p
        # (with 0 meaning k is divisible by p)
        
        # Count how many critical positions need each residue
        crit_positions = np.where(critical_mask)[0]
        if len(crit_positions) == 0:
            return 1  # default
        
        # For each critical position k, the residue needed is k % p
        # But we also need to consider that changing r_p affects all positions
        # in the arithmetic progression
        
        # Simpler: for each candidate residue r, count how many critical
        # positions are in the progression r, r+p, r+2p, ...
        
        # Candidate residues: 0, 1, 2, ..., p-1
        # But for large p, this is too many. Limit to residues that appear
        # as k % p for some critical position k
        
        # Get unique residues needed by critical positions
        needed_residues = np.unique(crit_positions % p)
        
        best_r = 1
        best_score = -1
        
        for r in needed_residues:
            r = int(r)
            # Count critical positions covered by residue r
            # Positions: r, r+p, r+2p, ... up to max_L
            # (with r=0 meaning p, 2p, 3p, ...)
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                continue
            # Number of positions in [start, max_L] with step p
            n_pos = (max_L - start) // p + 1
            # How many of these are critical?
            # We can check directly
            positions = np.arange(start, max_L + 1, p)
            n_crit = np.sum(critical_mask[positions])
            
            # Also consider: does this break any already-good positions?
            # A position k is "good" if count[k] >= 2
            # If we change r_p, we lose coverage at old positions and gain at new
            # For now, just maximize critical coverage
            
            if n_crit > best_score:
                best_score = n_crit
                best_r = r
        
        return best_r
    
    # Iterative greedy
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Find critical positions (count < 2) in [1, current_L + some_margin]
        search_end = min(max_L, current_L + 500)
        critical_mask = np.zeros(max_L + 1, dtype=bool)
        critical_mask[1:search_end+1] = counts[1:search_end+1] < 2
        
        if not critical_mask.any():
            # No critical positions, try to extend further
            break
        
        # For each prime (in random order), try to improve
        order = rng.permutation(n_primes)
        
        for i in order:
            if time.time() > deadline:
                break
            
            p = primes[i]
            old_r = int(residues[i])
            
            # Find best residue for this prime
            new_r = find_best_residue(p, counts, critical_mask)
            
            if new_r == old_r:
                continue
            
            # Apply the change and check if it improves
            # Remove old coverage
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if old_start <= max_L:
                counts[old_start::p] -= 1
            
            # Add new coverage
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            if new_start <= max_L:
                counts[new_start::p] += 1
            
            new_L = get_L(counts)
            
            if new_L > current_L:
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_res = residues.copy()
            else:
                # Revert
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                if new_start <= max_L:
                    counts[new_start::p] -= 1
                
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if old_start <= max_L:
                    counts[old_start::p] += 1
    
    # Phase 3: Local search / hill climbing to fine-tune
    # Try small perturbations to see if we can extend further
    
    for pass_num in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n_primes)
        
        for i in order:
            if time.time() > deadline:
                break
            
            p = primes[i]
            old_r = int(residues[i])
            
            # Try a few random nearby residues
            n_tries = min(20, p)
            for _ in range(n_tries):
                new_r = int(rng.integers(0, p))
                if new_r == old_r:
                    continue
                
                # Apply change
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if old_start <= max_L:
                    counts[old_start::p] -= 1
                
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                if new_start <= max_L:
                    counts[new_start::p] += 1
                
                new_L = get_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_res = residues.copy()
                    break
                else:
                    # Revert
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    if new_start <= max_L:
                        counts[new_start::p] -= 1
                    
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if old_start <= max_L:
                        counts[old_start::p] += 1
        
        if not improved:
            # Try a different random seed order
            continue
    
    # Final: compute the actual score
    final_score = _compute_L(best_res.tolist(), primes, cap=100000)
    
    return best_res.tolist()
# EVOLVE-BLOCK-END
