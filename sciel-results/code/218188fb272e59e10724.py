"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _calculate_score(residues, primes, max_check=100000):
    """
    Calculate the largest L such that every integer 1..L satisfies at least two congruences.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s limit
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Greedy set-cover heuristic: process primes in increasing order.
    # For each prime, select the residue that maximizes the number of integers
    # in the current critical interval [1, current_L] that are covered by exactly
    # one congruence, thereby converting single-covered positions to double-covered.
    
    max_check = 100000
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Initialize with all zeros
    for i, p in enumerate(primes):
        residues[i] = 0
    
    # We'll track the current prefix length L where all positions 1..L have count >= 2
    # Start by computing initial coverage
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    # Find current L
    def get_current_L():
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]
    
    current_L = get_current_L()
    
    # Now process primes in increasing order, trying to extend the prefix
    # For each prime, try all possible residues and pick the one that maximizes
    # the number of positions in [1, current_L] that go from count=1 to count=2
    
    for idx in range(n_primes):
        if time.time() > deadline - 5:
            break
            
        p = primes[idx]
        old_r = residues[idx]
        
        # Remove old contribution
        old_start = old_r if old_r > 0 else p
        if old_start <= max_check:
            counts[old_start::p] -= 1
        
        best_r = old_r
        best_gain = -1
        
        # Try all possible residues for this prime
        # For each residue r, the positions covered are r, r+p, r+2p, ...
        # We want to maximize the number of positions in [1, current_L] that have count == 1
        # (before adding this prime's contribution)
        
        # Efficient approach: for each candidate r, compute how many positions in [1, current_L]
        # are covered by this residue AND have current count == 1
        
        # Precompute which positions in [1, current_L] have count == 1
        single_covered = (counts[1:current_L+1] == 1)
        
        # For each candidate residue r (0 <= r < p), the positions covered in [1, current_L] are:
        # r, r+p, r+2p, ... up to current_L
        # We need to count how many of these have single_covered[position] == True
        
        # Vectorized approach: for each r, we can check positions r, r+p, ...
        # But iterating over all r (up to p, which can be ~1000) and checking up to current_L/p positions
        # might be expensive. Let's try a smarter approach.
        
        # Alternative: for each position x in [1, current_L] with count==1, find which residue r
        # would cover it: r = x % p. Then for each such x, increment a counter for residue r.
        # The residue with the highest counter is the best choice.
        
        # This is O(current_L) per prime, which is acceptable.
        
        residue_gains = np.zeros(p, dtype=np.int32)
        
        # Get all positions in [1, current_L] that are single-covered
        single_positions = np.where(single_covered)[0] + 1  # Convert to 1-indexed positions
        
        if len(single_positions) > 0:
            # For each single-covered position x, the residue that would cover it is x % p
            # But we need to be careful: if x % p == 0, then the residue is 0
            # Otherwise, the residue is x % p
            residues_for_positions = single_positions % p
            # Count how many times each residue appears
            np.add.at(residue_gains, residues_for_positions, 1)
        
        # Find the residue with maximum gain
        best_r = np.argmax(residue_gains)
        best_gain = residue_gains[best_r]
        
        # Also consider that we might want to keep the old residue if it's better
        # But the old residue was removed, so its gain is already accounted for in the counts
        # Actually, we should compare: what was the gain of the old residue?
        # The old residue's gain would be the number of single-covered positions it would have covered.
        # But since we removed it, those positions are now in the counts without this prime's contribution.
        
        # Let's just pick the best residue based on current single-covered positions
        residues[idx] = best_r
        
        # Add new contribution
        new_start = best_r if best_r > 0 else p
        if new_start <= max_check:
            counts[new_start::p] += 1
        
        # Update current_L
        new_L = get_current_L()
        if new_L > current_L:
            current_L = new_L
    
    # Now do local search to further improve
    best_score = get_current_L()
    best_residues = residues.copy()
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # Iterate through primes in random order
        idxs = np.random.permutation(n_primes)
        
        for i in idxs:
            if time.time() > deadline - 5:
                break
                
            p = primes[i]
            old_r = residues[i]
            
            # Try a few random new residues
            for _ in range(20):
                new_r = np.random.randint(0, p)
                if new_r == old_r:
                    continue
                
                # Apply change
                old_start = old_r if old_r > 0 else p
                new_start = new_r if new_r > 0 else p
                
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                residues[i] = new_r
                new_score = get_current_L()
                
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
                    residues[i] = old_r
            
            if improved:
                pass
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
