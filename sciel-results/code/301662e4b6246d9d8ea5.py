"""Structural greedy + backward refinement solver for covering problem."""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_count_array(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r = p
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    # Find first position where count < 2
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return deficient[0]

def _score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    counts = _build_count_array(residues, primes, max_L)
    return _find_L(counts, max_L)

def _greedy_construct(primes, max_L_check=2000):
    """
    Forward greedy: process primes in order, for each prime choose the residue
    that maximizes coverage of currently-deficient positions.
    """
    n = len(primes)
    residues = [0] * n
    
    # Start with all zeros, build up incrementally
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    for idx in range(n):
        p = primes[idx]
        
        # For each possible residue, compute how many currently-deficient positions it covers
        # Deficient positions are those with count < 2
        best_r = 0
        best_gain = -1
        
        # We want to cover positions that currently have count 0 or 1
        # A residue r covers positions r, r+p, r+2p, ...
        # For each position x with count[x] < 2, if x % p == r, then this residue helps
        
        # Efficient approach: for each candidate r, count how many deficient positions it covers
        # But p can be up to 997, so we have up to 997 candidates
        # For each candidate, we check positions r, r+p, r+2p, ...
        
        # Optimization: only consider residues that would help
        # A position x is helped by residue r if x ≡ r (mod p) and count[x] < 2
        
        # Let's precompute: for each position x in 1..max_L_check with count[x] < 2,
        # what residue r = x % p would help?
        # Then for each r, count how many such positions exist.
        
        # But this is O(max_L_check) per prime, which is 2000 * 168 = 336000, fine.
        
        # Actually, let's just try a subset of residues for speed
        # Focus on residues that cover early deficient positions
        
        # Find the first few deficient positions
        deficient_positions = []
        for pos in range(1, min(max_L_check + 1, 200)):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 50:
                    break
        
        if not deficient_positions:
            # No deficient positions in early range, pick residue to maximize overall coverage
            # Just pick r that covers the most positions in 1..max_L_check
            best_r = 1  # Default
            best_count = 0
            for r in range(min(p, 50)):  # Limit candidates for speed
                if r == 0:
                    r_eff = p
                else:
                    r_eff = r
                if r_eff > max_L_check:
                    continue
                # Count positions covered
                num_positions = (max_L_check - r_eff) // p + 1
                if num_positions > best_count:
                    best_count = num_positions
                    best_r = r
        else:
            # For each candidate residue, count how many deficient positions it covers
            # Use a histogram approach
            residue_counts = np.zeros(p, dtype=np.int32)
            
            for pos in deficient_positions:
                r_needed = pos % p
                residue_counts[r_needed] += 1
            
            # Also consider: covering a position with count 0 is better than count 1
            # Let's weight: positions with count 0 get weight 2, count 1 get weight 1
            residue_weighted = np.zeros(p, dtype=np.int32)
            for pos in deficient_positions:
                r_needed = pos % p
                weight = 2 if counts[pos] == 0 else 1
                residue_weighted[r_needed] += weight
            
            # Pick the residue with highest weighted count
            # But also consider: we don't want to waste coverage on already-well-covered positions
            # For now, just pick max weighted
            best_r = int(np.argmax(residue_weighted))
            best_gain = residue_weighted[best_r]
        
        residues[idx] = best_r
        
        # Update counts
        r = best_r
        if r == 0:
            r_eff = p
        else:
            r_eff = r
        if r_eff <= max_L_check:
            positions = np.arange(r_eff, max_L_check + 1, p)
            counts[positions] += 1
    
    return residues

def _backward_refine(residues, primes, max_L_check, deadline, num_passes=3):
    """
    Backward refinement: iterate through primes in reverse, for each prime
    try all residues and keep the one that gives best score.
    """
    n = len(primes)
    best_score = _score(residues, primes, max_L_check)
    
    for pass_num in range(num_passes):
        if time.time() > deadline:
            break
        
        improved = False
        # Process in reverse order
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            
            p = primes[idx]
            old_r = residues[idx]
            
            # Build counts without this prime
            counts = _build_count_array(residues, primes, max_L_check)
            # Subtract this prime's contribution
            r = old_r
            if r == 0:
                r_eff = p
            else:
                r_eff = r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] -= 1
            
            # Now try each possible residue for this prime
            best_r = old_r
            best_local_score = _find_L(counts, max_L_check)
            
            # For large primes, limit candidates
            if p > 100:
                candidates = list(range(p))
            else:
                candidates = list(range(p))
            
            for new_r in candidates:
                if time.time() > deadline:
                    break
                
                # Add this prime's contribution with new_r
                if new_r == 0:
                    r_eff = p
                else:
                    r_eff = new_r
                
                if r_eff <= max_L_check:
                    positions = np.arange(r_eff, max_L_check + 1, p)
                    counts[positions] += 1
                
                new_score = _find_L(counts, max_L_check)
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = new_r
                
                # Subtract again for next candidate
                if r_eff <= max_L_check:
                    positions = np.arange(r_eff, max_L_check + 1, p)
                    counts[positions] -= 1
            
            if best_r != old_r:
                residues[idx] = best_r
                improved = True
        
        # Update best score
        new_score = _score(residues, primes, max_L_check)
        if new_score > best_score:
            best_score = new_score
        else:
            break
    
    return residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Phase 1: Greedy construction with small max_L
    max_L_small = 500
    residues = _greedy_construct(primes, max_L_small)
    
    # Phase 2: Increase max_L and do backward refinement
    max_L_medium = 2000
    residues, score_med = _backward_refine(residues, primes, max_L_medium, deadline, num_passes=2)
    
    # Phase 3: Further refinement with larger max_L if time permits
    if time.time() < deadline - 30:
        max_L_large = 5000
        residues, score_large = _backward_refine(residues, primes, max_L_large, deadline, num_passes=2)
    
    # Phase 4: Final full-range refinement
    if time.time() < deadline - 15:
        max_L_final = 10000
        residues, score_final = _backward_refine(residues, primes, max_L_final, deadline, num_passes=1)
    
    # Phase 5: One more pass with even larger range if time allows
    if time.time() < deadline - 10:
        max_L_very_large = 20000
        residues, score_vl = _backward_refine(residues, primes, max_L_very_large, deadline, num_passes=1)
    
    # Final evaluation
    final_score = _score(residues, primes, 100000)
    
    return residues
