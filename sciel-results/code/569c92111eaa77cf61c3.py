"""Rewritten program: greedy small-prime assignment + deficit-based large-prime placement."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    
    # We'll build coverage counts for positions 1..L_max
    # Start with a reasonable L_max; we'll verify at the end
    L_max = 20000
    
    # coverage[x] = number of congruences satisfied by x
    coverage = [0] * (L_max + 1)
    
    # residues[i] = r_{p_i}
    residues = [0] * n_primes
    
    # Phase 1: Assign small primes (p <= 50) greedily
    # For each small prime, try each residue and pick the one that maximizes
    # the minimum coverage in [1, some window]
    small_threshold = 50
    
    # For small primes, we want to cover early positions densely
    # Strategy: for prime p, choose r to maximize coverage in [1, 200]
    for i, p in enumerate(primes):
        if p > small_threshold:
            break
        
        best_r = 0
        best_score = -1
        
        # Try each residue; evaluate by sum of coverage in [1, 300] weighted by deficit
        # Actually, let's just maximize the number of new positions covered in [1, 300]
        # that currently have coverage < 2
        for r in range(p):
            # Count how many positions in [1, 300] with x ≡ r (mod p) have coverage < 2
            gain = 0
            x = r if r > 0 else p  # first positive x with x ≡ r (mod p)
            if r == 0:
                x = p
            else:
                x = r
            while x <= 300:
                if coverage[x] < 2:
                    gain += 1
                x += p
            # Also weight by how much this helps the minimum
            # Simpler: just maximize gain
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[i] = best_r
        # Update coverage
        x = best_r if best_r > 0 else p
        if best_r == 0:
            x = p
        else:
            x = best_r
        while x <= L_max:
            coverage[x] += 1
            x += p
    
    # Phase 2: Assign large primes (p > 50) using deficit-based placement
    # For each large prime, it covers ~L_max/p positions. For very large p, only 1-2 positions.
    # Strategy: place the coverage at positions with the lowest current coverage (deficit)
    
    for i, p in enumerate(primes):
        if p <= small_threshold:
            continue
        
        # Find the best residue r that minimizes the maximum deficit
        # or maximizes coverage at deficient positions
        # For large p, only 1 or 2 positions in [1, L_max] are covered
        # We want to cover positions that are most "deficient"
        
        # Collect candidate positions and their coverage
        # For residue r, positions are r, r+p, r+2p, ... up to L_max
        # For p > L_max/2, only 1 position
        # For p > L_max/3, at most 2 positions
        
        best_r = 0
        best_deficit_score = -1
        
        # For efficiency, only check residues that hit positions in [1, 500]
        # (the early positions are the bottleneck)
        # Actually, let's check all residues but score based on deficit in [1, 1000]
        
        # Precompute: for each position x in [1, min(L_max, 2000)], what's its coverage?
        # For residue r, the positions hit in [1, 2000] are: r (if r>0), r+p, r+2p, ...
        
        # Score: sum of (2 - coverage[x]) for all x ≡ r (mod p) in [1, 2000] where coverage[x] < 2
        # This maximizes help to deficient positions
        
        max_check = min(L_max, 3000)
        
        for r in range(p):
            score = 0
            x = r if r > 0 else p
            if r == 0:
                x = p
            else:
                x = r
            while x <= max_check:
                if coverage[x] < 2:
                    score += (2 - coverage[x])
                x += p
            if score > best_deficit_score:
                best_deficit_score = score
                best_r = r
        
        residues[i] = best_r
        # Update coverage
        x = best_r if best_r > 0 else p
        if best_r == 0:
            x = p
        else:
            x = best_r
        while x <= L_max:
            coverage[x] += 1
            x += p
    
    # Verify: find the largest L such that every integer 1..L has coverage >= 2
    L = 0
    for x in range(1, L_max + 1):
        if coverage[x] >= 2:
            L = x
        else:
            break
    
    return residues
# EVOLVE-BLOCK-END
