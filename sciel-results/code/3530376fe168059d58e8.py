import time

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    LIMIT = 5000  # Increased limit to look further ahead
    
    # counts[x] = number of primes covering x
    counts = [0] * (LIMIT + 1)
    residues = [0] * n_primes
    
    current_L = 0
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        
        # We want to maximize the number of positions in [1, LIMIT] that are converted from 1 to 2
        # or 0 to 1? The directive says: "maximizes the number of integers in the current critical interval [1, current_L] that are covered by exactly one congruence"
        # But actually, to extend L, we need positions beyond current_L to become >= 2.
        # The most effective way to extend L is to cover positions that are currently at count 1 (so they become 2) and also cover positions at count 0 (so they become 1, which helps future primes).
        
        # Let's score a residue r by:
        # 1. How many positions x in [current_L + 1, LIMIT] have counts[x] == 1 and x % p == r? (These become 2, potentially extending L)
        # 2. How many positions x in [current_L + 1, LIMIT] have counts[x] == 0 and x % p == r? (These become 1, helpful for future)
        # 3. Also, we should consider positions in [1, current_L] that have count 1? They become 2, but they don't extend L. However, they might be important if we consider that the "critical interval" might shift? No, L is fixed.
        # The directive specifically mentions "current critical interval [1, current_L]". But if we only look at [1, current_L], we won't extend L at all, because those are already >= 2.
        # Wait, the directive says: "selecting for each prime the residue that maximizes the number of integers in the current critical interval [1, current_L] that are covered by exactly one congruence".
        # This seems counter-intuitive for extending L. If [1, current_L] is all >= 2, then there are no integers covered by exactly one congruence in [1, current_L].
        # Perhaps the "critical interval" refers to the frontier? Or perhaps the intention is to maximize the "potential" for extension.
        
        # Let's re-read carefully: "maximizes the number of integers in the current critical interval [1, current_L] that are covered by exactly one congruence, thereby systematically converting single-covered positions into double-covered ones to extend the prefix."
        # This implies that there ARE single-covered positions in [1, current_L]? That contradicts the definition of current_L (largest L such that [1, L] is >= 2).
        # Unless... the definition of current_L in the seed program is slightly different?
        # Seed: "current_L is defined as the largest L such that [1, L] is >= 2."
        # So [1, current_L] has NO single-covered positions.
        
        # Maybe the directive means the interval [current_L, current_L + window]? Or maybe it means we should look at the positions that are "blocking" the extension?
        # The positions blocking the extension are those in [current_L + 1, ...] that have count 0 or 1.
        # To extend L, we need to increase the count of current_L + 1 to 2.
        
        # Let's interpret the directive as: Maximize the number of positions in a window ahead of current_L that are currently at count 1 (and thus can be promoted to 2) plus a smaller weight for positions at count 0 (promoted to 1).
        
        # Let's try a heuristic:
        # Score(r) = sum(1 for x in [current_L+1, LIMIT] if x%p==r and counts[x]==1) + 0.5 * sum(1 for x in [current_L+1, LIMIT] if x%p==r and counts[x]==0)
        
        # We can compute this efficiently.
        # For a given p, the positions x with x % p == r are r, r+p, r+2p, ...
        
        best_score = -1
        best_r = 0
        
        for r in range(p):
            score = 0
            # Iterate over positions x in [current_L+1, LIMIT] such that x % p == r
            # First such position >= current_L + 1
            start = r if r >= current_L + 1 else r + ((current_L + 1 - r + p - 1) // p) * p
            # Actually, simpler:
            # x = r, r+p, ...
            # We want x >= current_L + 1 and x <= LIMIT
            
            # Find the first x >= current_L + 1 with x % p == r
            # If r == 0, positions are p, 2p, ...
            # If r > 0, positions are r, r+p, ...
            
            # Let's just iterate from the first valid x.
            if r == 0:
                first_x = p
            else:
                first_x = r
            
            # Adjust first_x to be >= current_L + 1
            if first_x < current_L + 1:
                # How many steps of p to get to >= current_L + 1?
                steps = (current_L + 1 - first_x + p - 1) // p
                first_x += steps * p
            
            for x in range(first_x, LIMIT + 1, p):
                if counts[x] == 1:
                    score += 1
                elif counts[x] == 0:
                    score += 0.5
            
            if score > best_score:
                best_score = score
                best_r = r
        
        # Apply the best residue
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        if start <= LIMIT:
            for x in range(start, LIMIT + 1, p):
                counts[x] += 1
        
        # Update current_L
        x = current_L + 1
        while x <= LIMIT and counts[x] >= 2:
            current_L = x
            x += 1

    return residues
