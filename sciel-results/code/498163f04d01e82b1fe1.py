import time
import random
import heapq
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We evaluate up to a large limit to ensure we can find improvements
    MAX_EVAL = 100000
    
    # Precompute coverage structure for efficiency
    # coverage[x] = number of primes p such that x % p == r_p
    # Since MAX_EVAL is large, we cannot store a full 100k x 168 matrix easily in some contexts, 
    # but 100k * 168 * 8 bytes = 134 MB which is fine.
    # However, we can be smarter.
    
    # Strategy: 
    # 1. Start with a random or heuristic initial assignment.
    # 2. Calculate coverage for all x in 1..MAX_EVAL.
    # 3. Find the first x where coverage[x] < 2.
    # 4. If none, we are done (or hit MAX_EVAL).
    # 5. If found, we need to increase coverage of x.
    #    We can do this by changing the residue of some prime p that does NOT currently cover x.
    #    Changing p's residue to x % p will add coverage to x.
    #    However, it will REMOVE coverage from all y where y % p == old_r_p and y != x (and y not covered by new residue).
    #    Actually, it removes coverage from y where y % p == old_r_p, and adds coverage to y where y % p == new_r_p.
    #    We want to choose p that minimizes the damage to the prefix.
    #    "Damage" is the number of integers y < x (or y in the current valid prefix) that lose coverage and drop below 2.
    
    # Let's use a priority queue of "deficits" or just iterate?
    # Iterating from 1 to MAX_EVAL is O(MAX_EVAL * n_primes) which is 100k * 168 = 16.8M ops. 
    # This is very fast in Python if optimized, or even slower. 
    # But we need to update coverage efficiently.
    
    # Let's try a different approach: 
    # Since the constraint is "at least 2", and we have 168 primes, 
    # we can model this as a flow or matching problem? 
    # No, the residues are global.
    
    # Let's stick to the repair strategy but make it smarter.
    # The current program uses random SA. 
    # My new strategy: 
    # 1. Initialize residues to 0.
    # 2. Sweep x from 1 to MAX_EVAL.
    # 3. Maintain an array `coverage` of length MAX_EVAL+1.
    # 4. Maintain `coverers[x]` - list of primes covering x? No, just count.
    #    Actually, to know if changing a prime breaks something, we need to know 
    #    which primes cover each x.
    
    # Efficient data structure:
    # For each prime p, we know the positions it covers: r_p, r_p+p, r_p+2p, ...
    # When we change p from r_old to r_new, we remove coverage from positions in {r_old + k*p} 
    # and add coverage to positions in {r_new + k*p}.
    
    # Let's implement this incremental update.
    
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(MAX_EVAL + 1, dtype=np.int64)
    
    # Helper to update coverage for a prime change
    def update_prime(idx, new_res):
        p = primes[idx]
        old_res = residues[idx]
        if old_res == new_res:
            return
        
        # Remove old coverage
        for y in range(old_res, MAX_EVAL + 1, p):
            if y > 0: # x starts from 1
                coverage[y] -= 1
        
        # Add new coverage
        for y in range(new_res, MAX_EVAL + 1, p):
            if y > 0:
                coverage[y] += 1
        
        residues[idx] = new_res
    
    # Initialize coverage for all primes with residue 0
    for i, p in enumerate(primes):
        for y in range(0, MAX_EVAL + 1, p):
            if y > 0:
                coverage[y] += 1
                
    # Now, sweep x from 1 to MAX_EVAL
    # We want to find the largest L such that coverage[x] >= 2 for all x in 1..L.
    # We will fix deficits as we go.
    
    current_L = 0
    # We process x from 1 to MAX_EVAL
    # If coverage[x] < 2, we must fix it.
    # How to fix? 
    # We need to add at least (2 - coverage[x]) covers.
    # We can change residues of primes.
    
    # To make this fast, we should only look at primes that are "candidates".
    # A prime p is a candidate if it does NOT currently cover x.
    # Changing p to cover x adds 1 to coverage[x].
    
    # Heuristic for choosing which prime to change:
    # We want to minimize the number of "breaks" in the prefix [1, x-1].
    # A break occurs if some y in [1, x-1] had coverage exactly 2, 
    # and we remove a cover from it, making it 1.
    
    # Let's precompute for each prime p, the list of positions it covers? 
    # That's memory heavy. 
    # Instead, when considering changing p from r_old to r_new:
    # The positions losing coverage are {y : y % p == r_old, 1 <= y <= MAX_EVAL}.
    # The positions gaining coverage are {y : y % p == r_new, 1 <= y <= MAX_EVAL}.
    # We care about y < x (the prefix already processed).
    
    # Let's try a simpler heuristic first: 
    # Pick the prime p that covers the FEWEST positions in [1, x-1] with coverage exactly 2.
    # Or better: pick the prime p that minimizes the "loss" of critical covers.
    
    # Actually, there is a known strategy for this type of "covering" problem:
    # Greedy with backtracking or local search.
    # But let's try a "Lookahead" greedy.
    
    # To speed up the search for the best prime to change:
    # We can precompute `critical_count[p]` = number of y in [1, current_L] 
    # such that p covers y AND coverage[y] == 2.
    # If we change p, we lose these critical covers.
    # We want to minimize this loss.
    
    # However, `critical_count` changes as we move x.
    
    # Let's implement a loop that fixes the first failure.
    
    x = 1
    while x <= MAX_EVAL:
        if time.time() > deadline:
            break
            
        if coverage[x] >= 2:
            current_L = x
            x += 1
            continue
            
        # coverage[x] < 2
        # We need to increase coverage[x].
        # Let deficit = 2 - coverage[x]
        deficit = 2 - coverage[x]
        
        # Find primes that do NOT cover x
        # x % p != residues[i]
        
        # We will try to fix `deficit` units.
        # For simplicity, let's try to add one cover at a time.
        
        fixed = False
        for _ in range(deficit):
            # Find candidate primes
            # We want to pick a prime p that:
            # 1. Does not cover x.
            # 2. Minimizes the number of "critical" positions y < x that it covers.
            
            best_p_idx = -1
            best_loss = float('inf')
            
            # To make this fast, we might not want to check all 168 primes every time.
            # But 168 is small.
            
            # We need to calculate "loss" for each candidate.
            # Loss = count of y in [1, x-1] such that:
            #   y % p == residues[i]  (p currently covers y)
            #   AND coverage[y] == 2   (y is critically covered)
            
            # We can precompute a set of "critical" positions?
            # Critical positions are those with coverage == 2.
            
            # Let's maintain a list of critical positions? 
            # Or just iterate?
            # Iterating over all y < x for each candidate is O(x * n_primes).
            # Summed over x, this is O(MAX_EVAL^2 * n_primes) which is too slow (10^10).
            
            # We need a faster way to estimate loss.
            
            # Idea: 
            # The positions covered by p are arithmetic progression.
            # We can check if any of these positions are critical.
            # We can maintain an array `is_critical[y]` = (coverage[y] == 2).
            # Then for a given p and residue r, the loss is:
            # sum(is_critical[y] for y in {r, r+p, ...} if y < x)
            
            # We can precompute prefix sums of `is_critical` for each residue class modulo p?
            # That would be O(MAX_EVAL * n_primes) memory and time to build, which is 16.8M.
            # This is feasible!
            
            # Let's build `crit_prefix[p_idx][r]` = prefix sum of is_critical for positions congruent to r mod p.
            # Actually, we just need the count up to x.
            # So we can just sum the values.
            
            # But `is_critical` changes as we update coverage.
            # So we can't precompute static prefix sums.
            
            # Alternative: 
            # Just sample a few candidates? 
            # Or use a simpler heuristic: 
            # Pick the prime p with the LARGEST value.
            # Why? Larger primes have fewer multiples in [1, x].
            # So changing a large prime affects fewer positions in the prefix.
            
            # Let's try this heuristic first. It is very fast.
            
            candidates = []
            for i in range(n_primes):
                if (x - residues[i]) % primes[i] != 0:
                    # p does not cover x
                    # Heuristic: prefer larger primes
                    candidates.append((primes[i], i))
            
            # Sort by prime size descending
            candidates.sort(reverse=True)
            
            # Try the top few candidates
            found_fix = False
            for p_val, i in candidates[:10]: # Check top 10 largest primes
                new_res = x % p_val
                if new_res == residues[i]:
                    continue
                    
                # Simulate the change to see if it breaks anything
                # We need to check if any y < x loses coverage and drops below 2.
                # y loses coverage if y % p_val == residues[i]
                
                breaks = 0
                old_res = residues[i]
                
                # Iterate through positions covered by p_val with old_res
                # y = old_res + k * p_val
                # We only care about y < x
                for y in range(old_res, x, p_val):
                    if y > 0 and coverage[y] == 2:
                        breaks += 1
                        # If breaks > 0, this move is bad?
                        # Not necessarily, if we also add coverage elsewhere?
                        # But we are only adding coverage to x (and y > x).
                        # So any break in [1, x-1] is a net negative for the prefix.
                        break # Early exit if we find a break? 
                        # Actually, we want to MINIMIZE breaks. 
                        # If we find 1 break, we might still choose this if others have more.
                        # But for speed, let's just pick the one with 0 breaks if possible.
                
                if breaks == 0:
                    # This move is safe!
                    update_prime(i, new_res)
                    found_fix = True
                    break
                else:
                    # We could keep track of min_breaks, but for now, 
                    # let's just try the next candidate.
                    pass
            
            if not found_fix:
                # If no safe move found, try the one with least breaks?
                # Or just pick the largest prime blindly?
                # Let's pick the largest prime that doesn't cover x.
                if candidates:
                    i = candidates[0][1]
                    new_res = x % primes[i]
                    update_prime(i, new_res)
                else:
                    # All primes cover x? Impossible if coverage < 2 and n_primes > 2.
                    # Unless coverage is 0 or 1.
                    # If all primes cover x, then coverage[x] >= n_primes >= 2.
                    # So this case shouldn't happen if coverage[x] < 2.
                    pass
            
            # After fixing one unit, re-check coverage[x]
            # It might still be < 2 if deficit > 1
            # The loop `for _ in range(deficit)` handles this.
            
        # After attempting to fix, if coverage[x] is still < 2, 
        # we might be stuck. 
        # But we always increase coverage[x] by 1 per successful update.
        # So it should eventually reach 2.
        
        x += 1
        
    return residues.tolist()

# Test
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start}")
    print(f"Residues: {res[:10]}")
    
    # Quick check
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    def check_L(residues, primes, limit=10000):
        cov = [0]*(limit+1)
        for i, p in enumerate(primes):
            r = residues[i]
            for y in range(r, limit+1, p):
                if y > 0:
                    cov[y] += 1
        L = 0
        for x in range(1, limit+1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L

    L = check_L(res, primes, 5000)
    print(f"Score (L): {L}")
