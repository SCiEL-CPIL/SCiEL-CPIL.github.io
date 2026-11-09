import sys
import time
import random

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    
    # We will try both ascending and descending orders for the greedy phase
    # and then apply local search.
    
    def get_primes_sorted(reverse=False):
        return sorted(primes, reverse=reverse)
    
    def greedy_solve(prime_order, L_max):
        cov = [0] * (L_max + 1)
        residues = {}
        
        for p in prime_order:
            best_r = 0
            best_gain = -1
            for r in range(p):
                gain = 0
                start = r if r > 0 else p
                for x in range(start, L_max + 1, p):
                    if cov[x] < 2:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[p] = best_r
            start = best_r if best_r > 0 else p
            for x in range(start, L_max + 1, p):
                cov[x] += 1
        return residues, cov

    def calc_L(cov, L_max):
        L = 0
        for x in range(1, L_max + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L

    def local_search(residues, cov, L_max, time_limit=100):
        start_time = time.time()
        current_L = calc_L(cov, L_max)
        primes_list = list(residues.keys())
        
        while time.time() - start_time < time_limit:
            improved = False
            # Shuffle primes to avoid local minima bias? Or just iterate.
            # Iterating in random order is better for local search.
            random.shuffle(primes_list)
            for p in primes_list:
                old_r = residues[p]
                # Update cov to remove old residue
                old_start = old_r if old_r > 0 else p
                for x in range(old_start, L_max + 1, p):
                    cov[x] -= 1
                
                best_r = old_r
                best_L = current_L
                
                # Try all other residues
                for r in range(p):
                    if r == old_r:
                        continue
                    # Apply new residue
                    new_start = r if r > 0 else p
                    for x in range(new_start, L_max + 1, p):
                        cov[x] += 1
                    
                    # Calculate new L
                    # Optimization: We can just check if the prefix extends.
                    # But calc_L is O(L_max). L_max is ~5000. 168 primes * 1000 residues * 5000 = 8.4e8. Too slow.
                    # We need a faster L calculation or a heuristic.
                    
                    # Heuristic: Check if the current gap is filled and if we can extend.
                    # Let's just do a quick check:
                    # Find the first x where cov[x] < 2.
                    new_L = 0
                    for x in range(1, L_max + 1):
                        if cov[x] >= 2:
                            new_L = x
                        else:
                            break
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    else:
                        # Roll back
                        for x in range(new_start, L_max + 1, p):
                            cov[x] -= 1
                        continue
                
                # Apply best_r
                if best_r != old_r:
                    new_start = best_r if best_r > 0 else p
                    for x in range(new_start, L_max + 1, p):
                        cov[x] += 1
                    residues[p] = best_r
                    if best_L > current_L:
                        current_L = best_L
                        improved = True
                else:
                    # Roll back to old_r (it was already removed, so re-add)
                    for x in range(old_start, L_max + 1, p):
                        cov[x] += 1
            
            if not improved:
                break
        
        return residues, current_L

    # Set L_max
    # Sum 1/p ~ log(log(1000)) ~ 2.5. 2 * Sum 1/p ~ 5.
    # Expected coverage is ~5. We need >= 2.
    # L=2000 is safe. L=5000 is safe.
    L_max = 5000
    
    best_residues = None
    best_L = 0
    
    # Try both orders
    for reverse in [False, True]:
        prime_order = get_primes_sorted(reverse)
        res, cov = greedy_solve(prime_order, L_max)
        L = calc_L(cov, L_max)
        
        # Local search
        res, L = local_search(res, cov, L_max, time_limit=50)
        
        if L > best_L:
            best_L = L
            best_residues = res
    
    # Format output
    final_residues = []
    for p in sorted(primes):
        final_residues.append(best_residues[p])
        
    return final_residues
