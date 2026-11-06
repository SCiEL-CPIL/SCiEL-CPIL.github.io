import math
import time
import sys

def construct():
    moduli = list(range(20, 201))
    # Sort by descending weight (ascending m) to improve pruning in B&B
    moduli.sort(key=lambda x: 1.0/x, reverse=True)
    n = len(moduli)
    
    # Precompute gcds
    gcd_cache = {}
    for i in range(n):
        for j in range(i):
            gcd_cache[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    # Suffix sums for bounding
    suffix_sums = [0.0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + 1.0/moduli[i]
        
    # Greedy initialization to get a good starting bound
    def greedy_init():
        selected = []
        res_map = {}
        for m in moduli:
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in selected:
                    g = gcd_cache.get((m, m2), math.gcd(m, m2))
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append((m, r))
                    res_map[m] = r
                    found = True
                    break
        return selected, res_map

    init_sol, init_res = greedy_init()
    best_sum = sum(1.0/m for m, _ in init_sol)
    best_sol = list(init_sol)
    
    start_time = time.time()
    time_limit = 115.0
    
    sys.setrecursionlimit(10000)
    
    def is_valid_residue(m_new, r_new, current_mods, current_res):
        for m_old in current_mods:
            g = gcd_cache.get((m_new, m_old), math.gcd(m_new, m_old))
            if (r_new - current_res[m_old]) % g == 0:
                return False
        return True
        
    def solve(idx, current_mods, current_res, current_sum):
        nonlocal best_sum, best_sol
        
        if time.time() - start_time > time_limit:
            return
            
        # Bound
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_sol = [(m, current_res[m]) for m in current_mods]
            return
            
        m = moduli[idx]
        
        # Option 1: Exclude m
        solve(idx + 1, current_mods, current_res, current_sum)
        
        # Option 2: Include m
        # Try to find a valid residue
        # To improve quality, we might try multiple residues, but for speed we try the first one found.
        # However, different residues have different "future compatibility".
        # We'll try the first valid one.
        for r in range(m):
            if is_valid_residue(m, r, current_mods, current_res):
                current_mods.append(m)
                current_res[m] = r
                solve(idx + 1, current_mods, current_res, current_sum + 1.0/m)
                current_mods.pop()
                del current_res[m]
                break # Only try one residue per modulus in this branch to keep it fast
                
    solve(0, [], {}, 0.0)
    
    # If B&B didn't improve, return the best found (which includes the greedy init)
    return [[m, r] for m, r in best_sol]
