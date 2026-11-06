"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Moduli we can choose from: 2..128
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * (n_moduli + 1) for _ in range(n_moduli + 1)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(selected, r_new, m_idx_new):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected."""
        m_new = moduli[m_idx_new]
        for (r_sel, m_idx_sel) in selected:
            g = gcd_table[m_idx_sel][m_idx_new]
            if (r_new - r_sel) % g == 0:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick best residue for each."""
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            # Try residues 0..m-1, pick first that's disjoint
            for r in range(m):
                if can_add(selected, r, m_idx):
                    selected.append((r, m_idx))
                    break
        return selected
    
    def local_search(selected, max_iter=5000):
        """Try to improve by swapping residues or adding/removing moduli."""
        best = selected[:]
        best_len = len(best)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new modulus
            for m_idx in range(n_moduli):
                if any(m_idx == si[1] for si in best):
                    continue
                m = moduli[m_idx]
                for r in range(m):
                    if can_add(best, r, m_idx):
                        new_best = best + [(r, m_idx)]
                        if len(new_best) > best_len:
                            best = new_best
                            best_len += 1
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try swapping a residue
            for i, (r_old, m_idx) in enumerate(best):
                m = moduli[m_idx]
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    # Check if new residue is disjoint from all others
                    ok = True
                    for j, (r_j, m_idx_j) in enumerate(best):
                        if i == j:
                            continue
                        g = gcd_table[m_idx][m_idx_j]
                        if (r_new - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        # This swap doesn't increase count, but might enable future additions
                        # Skip for now; focus on additions
                        pass
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy with moduli in ascending order
    order_asc = list(range(n_moduli))
    sol1 = greedy_construct(order_asc)
    sol1 = local_search(sol1)
    
    if time.time() > deadline:
        return [[r, moduli[m_idx]] for r, m_idx in sol1]
    
    # Strategy 2: Greedy with moduli sorted by number of prime factors (fewer = less conflict)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order_by_pf = sorted(range(n_moduli), key=lambda i: num_prime_factors(moduli[i]))
    sol2 = greedy_construct(order_by_pf)
    sol2 = local_search(sol2)
    
    if time.time() > deadline:
        best_sol = max([sol1, sol2], key=len)
        return [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Strategy 3: Greedy with moduli sorted by value (larger moduli have more residue options)
    order_desc = list(range(n_moduli - 1, -1, -1))
    sol3 = greedy_construct(order_desc)
    sol3 = local_search(sol3)
    
    if time.time() > deadline:
        best_sol = max([sol1, sol2, sol3], key=len)
        return [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Strategy 4: Try multiple random orderings
    import random
    random.seed(42)
    best_sol = max([sol1, sol2, sol3], key=len)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        random_order = list(range(n_moduli))
        random.shuffle(random_order)
        sol_r = greedy_construct(random_order)
        sol_r = local_search(sol_r, max_iter=2000)
        if len(sol_r) > len(best_sol):
            best_sol = sol_r
    
    # Final local search on the best solution
    best_sol = local_search(best_sol, max_iter=10000)
    
    # Convert to required format
    result = [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Ensure nonempty
    if not result:
        return [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END
