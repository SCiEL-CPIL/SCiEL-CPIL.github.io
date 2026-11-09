import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
    
    def check_compatible(assigned, idx, r):
        """Check if residue r for modulus idx is compatible with all assigned moduli."""
        for j, r_j in assigned.items():
            if j == idx:
                continue
            g = gcds[idx][j]
            if (r - r_j) % g == 0:
                return False
        return True
    
    def greedy(order):
        """Greedy assignment in given order."""
        assigned = {}
        for idx in order:
            m = moduli[idx]
            for r in range(m):
                if check_compatible(assigned, idx, r):
                    assigned[idx] = r
                    break
        return assigned
    
    best_assigned = None
    best_count = 0
    
    # Strategy 1: Increasing order
    order1 = list(range(n))
    a1 = greedy(order1)
    if len(a1) > best_count:
        best_count = len(a1)
        best_assigned = a1
    
    # Strategy 2: Decreasing order
    order2 = list(range(n-1, -1, -1))
    a2 = greedy(order2)
    if len(a2) > best_count:
        best_count = len(a2)
        best_assigned = a2
    
    # Strategy 3: Randomized orders
    for seed in range(300):
        if time.time() > deadline:
            break
        rng = random.Random(seed * 7919 + 13)
        order3 = list(range(n))
        rng.shuffle(order3)
        a3 = greedy(order3)
        if len(a3) > best_count:
            best_count = len(a3)
            best_assigned = a3
    
    # Strategy 4: Sort by number of prime factors (fewer first)
    def count_prime_factors(n_val):
        count = 0
        d = 2
        while d * d <= n_val:
            while n_val % d == 0:
                count += 1
                n_val //= d
            d += 1
        if n_val > 1:
            count += 1
        return count
    
    pf_counts = [count_prime_factors(moduli[i]) for i in range(n)]
    order4 = sorted(range(n), key=lambda i: pf_counts[i])
    a4 = greedy(order4)
    if len(a4) > best_count:
        best_count = len(a4)
        best_assigned = a4
    
    # Strategy 5: Try to add missing moduli to best solution
    if best_assigned is not None:
        missing = [i for i in range(n) if i not in best_assigned]
        # Sort missing by how many constraints they have (fewer first)
        missing.sort(key=lambda i: len(best_assigned))
        for i in missing:
            if time.time() > deadline:
                break
            m = moduli[i]
            for r in range(m):
                if check_compatible(best_assigned, i, r):
                    best_assigned[i] = r
                    best_count += 1
                    break
    
    # Strategy 6: Local search - try to fix conflicts by changing residues
    if best_assigned is not None and best_count < n:
        # Count current conflicts among assigned
        def count_conflicts(assigned_dict):
            conflicts = 0
            keys = list(assigned_dict.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    idx_i, idx_j = keys[i], keys[j]
                    g = gcds[idx_i][idx_j]
                    if (assigned_dict[idx_i] - assigned_dict[idx_j]) % g == 0:
                        conflicts += 1
            return conflicts
        
        # Try to add missing moduli with local search on residues
        missing = [i for i in range(n) if i not in best_assigned]
        for i in missing:
            if time.time() > deadline:
                break
            m = moduli[i]
            # Try all residues
            best_r = None
            best_new_conflicts = None
            for r in range(m):
                # Temporarily assign
                test_assigned = dict(best_assigned)
                test_assigned[i] = r
                nc = count_conflicts(test_assigned)
                if best_new_conflicts is None or nc < best_new_conflicts:
                    best_new_conflicts = nc
                    best_r = r
            if best_r is not None:
                # Check if adding i creates no new conflicts with existing
                if check_compatible(best_assigned, i, best_r):
                    best_assigned[i] = best_r
                    best_count += 1
    
    if best_count == 0:
        return [[0, 2]]
    
    return [[best_assigned[i], moduli[i]] for i in sorted(best_assigned.keys())]
