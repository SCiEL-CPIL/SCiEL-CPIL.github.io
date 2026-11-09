import math
import time
import random

def construct():
    """
    Find the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (a, m) and (b, n) are disjoint iff a mod gcd(m,n) != b mod gcd(m,n).
    We maximize the number of such classes.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds for all pairs
    gcd_matrix = [[0] * (n_moduli + 1) for _ in range(n_moduli + 1)]
    for i, m in enumerate(moduli):
        for j, n in enumerate(moduli):
            if i != j:
                gcd_matrix[i][j] = math.gcd(m, n)
    
    # Precompute number of divisors for each modulus
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
        
    div_counts = [count_divisors(m) for m in moduli]
    
    # Precompute max gcd with any other modulus
    max_gcds = [0] * n_moduli
    for i in range(n_moduli):
        m = moduli[i]
        for j in range(n_moduli):
            if i != j:
                g = gcd_matrix[i][j]
                if g > max_gcds[i]:
                    max_gcds[i] = g
    
    def is_compatible(selected_pairs, m_idx, r):
        """
        Check if (r, moduli[m_idx]) is compatible with all pairs in selected_pairs.
        selected_pairs is a list of (residue, modulus_idx).
        """
        m = moduli[m_idx]
        for r_p, m_p_idx in selected_pairs:
            m_p = moduli[m_p_idx]
            g = gcd_matrix[m_idx][m_p_idx]
            if (r - r_p) % g == 0:
                return False
        return True

    def greedy_pass(order):
        """
        Run a greedy pass given an order of modulus indices.
        Returns a list of (residue, modulus_idx).
        """
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            found = False
            # Try all residues for this modulus
            # Optimization: if selected is empty, any residue works, pick 0
            if not selected:
                selected.append((0, m_idx))
                continue
            
            for r in range(m):
                if is_compatible(selected, m_idx, r):
                    selected.append((r, m_idx))
                    found = True
                    break
            # If not found, skip this modulus
        return selected

    best_solution = []
    
    # Generate various orderings
    # 1. Random shuffles
    # 2. Sorted by max_gcds descending (dense first)
    # 3. Sorted by max_gcds ascending (sparse first)
    # 4. Sorted by div_counts descending
    # 5. Sorted by div_counts ascending
    # 6. Randomized versions of the sorted orders
    
    orders = []
    
    # Add specific sorted orders
    orders.append(sorted(range(n_moduli), key=lambda i: -max_gcds[i]))
    orders.append(sorted(range(n_moduli), key=lambda i: max_gcds[i]))
    orders.append(sorted(range(n_moduli), key=lambda i: -div_counts[i]))
    orders.append(sorted(range(n_moduli), key=lambda i: div_counts[i]))
    
    # Add random shuffles
    for _ in range(100):
        idx = list(range(n_moduli))
        random.shuffle(idx)
        orders.append(idx)
        
    # Add randomized sorted orders
    for _ in range(100):
        idx = list(range(n_moduli))
        random.shuffle(idx)
        # Sort by max_gcds with noise
        orders.append(sorted(idx, key=lambda i: -max_gcds[i] + random.uniform(-0.5, 0.5)))
        
    for _ in range(100):
        idx = list(range(n_moduli))
        random.shuffle(idx)
        # Sort by div_counts with noise
        orders.append(sorted(idx, key=lambda i: -div_counts[i] + random.uniform(-0.5, 0.5)))

    for order in orders:
        if time.time() > deadline:
            break
            
        solution = greedy_pass(order)
        if len(solution) > len(best_solution):
            best_solution = solution[:]
            
    # Convert best_solution (list of (residue, idx)) to list of [residue, modulus]
    result = [[r, moduli[idx]] for r, idx in best_solution]
    
    if not result:
        # Fallback: at least one class
        result = [[1, 2]]
        
    return result
