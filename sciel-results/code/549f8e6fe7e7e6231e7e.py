import math
import time
import random

def construct():
    """
    Find the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff 
    r1 mod gcd(m1,m2) != r2 mod gcd(m1,m2).
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_matrix[i][j] = math.gcd(moduli[i], moduli[j])
    
    def is_compatible(residues, m_idx, r):
        """Check if (r, moduli[m_idx]) is compatible with all assigned residues."""
        m = moduli[m_idx]
        for j in range(n_moduli):
            if j == m_idx or residues[j] is None:
                continue
            g = gcd_matrix[m_idx][j]
            if g > 1 and r % g == residues[j] % g:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy: assign residues in given order."""
        residues = [None] * n_moduli
        count = 0
        for m_idx in order:
            m = moduli[m_idx]
            for r in range(m):
                if is_compatible(residues, m_idx, r):
                    residues[m_idx] = r
                    count += 1
                    break
        return residues, count
    
    def local_search(residues, count):
        """Try to improve by swapping residues or adding/removing moduli."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try changing residue for an assigned modulus
            for m_idx in range(n_moduli):
                if residues[m_idx] is None:
                    continue
                old_r = residues[m_idx]
                m = moduli[m_idx]
                # Try other residues
                for r in range(m):
                    if r == old_r:
                        continue
                    residues[m_idx] = r
                    if is_compatible(residues, m_idx, r):
                        # Check if we can now add a new modulus
                        for new_idx in range(n_moduli):
                            if residues[new_idx] is not None:
                                continue
                            new_m = moduli[new_idx]
                            for nr in range(new_m):
                                if is_compatible(residues, new_idx, nr):
                                    residues[new_idx] = nr
                                    count += 1
                                    improved = True
                                    break
                            if improved:
                                break
                    if not improved:
                        residues[m_idx] = old_r
            # Try adding unassigned moduli
            for m_idx in range(n_moduli):
                if residues[m_idx] is not None:
                    continue
                m = moduli[m_idx]
                for r in range(m):
                    if is_compatible(residues, m_idx, r):
                        residues[m_idx] = r
                        count += 1
                        improved = True
                        break
        return residues, count
    
    best_residues = [None] * n_moduli
    best_count = 0
    
    # Multiple greedy passes with different orderings
    # Order 1: by modulus value
    order1 = list(range(n_moduli))
    res1, cnt1 = greedy_construct(order1)
    if cnt1 > best_count:
        best_residues = res1[:]
        best_count = cnt1
    
    # Order 2: by number of gcd>1 connections (most constrained first)
    conn_count = [0] * n_moduli
    for i in range(n_moduli):
        for j in range(n_moduli):
            if i != j and gcd_matrix[i][j] > 1:
                conn_count[i] += 1
    order2 = sorted(range(n_moduli), key=lambda i: -conn_count[i])
    res2, cnt2 = greedy_construct(order2)
    if cnt2 > best_count:
        best_residues = res2[:]
        best_count = cnt2
    
    # Order 3: by modulus value descending
    order3 = list(range(n_moduli - 1, -1, -1))
    res3, cnt3 = greedy_construct(order3)
    if cnt3 > best_count:
        best_residues = res3[:]
        best_count = cnt3
    
    # Random orders
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(n_moduli))
        random.shuffle(order)
        res, cnt = greedy_construct(order)
        if cnt > best_count:
            best_residues = res[:]
            best_count = cnt
    
    # Local search from best
    if time.time() < deadline:
        best_residues, best_count = local_search(best_residues, best_count)
    
    # Build result
    result = []
    for i in range(n_moduli):
        if best_residues[i] is not None:
            result.append([best_residues[i], moduli[i]])
    
    if not result:
        result = [[1, 2]]
    
    return result
