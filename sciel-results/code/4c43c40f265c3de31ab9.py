import math
import time
import random
from collections import defaultdict

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Pre-compute gcds
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0
    
    # Backtracking with forward checking
    # Process moduli in order of "difficulty" - those with more constraints first
    # A modulus m has constraints with all other moduli where gcd(m, m') > 1
    
    # Count how many other moduli share a gcd > 1 with each modulus
    constraint_count = {}
    for i, m in enumerate(all_mods):
        cnt = 0
        for j, m2 in enumerate(all_mods):
            if i != j and gcd_table[(m, m2)] > 1:
                cnt += 1
        constraint_count[m] = cnt
    
    # Sort moduli by constraint count descending, then by value
    sorted_mods = sorted(all_mods, key=lambda m: (-constraint_count[m], m))
    
    def backtrack(idx, assignment, used_residues_by_mod):
        """
        Backtracking solver.
        assignment[i] = residue for sorted_mods[i], or None if unassigned
        used_residues_by_mod[m] = set of residues already used for modulus m
        """
        if time.time() > deadline:
            return None
        
        if idx == len(sorted_mods):
            return assignment[:]
        
        m = sorted_mods[idx]
        
        # Try residues in order of least constraining first
        # A residue r is less constraining if it leaves more options for future moduli
        # For simplicity, try all residues but order by how many future moduli it conflicts with
        
        # Pre-compute which future moduli this residue would conflict with
        best_r = None
        best_conflicts = None
        best_order = []
        
        for r in range(m):
            # Check against already assigned
            valid = True
            for j in range(idx):
                if assignment[j] is not None:
                    m_j = sorted_mods[j]
                    if not fast_disjoint(r, m, assignment[j], m_j):
                        valid = False
                        break
            if not valid:
                continue
            
            # Count conflicts with unassigned moduli
            conflicts = 0
            for k in range(idx + 1, len(sorted_mods)):
                m_k = sorted_mods[k]
                g = gcd_table[(m, m_k)]
                if g > 1:
                    # r % g is taken, so m_k cannot use any residue ≡ r (mod g)
                    # Number of such residues in [0, m_k) is m_k // g
                    conflicts += m_k // g
            
            best_order.append((conflicts, r))
        
        if not best_order:
            return None
        
        # Sort by conflicts ascending (least constraining first)
        best_order.sort()
        
        for _, r in best_order:
            assignment[idx] = r
            result = backtrack(idx + 1, assignment, used_residues_by_mod)
            if result is not None:
                return result
            assignment[idx] = None
        
        return None
    
    # Run backtracking with multiple restarts and different orderings
    best_result = None
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline - 2.0:
            break
        
        random.seed(seed)
        
        # Shuffle the ordering slightly for diversity
        order = sorted_mods[:]
        if seed % 3 == 0:
            # Keep original order
            pass
        elif seed % 3 == 1:
            # Reverse
            order = order[::-1]
        else:
            # Partial shuffle
            random.shuffle(order)
        
        assignment = [None] * len(order)
        result = backtrack(0, assignment, {})
        
        if result is not None:
            count = sum(1 for a in result if a is not None)
            if count > best_count:
                best_count = count
                best_result = result[:]
    
    # Also try a greedy approach as fallback
    if best_count < 30:
        for seed in range(100):
            if time.time() > deadline - 1.0:
                break
            
            random.seed(seed)
            assignment = [None] * n_mods
            order = list(range(n_mods))
            random.shuffle(order)
            
            for idx in order:
                m = all_mods[idx]
                residues_to_try = list(range(m))
                random.shuffle(residues_to_try)
                for r in residues_to_try:
                    valid = True
                    for j in range(n_mods):
                        if j == idx:
                            continue
                        if assignment[j] is not None:
                            if not fast_disjoint(r, m, assignment[j], all_mods[j]):
                                valid = False
                                break
                    if valid:
                        assignment[idx] = r
                        break
            
            count = sum(1 for a in assignment if a is not None)
            if count > best_count:
                best_count = count
                best_result = assignment[:]
    
    if best_result is None:
        return [[0, 2]]
    
    result = []
    for i in range(len(all_mods)):
        if best_result[i] is not None:
            result.append([best_result[i], all_mods[i]])
    
    return result
