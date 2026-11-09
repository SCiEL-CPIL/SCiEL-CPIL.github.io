import math
import time
import random

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return True  # coprime moduli are always disjoint
    return ((r2 - r1) % g) != 0

def verify_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        if m1 < 2 or m1 > MAX_MOD:
            return False
        if r1 < 0 or r1 >= m1:
            return False
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if not is_disjoint(r1, m1, r2, m2):
                return False
    # Check distinct moduli
    moduli = [m for _, m in pairs]
    if len(set(moduli)) != len(moduli):
        return False
    return True

def count_conflicts(pairs):
    conflicts = 0
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if not is_disjoint(r1, m1, r2, m2):
                conflicts += 1
    return conflicts

def greedy_construct(moduli_list, deadline):
    """
    Greedy construction: process moduli in a specific order,
    choosing residues that are disjoint from all previously chosen.
    
    Key insight: if m1 | m2, then we need r2 != r1 (mod m1).
    So for each modulus, the number of forbidden residues depends on
    how many smaller moduli divide it.
    """
    # Sort moduli: process smaller ones first (they impose more constraints)
    # Actually, let's think differently.
    # A modulus m has constraints from all other moduli m' where gcd(m,m') > 1.
    # The tightest constraints come from divisors of m.
    
    # Strategy: sort by number of "conflicting" moduli (moduli sharing factors)
    # Process the most constrained first
    
    def conflict_degree(m):
        count = 0
        for m2 in moduli_list:
            if m2 != m and gcd(m, m2) > 1:
                count += 1
        return count
    
    sorted_moduli = sorted(moduli_list, key=lambda m: (-conflict_degree(m), m))
    
    pairs = []
    used_moduli = set()
    
    for m in sorted_moduli:
        if time.time() > deadline:
            break
        
        # Find all residues that are valid
        # A residue r is valid if for all existing (r_j, m_j), is_disjoint(r, m, r_j, m_j)
        valid_residues = []
        for r in range(m):
            ok = True
            for r_j, m_j in pairs:
                if not is_disjoint(r, m, r_j, m_j):
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        
        if valid_residues:
            # Choose randomly from valid residues
            r = random.choice(valid_residues)
            pairs.append([r, m])
            used_moduli.add(m)
    
    return pairs

def local_search_refine(pairs, moduli_set, deadline):
    """
    Refine a solution by trying to add more moduli or change residues.
    """
    n = len(pairs)
    if n == 0:
        return pairs
    
    current_pairs = [list(p) for p in pairs]
    used_moduli = set(m for _, m in current_pairs)
    available_moduli = [m for m in moduli_set if m not in used_moduli]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add a new modulus
        for m in available_moduli:
            if time.time() > deadline:
                break
            
            # Find valid residues for m
            valid_residues = []
            for r in range(m):
                ok = True
                for r_j, m_j in current_pairs:
                    if not is_disjoint(r, m, r_j, m_j):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                r = random.choice(valid_residues)
                current_pairs.append([r, m])
                used_moduli.add(m)
                available_moduli.remove(m)
                improved = True
                break
        
        # Try to change a residue to free up space for another modulus
        if not improved and available_moduli:
            for idx in range(len(current_pairs)):
                if time.time() > deadline:
                    break
                r, m = current_pairs[idx]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Check if changing this helps
                    test_pairs = [list(p) for p in current_pairs]
                    test_pairs[idx][0] = new_r
                    # Check if all valid
                    ok = True
                    for i in range(len(test_pairs)):
                        ri, mi = test_pairs[i]
                        for j in range(i + 1, len(test_pairs)):
                            rj, mj = test_pairs[j]
                            if not is_disjoint(ri, mi, rj, mj):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Try to add a modulus
                        for m2 in available_moduli:
                            valid_residues = []
                            for r2 in range(m2):
                                ok2 = True
                                for rj, mj in test_pairs:
                                    if not is_disjoint(r2, m2, rj, mj):
                                        ok2 = False
                                        break
                                if ok2:
                                    valid_residues.append(r2)
                            if valid_residues:
                                current_pairs = test_pairs
                                r_add = random.choice(valid_residues)
                                current_pairs.append([r_add, m2])
                                used_moduli.add(m2)
                                available_moduli.remove(m2)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    return current_pairs

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    best_solution = None
    best_size = 0
    
    # Multiple random restarts with greedy + local search
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        random.seed(trial * 12345 + 67890)
        
        # Greedy with random perturbation
        sol = greedy_construct(all_moduli, deadline)
        
        if sol is not None and len(sol) > 0:
            # Refine with local search
            sol = local_search_refine(sol, all_moduli, deadline)
            
            # Verify
            if verify_solution(sol):
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_solution = [list(p) for p in sol]
        
        # If we have a good solution, try to improve further
        if best_size >= 15:
            # Try more aggressive local search
            pass
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: construct a valid solution
    # Use moduli that are pairwise coprime or carefully chosen
    # Powers of 2 with r = m/2
    fallback = [[m//2, m] for m in [2, 4, 8, 16, 32, 64, 128]]
    return fallback

