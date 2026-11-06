import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 8.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _solve_repair(moduli, time_limit):
    n = len(moduli)
    if n == 0:
        return []
    
    # Precompute gcds for speed
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]

    def get_conflicts(residues):
        conflicts = []
        for i in range(n):
            ri = residues[i]
            mi = moduli[i]
            for j in range(i + 1, n):
                if (residues[j] - ri) % gcds[i][j] == 0:
                    conflicts.append((i, j))
        return conflicts

    def count_conflicts_for_index(idx, new_residue, residues):
        count = 0
        m = moduli[idx]
        for j in range(n):
            if j != idx:
                if (residues[j] - new_residue) % gcds[idx][j] == 0:
                    count += 1
        return count

    best_solution = None
    best_conflict_count = float('inf')
    
    # Multiple restarts
    while time.time() < time_limit:
        residues = [random.randint(0, m - 1) for m in moduli]
        
        for step in range(2000):
            if time.time() > time_limit:
                break
            
            conflicts = get_conflicts(residues)
            if len(conflicts) == 0:
                return [(residues[i], moduli[i]) for i in range(n)]
            
            if len(conflicts) < best_conflict_count:
                best_conflict_count = len(conflicts)
                best_solution = residues[:]
            
            i, j = random.choice(conflicts)
            
            # Min-conflict heuristic
            m_i = moduli[i]
            best_r_i = residues[i]
            min_conf_i = count_conflicts_for_index(i, best_r_i, residues)
            
            # Check a subset of residues to save time? 
            # Or all? m is small. Let's do all.
            for r in range(m_i):
                if r == residues[i]:
                    continue
                c = count_conflicts_for_index(i, r, residues)
                if c < min_conf_i:
                    min_conf_i = c
                    best_r_i = r
            
            m_j = moduli[j]
            best_r_j = residues[j]
            min_conf_j = count_conflicts_for_index(j, best_r_j, residues)
            
            for r in range(m_j):
                if r == residues[j]:
                    continue
                c = count_conflicts_for_index(j, r, residues)
                if c < min_conf_j:
                    min_conf_j = c
                    best_r_j = r
            
            if min_conf_i < min_conf_j:
                residues[i] = best_r_i
            elif min_conf_j < min_conf_i:
                residues[j] = best_r_j
            else:
                # Random kick
                if random.random() < 0.5:
                    residues[i] = random.randint(0, m_i - 1)
                else:
                    residues[j] = random.randint(0, m_j - 1)

    if best_solution:
        return [(best_solution[i], moduli[i]) for i in range(n)]
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    even_mods = [m for m in all_mods if m % 2 == 0]
    
    best_pairs = []
    
    # Use Repair algorithm
    pairs = _solve_repair(even_mods, deadline)
    if pairs is not None:
        best_pairs = pairs
        
    if not best_pairs:
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
