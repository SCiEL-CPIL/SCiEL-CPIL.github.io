import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute valid residues masks
    # valid_residues[m1][m2][r1] is bitmask of r2 in [0, m2) such that (r1, m1) and (r2, m2) are disjoint.
    valid_residues = [[None] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(2, MAX_MOD + 1):
            if m1 == m2:
                valid_residues[m1][m2] = [0] * m1
                continue
            g = gcd_table[m1][m2]
            row = [0] * m1
            for r1 in range(m1):
                mask = 0
                for r2 in range(m2):
                    if (r1 - r2) % g != 0:
                        mask |= (1 << r2)
                row[r1] = mask
            valid_residues[m1][m2] = row

    full_mask = [0] * (MAX_MOD + 1)
    for m in range(2, MAX_MOD + 1):
        full_mask[m] = (1 << m) - 1
        
    best_count = 0
    best_solution = None
    current_assignment = []
    
    def bit_count(x):
        return bin(x).count('1')
        
    def search(remaining, current_count):
        nonlocal best_count, best_solution
        
        if time.time() > deadline:
            return
            
        if current_count + len(remaining) <= best_count:
            return
            
        if not remaining:
            if current_count > best_count:
                best_count = current_count
                best_solution = [(m, r) for m, r in current_assignment]
            return
            
        min_valid_count = float('inf')
        best_idx = -1
        best_m = -1
        best_vm = 0
        
        for i, m in enumerate(remaining):
            if not current_assignment:
                count = m
                if count < min_valid_count:
                    min_valid_count = count
                    best_idx = i
                    best_m = m
                    best_vm = full_mask[m]
                continue
                
            vm = full_mask[m]
            for m_prime, r_prime in current_assignment:
                vm &= valid_residues[m_prime][m][r_prime]
                if vm == 0:
                    break
            if vm == 0:
                continue
            count = bit_count(vm)
            if count < min_valid_count:
                min_valid_count = count
                best_idx = i
                best_m = m
                best_vm = vm
                
        if best_idx == -1:
            return
            
        rest = remaining[:best_idx] + remaining[best_idx+1:]
        
        residues = []
        temp_mask = best_vm
        r = 0
        while temp_mask:
            if temp_mask & 1:
                residues.append(r)
            temp_mask >>= 1
            r += 1
            
        random.shuffle(residues)
        
        for r in residues:
            current_assignment.append((best_m, r))
            search(rest, current_count + 1)
            current_assignment.pop()
            
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random.shuffle(moduli_list)
        current_assignment = []
        search(moduli_list[:], 0)
        
    if best_solution is None:
        return [[1, 2]]
        
    return [[r, m] for m, r in best_solution]
