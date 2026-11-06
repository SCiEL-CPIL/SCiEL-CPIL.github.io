import math
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Greedy build: pick moduli that are non-coprime with all previously picked
    selected_idx = []
    remaining = set(range(n))
    
    # Sort by descending number of non-coprime partners (prefer highly connected)
    def non_coprime_count(i, subset):
        return sum(1 for j in subset if gcd_table[i][j] > 1)
    
    # Start with most connected
    for i in range(n):
        if time.time() > deadline - 10:
            break
        cnt = non_coprime_count(i, selected_idx)
        # Only add if non-coprime with ALL selected
        if all(gcd_table[i][j] > 1 for j in selected_idx):
            selected_idx.append(i)
            remaining.discard(i)
    
    sel_moduli = [moduli[i] for i in selected_idx]
    m = len(sel_moduli)
    
    # Assign residues: use backtracking with early termination
    residues = [0] * m
    success = True
    
    if m > 1:
        def check(idx, r):
            for j in range(idx):
                g = gcd_table[selected_idx[idx]][selected_idx[j]]
                if (r - residues[j]) % g == 0:
                    return False
            return True
        
        def bt(idx):
            if time.time() > deadline - 2:
                return False
            if idx == m:
                return True
            for r in range(sel_moduli[idx]):
                residues[idx] = r
                if check(idx, r):
                    if bt(idx+1):
                        return True
            residues[idx] = 0
            return False
        
        success = bt(1)
    
    if not success:
        # Reduce set until we find a valid one
        while m > 1:
            m -= 1
            sel_moduli = sel_moduli[:m]
            selected_idx = selected_idx[:m]
            residues = [0] * m
            if m <= 1:
                break
            def check2(idx, r):
                for j in range(idx):
                    g = gcd_table[selected_idx[idx]][selected_idx[j]]
                    if (r - residues[j]) % g == 0:
                        return False
                return True
            def bt2(idx):
                if time.time() > deadline - 1:
                    return False
                if idx == m:
                    return True
                for r in range(sel_moduli[idx]):
                    residues[idx] = r
                    if check2(idx, r):
                        if bt2(idx+1):
                            return True
                return False
            if bt2(1):
                break
    
    result = [[residues[i], sel_moduli[i]] for i in range(m)]
    return result
