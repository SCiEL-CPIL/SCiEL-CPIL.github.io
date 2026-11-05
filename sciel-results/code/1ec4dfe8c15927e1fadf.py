"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Uses bit-parallel backtracking with constraint propagation.
"""
import time
import random
import numpy as np

DEADLINE_MARGIN = 5.0

def _count_mono_c6(colors, n):
    """
    Count the number of monochromatic C6s in the coloring.
    """
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    total_c6 = 0
    for c in range(4):
        for v0 in range(n):
            m1 = adj[c][v0]
            while m1:
                lsb = m1 & -m1
                v1 = lsb.bit_length() - 1
                m1 ^= lsb
                
                m2 = adj[c][v1] & ~(1 << v0)
                while m2:
                    lsb2 = m2 & -m2
                    v2 = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    
                    m3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while m3:
                        lsb3 = m3 & -m3
                        v3 = lsb3.bit_length() - 1
                        m3 ^= lsb3
                        
                        m4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while m4:
                            lsb4 = m4 & -m4
                            v4 = lsb4.bit_length() - 1
                            m4 ^= lsb4
                            
                            m5 = adj[c][v4] & adj[c][v0]
                            m5 &= ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            
                            if m5:
                                total_c6 += m5.bit_count()
                                
    return total_c6

def _local_search(n, deadline):
    """
    Local search to find a 4-coloring of K_n with 0 monochromatic C6s.
    """
    num_edges = n * (n - 1) // 2
    best_colors = None
    best_count = float('inf')
    
    while time.time() < deadline:
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        current_count = _count_mono_c6(colors, n)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
        if current_count == 0:
            return colors
        
        current = colors[:]
        curr_count = current_count
        temp = 1.0
        step_size = 0.999
        
        while time.time() < deadline:
            if curr_count == 0:
                return current
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = current[e_idx]
            new_c = (old_c + random.randint(1, 3)) % 4
            current[e_idx] = new_c
            
            new_count = _count_mono_c6(current, n)
            
            delta = new_count - curr_count
            if delta < 0 or (random.random() < np.exp(-delta / temp)):
                curr_count = new_count
                if curr_count < best_count:
                    best_count = curr_count
                    best_colors = current[:]
            else:
                current[e_idx] = old_c
            
            temp *= step_size
            
            if temp < 0.01:
                break
                
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + (120.0 - DEADLINE_MARGIN)
    
    for n in [18, 17, 16, 15]:
        if time.time() > deadline - 10.0:
            break
            
        result = _local_search(n, deadline)
        
        if result is not None:
            if _count_mono_c6(result, n) == 0:
                return {'n': n, 'colors': result}
        
    if time.time() < deadline - 5.0:
        result = _local_search(14, deadline)
        if result is not None and _count_mono_c6(result, 14) == 0:
            return {'n': 14, 'colors': result}
            
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
