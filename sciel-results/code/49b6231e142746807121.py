import time
import random
import numpy as np

_DEADLINE = time.time() + 130.0  # Leave 20s safety margin

def _has_c5(n, adj_bits):
    """Check if graph contains a C5 using bitmask operations."""
    for u in range(n):
        for v in range(u + 1, n):
            if not ((adj_bits[u] >> v) & 1):
                continue
            nu = adj_bits[u] & ~(1 << v) & ~(1 << u)
            nv = adj_bits[v] & ~(1 << u) & ~(1 << v)
            temp_a = nu
            while temp_a:
                lsb_a = temp_a & -temp_a
                a = lsb_a.bit_length() - 1
                temp_a ^= lsb_a
                temp_c = nv
                while temp_c:
                    lsb_c = temp_c & -temp_c
                    c = lsb_c.bit_length() - 1
                    temp_c ^= lsb_c
                    if a == c:
                        continue
                    common = adj_bits[a] & adj_bits[c]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << c)
                    if common:
                        return True
    return False

def _find_independent_set(n, adj_bits, target_size):
    """Find an independent set of target_size, return list or None."""
    def search(remaining, current, result):
        if len(result) == target_size:
            return True
        if remaining.bit_count() < target_size - len(result):
            return False
        # Pick vertex with min degree in remaining
        min_deg = n
        best_v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = (adj_bits[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
        if best_v == -1:
            return False
        # Include best_v
        new_rem = remaining & ~(1 << best_v) & ~adj_bits[best_v]
        if search(new_rem, current + 1, result + [best_v]):
            return True
        # Exclude best_v
        new_rem = remaining & ~(1 << best_v)
        if search(new_rem, current, result):
            return True
        return False
    
    result = []
    if search((1 << n) - 1, 0, result):
        return result
    return None

def _find_c5_edge(n, adj_bits):
    """Find an edge that is part of a C5, return (u,v) or None."""
    for u in range(n):
        for v in range(u + 1, n):
            if not ((adj_bits[u] >> v) & 1):
                continue
            nu = adj_bits[u] & ~(1 << v) & ~(1 << u)
            nv = adj_bits[v] & ~(1 << u) & ~(1 << v)
            temp_a = nu
            while temp_a:
                lsb_a = temp_a & -temp_a
                a = lsb_a.bit_length() - 1
                temp_a ^= lsb_a
                temp_c = nv
                while temp_c:
                    lsb_c = temp_c & -temp_c
                    c = lsb_c.bit_length() - 1
                    temp_c ^= lsb_c
                    if a == c:
                        continue
                    common = adj_bits[a] & adj_bits[c]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << c)
                    if common:
                        return (u, v)
    return None

def construct():
    start_time = time.time()
    
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    for n in [35, 34, 33]:
        if time.time() > _DEADLINE - 30:
            break
        
        for restart in range(200):
            if time.time() > _DEADLINE - 5:
                break
            
            # Start with 8 K4s
            adj_bits = [0] * n
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj_bits[base + i] |= (1 << (base + j))
                        adj_bits[base + j] |= (1 << (base + i))
            
            # Add extra vertices (n-32)
            for v in range(32, n):
                # Connect to 2-4 random vertices
                deg = random.randint(2, 4)
                neighbors = random.sample(range(32), deg)
                for u in neighbors:
                    adj_bits[v] |= (1 << u)
                    adj_bits[u] |= (1 << v)
            
            # Local search
            for step in range(300):
                if time.time() > _DEADLINE - 3:
                    break
                
                # Check C5
                c5_edge = _find_c5_edge(n, adj_bits)
                if c5_edge:
                    u, v = c5_edge
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    continue
                
                # Check independent set
                indep_set = _find_independent_set(n, adj_bits, 9)
                if indep_set:
                    # Add edge between two vertices in the independent set
                    i, j = random.sample(indep_set, 2)
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    continue
                
                # Valid!
                if n > best_n:
                    best_n = n
                    edges = []
                    for i in range(n):
                        temp = adj_bits[i]
                        while temp:
                            lsb = temp & -temp
                            j = lsb.bit_length() - 1
                            if j > i:
                                edges.append([i, j])
                            temp ^= lsb
                    best_edges = edges
                break
            
            if best_n == n:
                break
    
    return {'n': best_n, 'edges': best_edges}
