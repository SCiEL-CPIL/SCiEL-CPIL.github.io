import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def has_c4_bitmask(adj_bits):
        """Check if graph with adjacency bitmasks has a C4."""
        n = len(adj_bits)
        for i in range(n):
            for j in range(i+1, n):
                common = adj_bits[i] & adj_bits[j]
                if common & (common - 1):  # more than one bit set
                    return True
        return False
    
    def min_degree_bitmask(adj_bits):
        return min(bin(b).count('1') for b in adj_bits)
    
    def try_repair_bitmask(adj_bits, n, target_min_deg):
        """
        Repair loop using bitmasks: remove edges to eliminate C4s while maintaining min degree.
        Returns True if successful.
        """
        degs = [bin(b).count('1') for b in adj_bits]
        
        max_iterations = 50000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return False
            
            # Find a violating pair (u, v) with >1 common neighbor
            found_violation = False
            for u in range(n):
                if degs[u] < target_min_deg:
                    continue
                for v in range(u+1, n):
                    if degs[v] < target_min_deg:
                        continue
                    common = adj_bits[u] & adj_bits[v]
                    if common & (common - 1):  # more than one common neighbor
                        found_violation = True
                        # Get common neighbors
                        cn = common
                        cn_list = []
                        while cn:
                            lsb = cn & (-cn)
                            a = lsb.bit_length() - 1
                            cn_list.append(a)
                            cn ^= lsb
                        
                        # Pick a common neighbor to remove
                        # Choose the one whose removal is least harmful
                        # Remove (u, a) if degs[u] > degs[v], else (v, a)
                        # But also consider degs[a]
                        best_a = None
                        best_score = float('inf')
                        for a in cn_list:
                            if degs[u] >= degs[v]:
                                # Remove (u, a)
                                score = min(degs[u] - 1, degs[a] - 1)
                            else:
                                # Remove (v, a)
                                score = min(degs[v] - 1, degs[a] - 1)
                            if score < best_score:
                                best_score = score
                                best_a = a
                        
                        a = best_a
                        if degs[u] >= degs[v]:
                            # Remove (u, a)
                            adj_bits[u] &= ~(1 << a)
                            adj_bits[a] &= ~(1 << u)
                            degs[u] -= 1
                            degs[a] -= 1
                        else:
                            # Remove (v, a)
                            adj_bits[v] &= ~(1 << a)
                            adj_bits[a] &= ~(1 << v)
                            degs[v] -= 1
                            degs[a] -= 1
                        break
                if found_violation:
                    break
            
            if not found_violation:
                # No C4s, check min degree
                if min(degs) >= target_min_deg:
                    return True
                else:
                    return False
        
        return False
    
    def try_construct_bitmask(n, p, seed):
        """
        Generate random graph using bitmasks and repair.
        """
        random.seed(seed)
        
        # Generate random graph using bitmasks
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Check min degree first
        degs = [bin(b).count('1') for b in adj_bits]
        if min(degs) < n - 39:
            return None
        
        # Repair C4s
        if try_repair_bitmask(adj_bits, n, n - 39):
            # Final verification
            if not has_c4_bitmask(adj_bits) and min_degree_bitmask(adj_bits) >= n - 39:
                return adj_bits
        return None
    
    best_n = 39
    best_edges = []
    
    for n in [47, 46]:
        target_min_deg = n - 39
        
        p_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        
        found = False
        for p in p_list:
            if found:
                break
            for seed in range(500):
                if time.time() > deadline:
                    break
                
                adj_bits = try_construct_bitmask(n, p, seed)
                if adj_bits is not None:
                    # Convert to edge list
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj_bits[u] & (1 << v):
                                edges.append([u, v])
                    
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            if time.time() > deadline:
                break
        
        if found:
            break

    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
