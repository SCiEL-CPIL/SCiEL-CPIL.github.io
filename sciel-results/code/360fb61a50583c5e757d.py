"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _popcount(x):
    return bin(x).count('1')

def _is_triangle_free(adj_mask, n):
    for u in range(n):
        neigh_u = adj_mask[u]
        while neigh_u:
            lsb_v = neigh_u & -neigh_u
            v = lsb_v.bit_length() - 1
            if adj_mask[v] & adj_mask[u]:
                return False
            neigh_u ^= lsb_v
    return True

def _has_independent_set_10(adj_mask, n):
    target = 10
    
    def dfs(candidates, size):
        if size == target:
            return True
        if size + _popcount(candidates) < target:
            return False
        
        if not candidates:
            return False
            
        best_v = -1
        max_deg = -1
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v_idx = lsb.bit_length() - 1
            deg = _popcount(adj_mask[v_idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
            temp_cand ^= lsb
            
        if best_v == -1:
            return False
            
        new_cand = candidates & ~(adj_mask[best_v] | (1 << best_v))
        if dfs(new_cand, size + 1):
            return True
            
        if dfs(candidates & ~(1 << best_v), size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    for target_n in [39, 40]:
        if time.time() > deadline - 20.0:
            break
            
        found = False
        for seed in range(2000):
            if time.time() > deadline - 5.0:
                break
                
            random.seed(seed)
            n = target_n
            adj_mask = [0] * n
            
            for i in range(n):
                if i == 0:
                    continue
                candidates = (1 << i) - 1
                S = 0
                temp_cand = candidates
                while temp_cand:
                    bits = []
                    t = temp_cand
                    while t:
                        lsb = t & -t
                        bits.append(lsb.bit_length() - 1)
                        t ^= lsb
                    if not bits: break
                    v = random.choice(bits)
                    S |= (1 << v)
                    temp_cand &= ~(adj_mask[v] | (1 << v))
                
                adj_mask[i] = S
                m = S
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    adj_mask[v] |= (1 << i)
                    m ^= lsb
            
            if not _is_triangle_free(adj_mask, n):
                continue
                
            if _has_independent_set_10(adj_mask, n):
                continue
            
            edges = []
            for u in range(n):
                m = adj_mask[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    m ^= lsb
            return {'n': n, 'edges': edges}
            
        if not found:
            continue
            
    # Fallback: n=26 Andrásfai
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
