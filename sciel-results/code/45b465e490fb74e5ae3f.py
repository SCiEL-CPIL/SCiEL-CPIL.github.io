"""C6-free graph maximization on 40 vertices."""
import numpy as np
import time

def _has_c6(adj_bits):
    """adj_bits[i] is a 64-bit int with bit j set if i~j. Check for C6."""
    n = len(adj_bits)
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bits[u] >> v) & 1:
                continue
            common = adj_bits[u] & adj_bits[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            if common & (common - 1) == 0:  # fewer than 2 bits set
                continue
            # Iterate over pairs of set bits in common
            c = common
            prev = -1
            vals = []
            while c:
                lb = c & (-c)
                idx = lb.bit_length() - 1
                vals.append(idx)
                c ^= lb
            for i in range(len(vals)):
                a = vals[i]
                for j in range(i + 1, len(vals)):
                    b = vals[j]
                    inter = adj_bits[a] & adj_bits[b]
                    inter &= ~(1 << u)
                    inter &= ~(1 << v)
                    if inter:
                        return True
    return False

def _edge_count(adj_bits):
    return sum(bin(x).count('1') for x in adj_bits) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 40
    rng = np.random.RandomState(777)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_bits = [0] * n
    best_edges = 0
    
    def to_adj_matrix(bits):
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if (bits[i] >> j) & 1:
                    adj[i, j] = 1
        return adj
    
    def try_improve(bits, edges):
        bits = bits[:]
        shuffled = all_pairs[:]
        rng.shuffle(shuffled)
        for (u, v) in shuffled:
            if (bits[u] >> v) & 1:
                continue
            if time.time() > deadline - 5:
                break
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
            if not _has_c6(bits):
                edges += 1
            else:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
        return bits, edges
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        density = 0.10 + 0.12 * rng.random()
        bits = [0] * n
        num_edges_target = int(density * len(all_pairs))
        
        shuffled_init = all_pairs[:]
        rng.shuffle(shuffled_init)
        for k in range(num_edges_target):
            u, v = shuffled_init[k]
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
        
        edges = _edge_count(bits)
        
        if _has_c6(bits):
            current = [(i, j) for i in range(n) for j in range(i + 1, n) if (bits[i] >> j) & 1]
            rng.shuffle(current)
            for (u, v) in current:
                if not _has_c6(bits):
                    break
                if time.time() > deadline - 5:
                    break
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            edges = _edge_count(bits)
        
        bits, edges = try_improve(bits, edges)
        
        if edges > best_edges:
            best_bits = bits
            best_edges = edges
    
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_bits[i] >> j) & 1:
                edges_list.append([i, j])
    
    return edges_list
