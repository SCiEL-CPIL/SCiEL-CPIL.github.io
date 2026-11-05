import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute bit masks for vertices? No, dynamic.
    
    def popcount(x):
        # Fallback if bit_count not available, but likely is.
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    def is_safe_addition_mask(adj_mask, u, v):
        nv_mask = adj_mask[v]
        # List neighbors of v
        nv_list = []
        m = nv_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            m ^= lsb
            
        len_nv = len(nv_list)
        for i in range(len_nv):
            u2 = nv_list[i]
            for j in range(i + 1, len_nv):
                u3 = nv_list[j]
                common = adj_mask[u] & adj_mask[u2] & adj_mask[u3]
                if popcount(common) >= 2:
                    return False
        return True

    def add_edge_mask(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def count_edges_mask(adj_mask):
        return sum(popcount(m) for m in adj_mask) // 2

    def get_edges_list_mask(adj_mask):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    best_count = 0
    best_adj = [0] * n
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    while time.time() < deadline:
        adj_mask = [0] * n
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_safe_addition_mask(adj_mask, u, v):
                add_edge_mask(adj_mask, u, v)
        
        curr_count = count_edges_mask(adj_mask)
        if curr_count > best_count:
            best_count = curr_count
            best_adj = adj_mask[:]

    return get_edges_list_mask(best_adj)
