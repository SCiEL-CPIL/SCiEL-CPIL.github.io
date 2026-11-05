import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    def is_regular_subset(adj_bits, subset_mask):
        k = bin(subset_mask).count('1')
        if k < 6:
            return False
        first_deg = -1
        mask = subset_mask
        while mask:
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            deg = bin(adj_bits[v] & subset_mask).count('1')
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            mask ^= lsb
        return True
    
    def find_violation(adj_bits, n):
        # Check sizes 6 to n
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_subset(adj_bits, mask):
                    return mask, k
        return None
    
    def random_graph(n, p=0.5):
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        return adj_bits
    
    def flip_edge(adj_bits, i, j):
        if adj_bits[i] & (1 << j):
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
        else:
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
    
    best_n = 5
    best_adj = None
    
    for target_n in range(15, 23):
        if time.time() > deadline - 5:
            break
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            adj = random_graph(target_n, p=0.5)
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                viol = find_violation(adj, target_n)
                if viol is None:
                    if target_n > best_n:
                        best_n = target_n
                        best_adj = adj[:]
                    found = True
                    break
                mask, k = viol
                verts = []
                m = mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    i, j = random.sample(verts, 2)
                    flip_edge(adj, i, j)
            if found:
                break
    
    if best_adj is not None and best_n > 5:
        n = best_n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}
