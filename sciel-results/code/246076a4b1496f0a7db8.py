import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def random_adj(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[v] |= (1 << i) # Typo in thought, correct in code
        return adj

    # Correct random_adj
    def random_adj(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def get_subsets(n):
        subsets = []
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                subsets.append(mask)
        return subsets

    def is_regular(adj, mask):
        lsb = mask & -mask
        v0 = lsb.bit_length() - 1
        target_deg = (adj[v0] & mask).bit_count()
        
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg != target_deg:
                return False
            m -= lsb
        return True

    def find_violation(adj, subsets):
        for mask in subsets:
            if is_regular(adj, mask):
                return mask
        return None

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def solve_for_n(n, deadline):
        subsets = get_subsets(n)
        
        while time.time() < deadline - 5:
            adj = random_adj(n)
            
            steps = 0
            while time.time() < deadline - 5:
                viol = find_violation(adj, subsets)
                if viol is None:
                    edges = adj_to_edges(adj, n)
                    return {'n': n, 'edges': edges}
                
                # Extract vertices
                verts = []
                m = viol
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m -= lsb
                
                if len(verts) >= 2:
                    u, v = random.sample(verts, 2)
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                steps += 1
                if steps > 50000:
                    break
        
        return None

    res = solve_for_n(20, deadline)
    if res:
        return res

    res = solve_for_n(19, deadline)
    if res:
        return res

    res = solve_for_n(18, deadline)
    if res:
        return res

    for n in range(15, 4, -1):
        if time.time() > deadline - 2:
            break
        while time.time() < deadline - 2:
            adj = random_adj(n)
            subsets = get_subsets(n)
            if find_violation(adj, subsets) is None:
                edges = adj_to_edges(adj, n)
                return {'n': n, 'edges': edges}

    return {'n': 5, 'edges': []}
