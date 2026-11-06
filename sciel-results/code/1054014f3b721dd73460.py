import time
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    adj = [0] * n_edges
    
    def get_edge_id(u, v):
        return edge_to_id[(min(u, v), max(u, v))]

    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = get_edge_id(v1, v2)
                e2 = get_edge_id(v3, v4)
                e3 = get_edge_id(v1, v3)
                e4 = get_edge_id(v2, v4)
                
                face_edges = [e1, e2, e3, e4]
                for a in range(4):
                    for b in range(a + 1, 4):
                        ea = face_edges[a]
                        eb = face_edges[b]
                        adj[ea] |= (1 << eb)
                        adj[eb] |= (1 << ea)

    start_time = time.time()
    timeout = 115
    
    # Initial Greedy Solution
    degs = [bin(adj[i]).count('1') for i in range(n_edges)]
    order = sorted(range(n_edges), key=lambda x: degs[x])
    
    is_set = set()
    c_mask = (1 << n_edges) - 1
    for v in order:
        if c_mask & (1 << v):
            is_set.add(v)
            c_mask &= ~(adj[v] | (1 << v))
            
    best_size = len(is_set)
    best_set = set(is_set)
    
    sys.setrecursionlimit(5000)
    
    def popcount(x):
        return bin(x).count('1')
        
    path = []
    
    def expand(C, current_size):
        nonlocal best_size, best_set, start_time, timeout, path
        
        if time.time() - start_time > timeout:
            return False
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(path[:])
            return True
            
        pc = popcount(C)
        if current_size + pc <= best_size:
            return True
            
        # Pick vertex with max degree in C
        max_deg = -1
        best_v = -1
        temp = C
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(adj[v] & C)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        v = best_v
        
        # Branch 1: Include v
        path.append(v)
        C1 = C & ~(adj[v] | (1 << v))
        if not expand(C1, current_size + 1):
            return False
        path.pop()
        
        # Branch 2: Exclude v
        C2 = C & ~(1 << v)
        if not expand(C2, current_size):
            return False
            
        return True
        
    expand((1 << n_edges) - 1, 0)
    
    if not best_set:
        best_set = is_set
        
    result = []
    for i in range(n_edges):
        if i in best_set:
            u, v = edges[i]
            result.append([u, v])
            
    return result
