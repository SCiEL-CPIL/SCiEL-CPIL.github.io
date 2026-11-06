import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Build Conflict Graph
    # conflicts[e] is a set of edge indices that conflict with e
    conflicts = [set() for _ in range(n_edges)]
    
    # For each edge, determine its conflicts
    # Edge e is (u, v) where v = u ^ (1 << i). u < v.
    # i is the dimension.
    # Conflicts are in faces.
    # Face is defined by dim i and dim j.
    # The 4 edges in the face are:
    # 1. (u, u^2^i)  -> e
    # 2. (u^2^j, u^2^i^2^j) -> parallel in dim i
    # 3. (u, u^2^j) -> cross in dim j
    # 4. (u^2^i, u^2^i^2^j) -> cross in dim j
    
    for e_idx in range(n_edges):
        u, v = idx_to_edge[e_idx]
        # Determine dimension i
        diff = u ^ v
        # i is the bit position of diff
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            
            # Parallel edge in dim i, offset by 2^j
            p1 = u ^ (1 << j)
            p2 = v ^ (1 << j)
            pe = (min(p1, p2), max(p1, p2))
            pe_idx = edge_to_idx[pe]
            conflicts[e_idx].add(pe_idx)
            
            # Cross edge 1 in dim j, starting at u
            c1 = u
            c2 = u ^ (1 << j)
            ce1 = (min(c1, c2), max(c1, c2))
            ce1_idx = edge_to_idx[ce1]
            conflicts[e_idx].add(ce1_idx)
            
            # Cross edge 2 in dim j, starting at v
            c3 = v
            c4 = v ^ (1 << j)
            ce2 = (min(c3, c4), max(c3, c4))
            ce2_idx = edge_to_idx[ce2]
            conflicts[e_idx].add(ce2_idx)

    # 3. Heuristic Search
    
    def saturate(selected):
        improved = True
        while improved:
            improved = False
            for e in range(n_edges):
                if e not in selected:
                    # Check if e conflicts with any in selected
                    if not (selected & conflicts[e]):
                        selected.add(e)
                        improved = True
        return selected

    def greedy_tomita():
        selected = set()
        remaining = set(range(n_edges))
        
        while remaining:
            min_deg = float('inf')
            best_v = -1
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            selected.add(best_v)
            to_remove = {best_v} | (conflicts[best_v] & remaining)
            remaining -= to_remove
            
        return selected

    def greedy_random(seed):
        rng = random.Random(seed)
        selected = set()
        remaining = list(range(n_edges))
        rng.shuffle(remaining)
        
        for v in remaining:
            if v not in selected:
                if not (selected & conflicts[v]):
                    selected.add(v)
        
        return selected

    best_set = set()
    best_size = 0
    
    # Run Tomita
    S = greedy_tomita()
    S = saturate(S)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Run Random Greedy multiple times
    # 1000 iterations is fast
    for k in range(1000):
        S = greedy_random(k)
        S = saturate(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Construct result
    result = []
    for e_idx in best_set:
        u, v = idx_to_edge[e_idx]
        result.append([u, v])
        
    return result
