import time

def construct():
    start_time = time.time()
    deadline = start_time + 108
    
    def build_edge_map(n):
        edge_to_idx = {}
        idx_to_edge = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx_to_edge.append((i, j))
                idx += 1
        return edge_to_idx, idx_to_edge
    
    def check_c6(adj, u, v, n):
        """adj is a set of (a,b) pairs. Check if adding (u,v) creates C6."""
        # Build adjacency sets for speed
        # But we already have adj as sets of (a,b) tuples
        # Let's use a different representation for speed
        
        # Actually, let me pass adjacency lists
        pass
    
    def solve(n):
        edge_to_idx, idx_to_edge = build_edge_map(n)
        num_edges = len(idx_to_edge)
        colors = [-1] * num_edges
        
        # adj[c][v] = set of neighbors
        adj = [dict() for _ in range(4)]
        for c in range(4):
            for v in range(n):
                adj[c][v] = set()
        
        def has_c6(c, u, v):
            """Check if adding edge (u,v) with color c creates a C6."""
            # C6: u-a-b-c-d-v-u
            # a in adj[c][u], d in adj[c][v], a!=d
            # b in adj[c][a], c2 in adj[c][b] ∩ adj[c][d]
            # all distinct
            Nu = adj[c][u]
            Nv = adj[c][v]
            if not Nu or not Nv:
                return False
            for a in Nu:
                if a == v:
                    continue
                for d in Nv:
                    if d == u or d == a:
                        continue
                    Na = adj[c][a]
                    Nd = adj[c][d]
                    for b in Na:
                        if b in (u, v, a, d):
                            continue
                        Nb = adj[c][b]
                        # c2 in Nb ∩ Nd, c2 not in {u,v,a,b,d}
                        for c2 in Nb:
                            if c2 in (u, v, a, b, d):
                                continue
                            if c2 in Nd:
                                return True
            return False
        
        # Symmetry breaking
        fixed_edges = []
        if n >= 2:
            fixed_edges.append(((0,1), 0))
        if n >= 3:
            fixed_edges.append(((0,2), 0))
        if n >= 4:
            fixed_edges.append(((0,3), 0))
        
        fixed_set = set()
        for (i,j), c in fixed_edges:
            idx = edge_to_idx[(i,j)]
            colors[idx] = c
            adj[c][i].add(j)
            adj[c][j].add(i)
            fixed_set.add((i,j))
        
        remaining = [(i,j) for (i,j) in idx_to_edge if (i,j) not in fixed_set]
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(remaining):
                return True
            
            u, v = remaining[pos]
            for c in range(4):
                if not has_c6(c, u, v):
                    idx = edge_to_idx[(u,v)]
                    colors[idx] = c
                    adj[c][u].add(v)
                    adj[c][v].add(u)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    adj[c][u].discard(v)
                    adj[c][v].discard(u)
                    colors[idx] = -1
            
            return False
        
        if backtrack(0):
            result = []
            for (i,j) in idx_to_edge:
                result.append(colors[edge_to_idx[(i,j)]])
            return result
        return None
    
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
