import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # safety margin under 90s
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build signed adjacency matrix from signs
    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def spectral_radius(signs):
        S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def is_valid_lift(signs):
        # Check connectivity of the 2-lift
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Total 2n vertices
        # Edge (u,v) with sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        total_vertices = 2 * n
        adj = [[] for _ in range(total_vertices)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # (u,0)-(v,0) and (u,1)-(v,1)
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # (u,0)-(v,1) and (u,1)-(v,0)
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from vertex 0
        visited = [False] * total_vertices
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == total_vertices
    
    def check_and_score(signs):
        if not is_valid_lift(signs):
            return float('inf')
        return spectral_radius(signs)
    
    best_signs = None
    best_obj = float('inf')
    
    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # The key insight: the spectrum of the signed prism graph depends on
    # the sign products around the cycles and matching.
    # For a prism graph with top cycle C_m (vertices 0..m-1), 
    # bottom cycle C_m (vertices m..2m-1), and matching (i, i+m),
    # the eigenvalues of the signed adjacency matrix are determined by:
    # - For each k = 0, ..., m-1: eigenvalues related to 
    #   2*cos(2*pi*k/m) and the sign products.
    #
    # More precisely, the signed adjacency matrix has a block structure.
    # The eigenvalues come in pairs: for each k, we get eigenvalues
    # from a 2x2 matrix involving the sign products.
    #
    # Let's use the fact that we can enumerate all 2^(3m) assignments for small m,
    # but that's too many for m=9 (2^27). Instead, we use the parity structure.
    #
    # Actually, let's just do a smart search: start with random assignments,
    # then use local search (flip edges) to minimize spectral radius.
    
    rng = np.random.default_rng(3)
    
    def local_search(signs, max_time=1.0):
        nonlocal best_signs, best_obj
        cur_signs = signs.copy()
        cur_obj = check_and_score(cur_signs)
        if cur_obj == float('inf'):
            return None, float('inf')
        
        update_best(cur_signs, cur_obj)
        start_time = time.time()
        
        while time.time() - start_time < max_time and time.time() < deadline - 3.0:
            improved = False
            
            # Try single flips
            best_flip = -1
            best_flip_obj = cur_obj
            for e in range(num_edges):
                old = cur_signs[e]
                cur_signs[e] = -old
                o = check_and_score(cur_signs)
                cur_signs[e] = old
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e
            
            if best_flip >= 0:
                cur_signs[best_flip] = -cur_signs[best_flip]
                cur_obj = best_flip_obj
                update_best(cur_signs, cur_obj)
                improved = True
            else:
                # Try pair flips
                found = False
                for _ in range(100):
                    e1 = rng.integers(num_edges)
                    e2 = rng.integers(num_edges)
                    if e1 == e2:
                        continue
                    cur_signs[e1] = -cur_signs[e1]
                    cur_signs[e2] = -cur_signs[e2]
                    o = check_and_score(cur_signs)
                    if o < cur_obj - 1e-12:
                        cur_obj = o
                        update_best(cur_signs, cur_obj)
                        found = True
                        break
                    else:
                        cur_signs[e1] = -cur_signs[e1]
                        cur_signs[e2] = -cur_signs[e2]
                if not found:
                    break
            
            if not improved:
                break
        
        return cur_signs, cur_obj
    
    # Do many random restarts with local search
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        signs = rng.choice([-1, 1], size=num_edges)
        cur_signs, cur_obj = local_search(signs, max_time=0.1)
        
        if cur_obj == float('inf'):
            continue
    
    if best_signs is None:
        # Fallback: try all +1
        signs = np.array([1] * num_edges, dtype=np.int8)
        obj = check_and_score(signs)
        if obj != float('inf'):
            best_signs = signs
        else:
            # Try a specific pattern
            for pattern in product([-1, 1], repeat=3):
                signs = np.ones(num_edges, dtype=np.int8)
                # Set some edges based on pattern
                for i in range(m):
                    signs[i] = pattern[0]  # top cycle edges (approximate)
                best_signs = signs
                break
    
    return best_signs.astype(np.int8)
