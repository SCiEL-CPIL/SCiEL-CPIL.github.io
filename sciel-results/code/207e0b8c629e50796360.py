import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 125  # Leave 25s buffer for safety
    
    # Helper: Check if graph is 3-colorable using bitmask backtracking
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Available colors are those not used by neighbors
            used = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        common = adj[u] & adj[v]
        return common != 0

    # Helper: Generate a random maximal triangle-free graph
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    # Helper: Local search improvement via edge swap
    def local_search(adj, edges):
        for _ in range(15):
            if len(edges) == 0:
                break
            idx = random.randint(0, len(edges)-1)
            u, v = edges[idx]
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop(idx)
            
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for u2, v2 in possible_edges:
                if not ((adj[u2] >> v2) & 1):
                    if not creates_triangle(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        edges.append([u2, v2])
                        
        return adj, edges

    # Initialize with Clebsch graph as baseline
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges_count = len(clebsch_edges)
    best_edges = [e[:] for e in clebsch_edges]
    best_adj = clebsch_adj[:]

    # Verify baseline
    if not is_3_colorable(best_adj):
        pass
    else:
        best_edges_count = 0
        best_edges = []
        best_adj = [0]*n

    # Main search loop with simulated annealing flavor
    # We alternate between random restarts and local search from Clebsch
    while time.time() < deadline:
        # 50% chance: random restart, 50% chance: perturb from best
        if random.random() < 0.5:
            adj, edges = random_maximal_tf()
        else:
            # Start from best known, remove a random edge, then re-optimize
            adj = best_adj[:]
            edges = [e[:] for e in best_edges]
            if edges:
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(idx)
                # Greedily refill
                possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(possible_edges)
                for u2, v2 in possible_edges:
                    if not ((adj[u2] >> v2) & 1):
                        if not creates_triangle(adj, u2, v2):
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            edges.append([u2, v2])
        
        # Local search
        adj, edges = local_search(adj, edges)
        
        # Check validity
        if len(edges) > best_edges_count:
            if not is_3_colorable(adj):
                best_edges_count = len(edges)
                best_edges = [e[:] for e in edges]
                best_adj = adj[:]

    return best_edges
