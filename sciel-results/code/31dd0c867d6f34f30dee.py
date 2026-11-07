"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Precompute remaining edge count per vertex
    rem_count = [len(adj[i]) for i in range(n)]
    
    # Greedy biclique partition with multiple strategies
    best_bicliques = None
    best_count = len(edges) + 1  # worst case: one biclique per edge
    
    deadline = time.time() + 80  # 80 second deadline to be safe
    
    # Strategy 1: Greedy by max edges covered
    # We'll try multiple random orderings and keep the best
    
    def greedy_partition(order_seed=None, max_iters=5000):
        """Greedy biclique partition trying to maximize edges per biclique."""
        if order_seed is not None:
            random.seed(order_seed)
        
        # Track remaining edges
        remaining = set(edge_set)
        bicliques = []
        
        # Precompute neighbors for each vertex
        # We'll use a greedy approach: at each step, find the best biclique
        # that covers the most remaining edges
        
        # For efficiency, we'll limit the search
        # Try to find bicliques by picking pairs and expanding
        
        # Build a list of vertices sorted by remaining degree
        for _ in range(max_iters):
            if not remaining:
                break
            if time.time() > deadline:
                break
            
            # Find the vertex with max remaining degree
            best_v = -1
            best_deg = -1
            for v in range(n):
                deg = len([e for e in remaining if v in e])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1 or best_deg == 0:
                break
            
            # Try to find the best biclique containing best_v
            # A biclique is defined by (L, R) where all cross edges exist
            # and all cross edges are in remaining
            
            # Strategy: pick a vertex u on the other side, then find
            # the common neighbors that form a valid biclique
            
            best_biclique = None
            best_edges_covered = 0
            
            # Try each potential partner
            for u in range(n):
                if u == best_v:
                    continue
                if (min(u, best_v), max(u, best_v)) not in remaining:
                    continue
                
                # Find common neighbors w such that (best_v, w) and (u, w) are both in remaining
                common = []
                for w in range(n):
                    if w == best_v or w == u:
                        continue
                    e1 = (min(best_v, w), max(best_v, w))
                    e2 = (min(u, w), max(u, w))
                    if e1 in remaining and e2 in remaining:
                        common.append(w)
                
                # The biclique would be L={best_v}, R={u}+common
                # But we need ALL cross edges to be in remaining
                # Cross edges: (best_v, u), (best_v, w) for w in common
                # That's 1 + len(common) edges
                
                # We could also try larger bicliques
                # Let's try: L = {best_v} + some subset of common, R = {u} + some subset
                # For now, keep it simple: L={best_v}, R={u}+common
                
                edges_in_biclique = 1 + len(common)
                
                if edges_in_biclique > best_edges_covered:
                    best_edges_covered = edges_in_biclique
                    best_biclique = ([best_v], [u] + common)
            
            if best_biclique is None:
                # Just pick one edge
                e = next(iter(remaining))
                best_biclique = ([e[0]], [e[1]])
                best_edges_covered = 1
            
            # Add this biclique and remove its edges
            L, R = best_biclique
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    remaining.discard(e)
            
            bicliques.append([list(L), list(R)])
        
        # Handle any remaining edges
        for e in remaining:
            bicliques.append([[e[0]], [e[1]]])
        
        return bicliques
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        bicliques = greedy_partition(order_seed=seed)
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    # Strategy 2: More aggressive greedy - try larger bicliques
    def greedy_partition_v2(max_iters=2000):
        """More aggressive greedy that tries to build larger bicliques."""
        remaining = set(edge_set)
        bicliques = []
        
        for _ in range(max_iters):
            if not remaining:
                break
            if time.time() > deadline:
                break
            
            # Find the pair with maximum common neighbors in remaining
            best_pair = None
            best_common = 0
            best_biclique = None
            
            # Sample some pairs for efficiency
            edge_list = list(remaining)
            # Shuffle and try a subset
            random.shuffle(edge_list)
            for e in edge_list[:100]:
                u, v = e
                # Find w such that (u,w) and (v,w) are in remaining
                common = []
                for w in range(n):
                    if w == u or w == v:
                        continue
                    e1 = (min(u, w), max(u, w))
                    e2 = (min(v, w), max(v, w))
                    if e1 in remaining and e2 in remaining:
                        common.append(w)
                
                if len(common) > best_common:
                    best_common = len(common)
                    best_pair = (u, v)
                    # Biclique: L={u}, R={v}+common
                    best_biclique = ([u], [v] + common)
            
            if best_biclique is None:
                e = next(iter(remaining))
                best_biclique = ([e[0]], [e[1]])
            
            L, R = best_biclique
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    remaining.discard(e)
            
            bicliques.append([list(L), list(R)])
        
        for e in remaining:
            bicliques.append([[e[0]], [e[1]]])
        
        return bicliques
    
    for seed in range(10):
        if time.time() > deadline:
            break
        random.seed(seed)
        bicliques = greedy_partition_v2()
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        # Fallback: one biclique per edge
        best_bicliques = [[[u], [v]] for u, v in edges]
    
    # Ensure proper types
    result = []
    for bic in best_bicliques:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
