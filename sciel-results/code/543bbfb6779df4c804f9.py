import time
from collections import defaultdict
import random


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before the 150 s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_randomized(adj, rng):
        """Find a cycle using DFS with randomized neighbor order."""
        color = {}
        parent = {}
        
        def dfs(u):
            color[u] = 1
            # Randomize neighbor order to explore different cycles
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            for w in neighbors:
                if w == parent.get(u):
                    continue
                if color.get(w, 0) == 0:
                    parent[w] = u
                    res = dfs(w)
                    if res is not None:
                        return res
                elif color.get(w, 0) == 1:
                    cyc = []
                    cur = u
                    while cur != w:
                        cyc.append(cur)
                        cur = parent[cur]
                    cyc.append(w)
                    cyc.reverse()
                    if len(cyc) >= 3:
                        return cyc
            color[u] = 2
            return None

        # Randomize start vertex
        vertices = list(adj.keys())
        rng.shuffle(vertices)
        for v in vertices:
            if color.get(v, 0) == 0:
                res = dfs(v)
                if res is not None:
                    return res
        return None

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        # Use multiple randomized restarts to find a good decomposition
        # The number of restarts can be adjusted based on time
        max_restarts = 5000 
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i) # Deterministic for reproducibility within restarts, but varied across restarts
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            # Greedy cycle packing with lookahead
            while True:
                if time.time() - start_time > DEADLINE:
                    break
                
                if not current_edges:
                    break

                # Find candidate cycles
                candidates = []
                num_candidates = 10
                for _ in range(num_candidates):
                    cycle = find_cycle_randomized(current_adj, rng)
                    if cycle is not None:
                        candidates.append(cycle)
                
                if not candidates:
                    break

                # Evaluate candidates
                best_cycle = None
                best_score = float('inf')
                
                for cyc in candidates:
                    # Simulate removal
                    temp_adj = {u: set(v) for u, v in current_adj.items()}
                    for j in range(len(cyc)):
                        a, b = cyc[j], cyc[(j + 1) % len(cyc)]
                        temp_adj[a].discard(b)
                        temp_adj[b].discard(a)
                    
                    # Score: Count edges incident to vertices of degree 1 in the remaining graph
                    # These edges cannot be part of any future cycle.
                    waste = 0
                    for u, v in current_edges:
                        # Check if u or v is in the cycle
                        # Actually, we need to check the degree in temp_adj
                        # If deg(u) == 1 in temp_adj, the edge (u, v) is "wasted" if it's not in the cycle?
                        # No, if u has degree 1 in temp_adj, the single edge connected to u in temp_adj is wasted.
                        # We need to be careful not to double count.
                        pass
                    
                    # Better score: Count the number of edges in the remaining graph 
                    # that are incident to a vertex of degree 1.
                    # An edge is wasted if at least one endpoint has degree 1 in the remaining graph?
                    # If u has degree 1, the edge (u,v) is the only one. It cannot be in a cycle.
                    # So we count the number of such edges.
                    
                    wasted_edges = set()
                    for u in temp_adj:
                        if len(temp_adj[u]) == 1:
                            v = next(iter(temp_adj[u]))
                            e = (min(u, v), max(u, v))
                            wasted_edges.add(e)
                    
                    score = len(wasted_edges)
                    
                    if score < best_score:
                        best_score = score
                        best_cycle = cyc

                if best_cycle is None:
                    break

                # Remove best_cycle edges
                for j in range(len(best_cycle)):
                    a, b = best_cycle[j], best_cycle[(j + 1) % len(best_cycle)]
                    e = (min(a, b), max(a, b))
                    current_edges.discard(e)
                    current_adj[a].discard(b)
                    current_adj[b].discard(a)
                
                pieces.append(list(best_cycle))
            
            # Add remaining edges as single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                
        if best_pieces is None:
             # Fallback: just return edges
            return [[u, v] for u, v in edges]
             
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
