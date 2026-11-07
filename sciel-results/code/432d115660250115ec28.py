import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Maximum closure via max-flow (using a simple BFS-based approach for small graphs)
    def max_closure(nodes, weights, edges_closure):
        """
        Find maximum weight closure in a directed graph.
        nodes: list of node ids
        weights: dict node -> weight (positive or negative)
        edges_closure: list of (u, v) meaning if u is in closure, v must be in closure
        
        Returns: set of nodes in the maximum closure, and its total weight.
        Uses min-cut formulation: 
          - Source connects to positive-weight nodes with capacity = weight
          - Negative-weight nodes connect to sink with capacity = -weight
          - Closure edges have infinite capacity
          Max closure weight = sum(positive weights) - min_cut
        """
        if not nodes:
            return set(), 0
        
        # Build flow network
        # Node indices: 0=source, 1..len(nodes)=nodes, len(nodes)+1=sink
        node_index = {node: i+1 for i, node in enumerate(nodes)}
        source = 0
        sink = len(nodes) + 1
        total_nodes = len(nodes) + 2
        
        INF = float('inf')
        # capacity[u][v]
        capacity = [[0.0] * total_nodes for _ in range(total_nodes)]
        
        pos_sum = 0.0
        for node in nodes:
            w = weights.get(node, 0.0)
            if w > 0:
                capacity[source][node_index[node]] = w
                pos_sum += w
            elif w < 0:
                capacity[node_index[node]][sink] = -w
        
        for u, v in edges_closure:
            if u in node_index and v in node_index:
                capacity[node_index[u]][node_index[v]] = INF
        
        # Ford-Fulkerson with BFS (Edmonds-Karp)
        from collections import deque
        
        def bfs_find_path():
            parent = [-1] * total_nodes
            parent[source] = source
            q = deque([source])
            while q:
                u = q.popleft()
                for v in range(total_nodes):
                    if parent[v] == -1 and capacity[u][v] > 0:
                        parent[v] = u
                        if v == sink:
                            return parent
                        q.append(v)
            return None
        
        max_flow = 0.0
        while True:
            parent = bfs_find_path()
            if parent is None:
                break
            # Find bottleneck
            path_flow = INF
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u
            # Update residual
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = u
            max_flow += path_flow
        
        # Extract closure: nodes reachable from source in residual
        visited = [False] * total_nodes
        q = deque([source])
        visited[source] = True
        while q:
            u = q.popleft()
            for v in range(total_nodes):
                if not visited[v] and capacity[u][v] > 0:
                    visited[v] = True
                    q.append(v)
        
        closure = set()
        for node in nodes:
            if visited[node_index[node]]:
                closure.add(node)
        
        closure_weight = pos_sum - max_flow
        return closure, closure_weight

    def find_weighted_biclique(remaining_edges, edge_weights):
        """
        Find a high-weight biclique in the residual graph using maximum closure.
        
        For a candidate biclique (L, R):
        - All cross pairs (l,r) for l in L, r in R must be in remaining_edges
        - Weight = sum of edge_weights[(l,r)] for l in L, r in R
        - But we also need to account for the constraint that L and R are disjoint
          and all cross edges exist.
        
        Approach: For each vertex as a "seed", build a closure problem where
        selecting vertices for L and R must satisfy the biclique constraint.
        
        Simplified: Try each vertex as anchor in L, then find best R, then expand L.
        """
        best_L = set()
        best_R = set()
        best_weight = 0
        
        for anchor in range(n):
            if time.time() > deadline:
                break
            
            # Start with L={anchor}, find best R
            # R candidates: neighbors of anchor in remaining
            R_candidates = set()
            for v in adj[anchor]:
                if (min(anchor, v), max(anchor, v)) in remaining_edges:
                    R_candidates.add(v)
            
            if not R_candidates:
                continue
            
            # For each candidate in R, compute its "value" = sum of weights to L
            # and check constraints
            # Use closure: select subset of R_candidates such that all selected
            # connect to anchor (they do by construction)
            
            # Now try to expand L: vertices that connect to all of R
            # Build closure problem:
            # Nodes: potential L vertices (excluding anchor which is already in L)
            # and potential R vertices (R_candidates)
            # If we select a vertex for L, it must connect to all selected R vertices
            # If we select a vertex for R, it must connect to anchor (already true)
            
            # Simpler greedy: iteratively add best vertex to L or R
            L = {anchor}
            R = set(R_candidates)
            
            # Compute initial weight
            w = 0.0
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    if e in edge_weights:
                        w += edge_weights[e]
            
            if w > best_weight:
                best_weight = w
                best_L = set(L)
                best_R = set(R)
            
            # Try to expand
            changed = True
            while changed and time.time() < deadline:
                changed = False
                
                # Try adding to L
                for v in range(n):
                    if v in L or v in R:
                        continue
                    # v can join L if it connects to all in R
                    can_add = all((min(v, r), max(v, r)) in remaining_edges for r in R)
                    if can_add:
                        # Check if adding improves weight
                        add_w = sum(edge_weights.get((min(v, r), max(v, r)), 0.0) for r in R)
                        if add_w > 0:
                            L.add(v)
                            w += add_w
                            changed = True
                            if w > best_weight:
                                best_weight = w
                                best_L = set(L)
                                best_R = set(R)
                
                # Try adding to R
                for v in range(n):
                    if v in L or v in R:
                        continue
                    can_add = all((min(v, l), max(v, l)) in remaining_edges for l in L)
                    if can_add:
                        add_w = sum(edge_weights.get((min(v, l), max(v, l)), 0.0) for l in L)
                        if add_w > 0:
                            R.add(v)
                            w += add_w
                            changed = True
                            if w > best_weight:
                                best_weight = w
                                best_L = set(L)
                                best_R = set(R)
        
        return best_L, best_R, best_weight

    def column_generation_partition():
        """
        Column generation: iteratively find high-value bicliques in residual.
        """
        remaining = set(edge_set)
        bicliques = []
        
        # Initialize edge weights: 1.0 for all edges
        edge_weights = {e: 1.0 for e in edge_set}
        
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not remaining:
                break
            
            # Find best weighted biclique in residual
            L, R, w = find_weighted_biclique(remaining, edge_weights)
            
            if not L or not R:
                # Fallback: pick a single edge
                e = next(iter(remaining))
                L = {e[0]}
                R = {e[1]}
            
            # Validate: all cross pairs must be in remaining
            valid_L = set()
            for l in L:
                if all((min(l, r), max(l, r)) in remaining for r in R):
                    valid_L.add(l)
            
            if not valid_L:
                e = next(iter(remaining))
                valid_L = {e[0]}
                R = {e[1]}
            
            # Remove covered edges
            for l in valid_L:
                for r in R:
                    remaining.discard((min(l, r), max(l, r)))
            
            bicliques.append((list(valid_L), list(R)))
            
            # Update weights: increase weight of uncovered edges
            for e in remaining:
                edge_weights[e] += 1.0
        
        # If there are still remaining edges, add them as single edges
        while remaining:
            if time.time() > deadline:
                break
            e = next(iter(remaining))
            remaining.discard(e)
            bicliques.append([e[0], [e[1]]])
        
        return bicliques

    def greedy_partition():
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_biclique = None
            best_size = 0
            
            for u in range(n):
                if time.time() > deadline:
                    break
                u_remaining_neighbors = set()
                for v in adj[u]:
                    if (min(u,v), max(u,v)) in remaining:
                        u_remaining_neighbors.add(v)
                
                if not u_remaining_neighbors:
                    continue
                
                L = {u}
                R = set(u_remaining_neighbors)
                
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,r), max(v,r)) in remaining for r in R):
                                new_L.add(v)
                    if new_L:
                        L = L | new_L
                        changed = True
                    
                    new_R = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,l), max(v,l)) in remaining for l in L):
                                new_R.add(v)
                    if new_R:
                        R = R | new_R
                        changed = True
                
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            if best_biclique is None:
                e = next(iter(remaining))
                best_biclique = ({e[0]}, {e[1]})
            
            L, R = best_biclique
            valid_L = set()
            for l in L:
                if all((min(l,r), max(l,r)) in remaining for r in R):
                    valid_L.add(l)
            
            if not valid_L:
                e = next(iter(remaining))
                valid_L = {e[0]}
                R = {e[1]}
            
            for l in valid_L:
                for r in R:
                    remaining.discard((min(l,r), max(l,r)))
            
            bicliques.append((list(valid_L), list(R)))
        
        return bicliques

    # Try column generation first, fall back to greedy
    best_solution = None
    best_count = float('inf')
    
    try:
        cg_result = column_generation_partition()
        if cg_result and len(cg_result) < best_count:
            best_count = len(cg_result)
            best_solution = cg_result
    except Exception:
        pass
    
    try:
        greedy_result = greedy_partition()
        if greedy_result and len(greedy_result) < best_count:
            best_count = len(greedy_result)
            best_solution = greedy_result
    except Exception:
        pass
    
    # Try randomized greedy for additional diversity
    import random
    random.seed(42)
    
    for attempt in range(15):
        if time.time() > deadline - 3:
            break
        
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 1:
                break
            
            best_biclique = None
            best_size = 0
            
            vertices = list(range(n))
            random.shuffle(vertices)
            
            for u in vertices:
                if time.time() > deadline - 1:
                    break
                u_remaining_neighbors = set()
                for v in adj[u]:
                    if (min(u,v), max(u,v)) in remaining:
                        u_remaining_neighbors.add(v)
                
                if not u_remaining_neighbors:
                    continue
                
                L = {u}
                R = set(u_remaining_neighbors)
                
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,r), max(v,r)) in remaining for r in R):
                                new_L.add(v)
                    if new_L:
                        L = L | new_L
                        changed = True
                    
                    new_R = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,l), max(v,l)) in remaining for l in L):
                                new_R.add(v)
                    if new_R:
                        R = R | new_R
                        changed = True
                
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            if best_biclique is None:
                e = next(iter(remaining))
                best_biclique = ({e[0]}, {e[1]})
            
            L, R = best_biclique
            valid_L = set()
            for l in L:
                if all((min(l,r), max(l,r)) in remaining for r in R):
                    valid_L.add(l)
            
            if not valid_L:
                e = next(iter(remaining))
                valid_L = {e[0]}
                R = {e[1]}
            
            for l in valid_L:
                for r in R:
                    remaining.discard((min(l,r), max(l,r)))
            
            bicliques.append((list(valid_L), list(R)))
        
        if bicliques and len(bicliques) < best_count:
            best_count = len(bicliques)
            best_solution = bicliques
        
        if best_count <= 3:
            break
    
    if best_solution is None:
        best_solution = greedy_partition()
    
    # Format result
    result = []
    for L, R in best_solution:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    return result
