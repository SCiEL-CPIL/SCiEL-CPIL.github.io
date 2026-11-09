import time
import random
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _find_2factor(adj, n):
        """
        Find a 2-factor (vertex-disjoint cycles covering max vertices) using bipartite matching.
        Each edge is a left node. Each vertex has 2 slots (in/out) on the right.
        We want to assign edges to slots such that each edge uses one slot of one of its endpoints,
        and no slot is used twice. This creates a subgraph where each vertex has degree <= 2.
        Since we maximize the number of matched edges, we get a maximal set of disjoint paths/cycles.
        We then extract cycles from this subgraph.
        """
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        
        m = len(edge_list)
        if m == 0:
            return []
        
        INF = 10**9
        # Cost matrix: m x 2n
        # Row i (edge u,v) can connect to slot 2u, 2u+1, 2v, 2v+1
        cost = [[INF] * (2 * n) for _ in range(m)]
        
        for i, (u, v) in enumerate(edge_list):
            cost[i][2 * u] = 0
            cost[i][2 * u + 1] = 0
            cost[i][2 * v] = 0
            cost[i][2 * v + 1] = 0
            
        # linear_sum_assignment finds a minimum cost matching
        # Since all valid costs are 0 and invalid are INF, it maximizes the number of matches
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            return []
        
        matched_edges = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i][j] < INF:
                matched_edges.add(edge_list[i])
        
        if not matched_edges:
            return []
            
        # Build subgraph
        sub_adj = {v: set() for v in range(n)}
        for u, v in matched_edges:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
            
        # Extract cycles
        cycles = []
        visited = set()
        
        for start in range(n):
            if start in visited or len(sub_adj[start]) != 2:
                continue
                
            # Walk the cycle
            path = [start]
            visited.add(start)
            prev = -1
            curr = start
            
            while True:
                neighbors = list(sub_adj[curr])
                if not neighbors:
                    break
                    
                # Choose next node
                if prev == -1:
                    nxt = neighbors[0]
                else:
                    # Avoid going back to prev unless it's the only option (and it's start)
                    if len(neighbors) == 2:
                        if neighbors[0] == prev:
                            nxt = neighbors[1]
                        else:
                            nxt = neighbors[0]
                    else:
                        nxt = neighbors[0]
                        
                if nxt == start:
                    if len(path) >= 3:
                        cycles.append(path)
                    break
                elif nxt in visited:
                    break
                else:
                    prev = curr
                    curr = nxt
                    path.append(curr)
                    visited.add(curr)
                    
        return cycles

    def _find_cycle_dfs(adj, n, start, rng, max_depth=50):
        """Find a cycle through start using randomized DFS."""
        visited = {start}
        path = [start]
        stack = [(start, list(adj[start]))]
        
        while stack:
            node, neighbors = stack[-1]
            if not neighbors:
                stack.pop()
                path.pop()
                continue
            nxt = neighbors.pop()
            if nxt == start and len(path) >= 3:
                return list(path)
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                stack.append((nxt, list(adj[nxt])))
                if len(path) > max_depth:
                    stack.pop()
                    path.pop()
                    break
        
        return None

    def _decompose(edges, n):
        """
        Hybrid decomposition:
        1. Try to find a 2-factor to remove many long cycles at once.
        2. Use greedy DFS for remaining edges.
        3. Multiple restarts.
        """
        if not edges:
            return []
            
        best_pieces = None
        best_score = float('inf')
        
        num_restarts = 50
        
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(1000 + r)
            current_edges = set(edges)
            adj = defaultdict(set)
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            cycles = []
            
            # Phase 1: 2-factor extraction
            # Try to find a 2-factor. If successful, extract cycles and remove them.
            # We might do this multiple times if the graph is dense.
            for _ in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                if not current_edges:
                    break
                
                # Only try 2-factor if we have enough edges
                if len(current_edges) < 6:
                    break
                    
                f_cycles = _find_2factor(adj, n)
                
                if not f_cycles:
                    break
                    
                # Verify and remove
                added_any = False
                for c in f_cycles:
                    # Check if edges are still in current_edges
                    valid = True
                    for i in range(len(c)):
                        u, v = c[i], c[(i+1) % len(c)]
                        e = (min(u,v), max(u,v))
                        if e not in current_edges:
                            valid = False
                            break
                    if valid:
                        cycles.append(c)
                        # Remove edges
                        for i in range(len(c)):
                            u, v = c[i], c[(i+1) % len(c)]
                            e = (min(u,v), max(u,v))
                            current_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                        added_any = True
                if not added_any:
                    break

            # Phase 2: Greedy DFS for remaining edges
            while current_edges:
                if time.time() - start_time > DEADLINE:
                    break
                    
                # Find a cycle
                best_cycle = None
                # Try multiple starts
                vertices = [u for u in adj if len(adj[u]) >= 2]
                if not vertices:
                    break
                    
                for _ in range(20):
                    start = rng.choice(vertices)
                    c = _find_cycle_dfs(adj, n, start, rng)
                    if c and (best_cycle is None or len(c) > len(best_cycle)):
                        best_cycle = c
                        
                if best_cycle:
                    cycles.append(best_cycle)
                    for i in range(len(best_cycle)):
                        u, v = best_cycle[i], best_cycle[(i+1) % len(best_cycle)]
                        e = (min(u,v), max(u,v))
                        current_edges.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                else:
                    # No cycle found, break to avoid infinite loop
                    # Remaining edges will be single pieces
                    break

            # Calculate score
            # Pieces = cycles + remaining single edges
            score = len(cycles) + len(current_edges)
            
            if score < best_score:
                best_score = score
                best_pieces = []
                for c in cycles:
                    best_pieces.append(list(c))
                for e in current_edges:
                    best_pieces.append([e[0], e[1]])
                
                # If we covered all edges, it's a strong candidate
                # But we still keep searching for fewer cycles in case we have time
                # Note: 2-factor usually yields a good structure.

        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append(_decompose(edges, n))
        
        # Check time
        if time.time() - start_time > DEADLINE:
            # If we ran out of time, fill remaining with simple fallback
            # (though the loop above should break)
            break

    # If we didn't finish all 3, we need to return something for the missing ones
    # The task expects 3 lists.
    # If we timed out, we might have fewer results.
    # However, the loop breaks, so results might be length < 3.
    # We must ensure we return 3 lists.
    # If we timed out, we can just return the ones we have and empty lists for others?
    # No, that's invalid.
    # The deadline is 140s, evaluation is 150s.
    # The graphs are small. It should finish.
    # But to be safe:
    while len(results) < 3:
        results.append([])

    return results
