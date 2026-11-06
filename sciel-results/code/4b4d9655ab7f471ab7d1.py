"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(n_left, n_right, adj):
        """
        Hopcroft-Karp algorithm for maximum matching in a bipartite graph.
        adj: list of lists, adj[u] contains neighbors v in right set.
        Returns matching: list of length n_left, match[u] = v or -1.
        """
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == INF:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v

    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining_edges = set(canonical(e) for e in edges)
        
        while remaining_edges:
            if time.time() - t0 > DEADLINE - 5:
                # Fallback: add all remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Build adjacency for current remaining graph
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            if not adj:
                break
                
            # Find maximum 2-matching via bipartite matching reduction
            # Left nodes: 0..n-1, Right nodes: 0..n-1
            n = max(max(adj.keys()), default=0) + 1
            
            # Build bipartite adjacency
            # adj_bipartite[u] = list of v such that (u,v) or (v,u) is in remaining_edges
            # We need to be careful: 
            # Edge (u,v) in G corresponds to (u_L, v_R) and (v_L, u_R) in B.
            # We want to find a set of edges in G.
            # If (u_L, v_R) is matched, it means we use edge (u,v) for u's "out" and v's "in"?
            # Actually, the standard reduction is:
            # Create B with U={u1..un}, V={v1..vn}.
            # For each edge (u,v) in G, add (u, v') and (v, u') to B.
            # A matching M in B corresponds to a 2-matching in G.
            
            adj_bip = [[] for _ in range(n)]
            # Map from (u, v) in bipartite to original edge (u, v) in G
            # We need to recover the edges.
            # If (u, v) is in matching (u in U, v in V), it corresponds to edge (u, v) in G.
            # Wait, if we add (u, v') for edge (u,v), then matching (u, v') means edge (u,v) is selected.
            # Also we add (v, u') for edge (u,v). Matching (v, u') means edge (v,u) is selected.
            # Since G is undirected, (u,v) and (v,u) are the same edge.
            # So if both (u, v') and (v, u') are in the matching, we have selected edge (u,v) twice?
            # No, the matching in B is a set of disjoint edges.
            # (u, v') and (v, u') are distinct edges in B.
            # If both are in the matching, it implies u is matched to v' and v is matched to u'.
            # This corresponds to selecting the undirected edge (u,v) in the 2-matching.
            # Does it count as 1 edge or 2?
            # In the 2-matching, the edge (u,v) is present. It contributes 1 to degree of u and 1 to degree of v.
            # The bipartite matching uses 2 "slots": u's left slot and v's right slot?
            # Let's trace:
            # u_L is matched to v_R. This uses u_L and v_R.
            # v_L is matched to u_R. This uses v_L and u_R.
            # So u has degree 1 in the "left" part and 1 in the "right" part?
            # The degree of u in the 2-matching is the number of matched edges incident to u_L plus number incident to u_R?
            # No, the degree of u in the 2-matching is simply the number of edges in the 2-matching incident to u.
            # If (u_L, v_R) is matched, edge (u,v) is in 2-matching.
            # If (v_L, u_R) is matched, edge (v,u) is in 2-matching.
            # These are the same edge.
            # So if both are matched, we have selected edge (u,v).
            # Does this violate degree constraint?
            # u is incident to edge (u,v). Degree 1.
            # v is incident to edge (v,u). Degree 1.
            # This is fine.
            # What if u is also matched to w_R via (u_L, w_R)?
            # Then u is incident to (u,v) and (u,w). Degree 2.
            # This is allowed.
            
            # So, to build adj_bip:
            for u, v in remaining_edges:
                # Add (u, v) to bipartite adj
                adj_bip[u].append(v)
                # Add (v, u) to bipartite adj
                adj_bip[v].append(u)
            
            # Run Hopcroft-Karp
            pair_u, pair_v = hopcroft_karp(n, n, adj_bip)
            
            # Reconstruct 2-matching edges
            # An edge (u, v) in G is in the 2-matching if:
            # 1. pair_u[u] == v (meaning u_L matched to v_R)
            # OR
            # 2. pair_u[v] == u (meaning v_L matched to u_R)
            # Actually, if pair_u[u] == v, it means (u, v) is selected.
            # If pair_u[v] == u, it means (v, u) is selected.
            # Since the graph is undirected, we just need to check if either direction is matched.
            
            two_match_edges = set()
            for u in range(n):
                v = pair_u[u]
                if v != -1:
                    # Edge (u, v) is in the 2-matching
                    two_match_edges.add(canonical((u, v)))
            
            # Now, find cycles in the graph (V, two_match_edges)
            # We need to extract cycles.
            # The components of (V, two_match_edges) are paths and cycles.
            # We only want to take the cycles.
            
            visited = set()
            cycles_found = []
            
            for start in two_match_edges:
                if start[0] in visited or start[1] in visited:
                    continue
                # We need to traverse the component.
                # Since max degree is 2, we can just walk.
                # But we need to identify if it's a cycle or path.
                
                # Let's find the connected component containing start[0]
                comp_nodes = set()
                comp_edges = set()
                stack = [start[0]]
                visited.add(start[0])
                
                while stack:
                    node = stack.pop()
                    comp_nodes.add(node)
                    for nei in adj.get(node, []):
                        if canonical((node, nei)) in two_match_edges:
                            comp_edges.add(canonical((node, nei)))
                            if nei not in visited:
                                visited.add(nei)
                                stack.append(nei)
                
                # Check if this component is a cycle
                # A component is a cycle if every node in it has degree 2 within the component.
                is_cycle = True
                for node in comp_nodes:
                    deg = 0
                    for nei in adj.get(node, []):
                        if canonical((node, nei)) in comp_edges:
                            deg += 1
                    if deg != 2:
                        is_cycle = False
                        break
                
                if is_cycle and len(comp_edges) >= 3:
                    # Extract the cycle sequence
                    # Start from any node in comp_nodes
                    v0 = next(iter(comp_nodes))
                    cycle = [v0]
                    current = v0
                    prev = -1
                    while True:
                        nexts = [nei for nei in adj.get(current, []) if canonical((current, nei)) in comp_edges]
                        # Filter out prev
                        nexts = [nei for nei in nexts if nei != prev]
                        if not nexts:
                            break
                        nxt = nexts[0]
                        if nxt == v0 and len(cycle) >= 3:
                            break
                        cycle.append(nxt)
                        prev = current
                        current = nxt
                    
                    if len(cycle) >= 3:
                        # Verify it's a valid simple cycle
                        valid = True
                        seen = set()
                        for i in range(len(cycle)):
                            u1 = cycle[i]
                            u2 = cycle[(i+1) % len(cycle)]
                            if canonical((u1, u2)) not in remaining_edges:
                                valid = False
                                break
                            if u1 in seen:
                                valid = False
                                break
                            seen.add(u1)
                        
                        if valid:
                            cycles_found.append(cycle)
            
            if cycles_found:
                # Add all found cycles to pieces
                # Remove their edges from remaining_edges
                for cycle in cycles_found:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        e = canonical((cycle[i], cycle[(i+1) % len(cycle)]))
                        remaining_edges.discard(e)
            else:
                # No cycles found in the 2-matching.
                # This means the 2-matching consists only of paths.
                # We should pick the longest path? Or just take one edge?
                # To minimize pieces, we should try to form cycles from remaining edges.
                # But if max 2-matching has no cycles, it's hard to find cycles.
                # However, it's possible that a different set of edges forms a cycle.
                # But for now, let's just take single edges to make progress.
                # Or, we can try to find ANY cycle in the remaining graph using DFS.
                
                # Let's try a quick DFS for any cycle.
                found_cycle = None
                # Pick a random start
                if remaining_edges:
                    start_v = next(iter(remaining_edges))[0]
                    # Simple DFS
                    def dfs_cycle(u, parent, visited, path):
                        for v in adj.get(u, []):
                            if canonical((u, v)) not in remaining_edges:
                                continue
                            if v == parent:
                                continue
                            if v == start_v and len(path) >= 3:
                                return list(path)
                            if v not in visited:
                                visited.add(v)
                                path.append(v)
                                res = dfs_cycle(v, u, visited, path)
                                if res:
                                    return res
                                path.pop()
                                visited.remove(v)
                        return None
                    
                    # Try a few starts
                    for _ in range(5):
                        if not remaining_edges:
                            break
                        s = random.choice(list(remaining_edges))[0]
                        res = dfs_cycle(s, -1, {s}, [s])
                        if res:
                            found_cycle = res
                            break
                
                if found_cycle:
                    pieces.append(found_cycle)
                    for i in range(len(found_cycle)):
                        e = canonical((found_cycle[i], found_cycle[(i+1) % len(found_cycle)]))
                        remaining_edges.discard(e)
                else:
                    # No cycles at all. Take all remaining as single edges.
                    for e in remaining_edges:
                        pieces.append(list(e))
                    remaining_edges = set()
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
