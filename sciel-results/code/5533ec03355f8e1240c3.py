"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def max_bipartite_matching(n, adj):
        """
        Finds maximum bipartite matching using Kuhn's algorithm (DFS augmenting paths).
        adj: list of lists, adj[u] contains v such that (u, v) is an edge in bipartite graph.
        n: number of nodes on each side.
        Returns: match_right, where match_right[v] = u if v is matched to u, else -1.
        """
        match_right = [-1] * n
        
        def try_kuhn(u, visited):
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    if match_right[v] == -1 or try_kuhn(match_right[v], visited):
                        match_right[v] = u
                        return True
            return False

        for u in range(n):
            visited = [False] * n
            try_kuhn(u, visited)
            
        return match_right

    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining_edges = set(edges)
        n = max(max(e) for e in edges) + 1 if edges else 0
        
        while True:
            if not remaining_edges:
                break
                
            # Build bipartite graph adjacency
            # Left side: 0..n-1, Right side: 0..n-1
            adj = [[] for _ in range(n)]
            for u, v in remaining_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Find max matching
            match_right = max_bipartite_matching(n, adj)
            
            # Extract cycles from matching
            # match_right[v] = u means edge (u, v) in bipartite graph, which is (u, v) in original
            # We need to follow the chain: u -> v -> match_right[v] ... wait.
            # If match_right[v] = u, it means u is matched to v.
            # In the original graph, this corresponds to traversing edge (u, v).
            # A cycle in the bipartite graph corresponds to a cycle in the original graph.
            # Let's trace the cycles.
            
            visited_nodes = [False] * n # For left nodes? No, for vertices in original graph.
            # Actually, the matching defines a permutation-like structure on the vertices involved.
            # If u is matched to v (match_right[v] == u), and v is matched to w (match_right[w] == v),
            # then we have a path u -> v -> w in the original graph?
            # No, match_right[v] = u means the edge (u, v) is in the matching.
            # So if we have a sequence of matches:
            # u1 is matched to v1 (match_right[v1] = u1)
            # u2 is matched to v2 (match_right[v2] = u2)
            # ...
            # To form a cycle, we need u_i = v_{i-1}.
            # So if match_right[v] = u, it implies an arc u -> v in the cycle.
            
            cycles = []
            visited = [False] * n
            
            for i in range(n):
                if not visited[i] and match_right[i] != -1:
                    # Start a cycle from i
                    # Note: match_right[i] = u means u is matched to i. So arc is u -> i.
                    # We need to find the start of the cycle.
                    # Let's just follow the pointers.
                    # If match_right[v] = u, then u -> v.
                    # We need to find a node that is not the target of any match? No, it's a permutation of a subset.
                    # Let's trace from i.
                    curr = i
                    path = []
                    while not visited[curr]:
                        visited[curr] = True
                        path.append(curr)
                        # Find who is matched to curr
                        u = match_right[curr]
                        if u == -1:
                            break
                        # The next node in the cycle is u?
                        # If u is matched to curr, the edge is (u, curr).
                        # So we move to u.
                        curr = u
                        
                    if len(path) >= 3:
                        # Verify it's a cycle
                        # The path is v0, v1, ... vk.
                        # Edges are (u0, v0), (u1, v1)...
                        # Wait, my trace logic:
                        # curr = i.
                        # u = match_right[i]. Edge (u, i).
                        # Next curr = u.
                        # So path is [i, u, ...]
                        # The edges are (prev_u, curr).
                        # Let's re-verify the cycle structure.
                        # If match_right[v] = u, it represents edge (u, v).
                        # If we have a cycle in the matching, it looks like:
                        # u1 -> v1 (match_right[v1]=u1)
                        # u2 -> v2 (match_right[v2]=u2)
                        # ...
                        # uk -> vk (match_right[vk]=uk)
                        # And v1 = u2, v2 = u3, ..., vk = u1.
                        # So the vertices in the original graph are u1, u2, ..., uk.
                        # And the edges are (u1, u2), (u2, u3), ..., (uk, u1).
                        
                        # My trace:
                        # Start at i.
                        # u = match_right[i]. So edge (u, i).
                        # Next curr = u.
                        # So the sequence of vertices visited is i, u, ...
                        # The edges are (prev, curr)?
                        # Step 1: curr=i. u=match_right[i]. Edge (u, i). Next curr=u.
                        # Step 2: curr=u. u2=match_right[u]. Edge (u2, u). Next curr=u2.
                        # So the edges are (u, i), (u2, u), ...
                        # This forms a cycle i -> u -> u2 ... ?
                        # No, edge (u, i) connects u and i.
                        # Edge (u2, u) connects u2 and u.
                        # So the path is i - u - u2 - ...
                        # This is a valid cycle in the original graph.
                        
                        # Check if it closes.
                        # The last node in path is the one that points back to the first?
                        # In the while loop, we stop when visited[curr] is True.
                        # If it's a valid cycle, the last node's match should point to the first node?
                        # Let's just check if the set of edges in the path forms a cycle.
                        
                        # Actually, simpler:
                        # The path `path` contains vertices.
                        # The edges are determined by `match_right`.
                        # Edge between path[j] and path[j+1]?
                        # In my trace:
                        # path[0] = i.
                        # u = match_right[i].
                        # path[1] = u.
                        # Edge is (u, i).
                        # path[1] = u.
                        # u2 = match_right[u].
                        # path[2] = u2.
                        # Edge is (u2, u).
                        # So edge between path[j] and path[j-1] is (match_right[path[j]], path[j]).
                        # This is consistent.
                        
                        # To ensure it's a cycle, we need the last edge to connect back to the start.
                        # The loop stops when we hit a visited node.
                        # If we start at i and end at some node x that is already visited, and x is i, it's a cycle.
                        # But we mark visited as we go.
                        # If the matching forms a cycle, we will return to i.
                        # So `path` will contain all nodes in the cycle.
                        # We need to verify that the last node in `path` is connected to the first.
                        # The last node is `path[-1]`.
                        # The node that points to `path[-1]` is `match_right[path[-1]]`.
                        # Wait, in the loop:
                        # curr = path[-1].
                        # u = match_right[curr].
                        # If u is visited (which is i, the start), then the loop terminates.
                        # So the edge (u, curr) is (i, path[-1]).
                        # This closes the cycle.
                        
                        # So `path` is the list of vertices in the cycle.
                        # We need to order them correctly.
                        # The edges are (path[1], path[0]), (path[2], path[1]), ..., (path[0], path[-1]).
                        # This is a valid cycle: path[0] - path[1] - ... - path[-1] - path[0].
                        
                        cycles.append(path)
                        
            if not cycles:
                # No cycles of length >= 3 found
                # Check if there are 2-cycles
                # If match_right has any entries, they form cycles.
                # If all cycles were length 2, we should remove those edges.
                # But we already filtered len(path) >= 3.
                # So if no cycles added, we break to avoid infinite loop if only 2-cycles exist.
                # However, we should remove the edges involved in 2-cycles to make progress.
                # A 2-cycle in matching means match_right[v]=u and match_right[u]=v.
                # This corresponds to edge (u,v).
                # We should remove this edge from remaining_edges.
                
                # Find all 2-cycles
                removed_any = False
                for v in range(n):
                    u = match_right[v]
                    if u != -1 and match_right[u] == v:
                        # Check if we haven't already removed this edge
                        edge = (min(u, v), max(u, v))
                        if edge in remaining_edges:
                            remaining_edges.discard(edge)
                            removed_any = True
                            # Ensure we don't process the reverse
                            match_right[v] = -1
                            match_right[u] = -1
                            
                if not removed_any:
                    break
            
            # Remove edges of found cycles
            for cyc in cycles:
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    remaining_edges.discard((min(u, v), max(u, v)))
                pieces.append(cyc)
            
            # If no cycles and no 2-cycles removed, break
            if not cycles and not any(match_right):
                 break

        # Add remaining edges as individual pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
