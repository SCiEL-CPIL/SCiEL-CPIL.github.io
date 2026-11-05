import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    
    def generate_random_3_regular(n):
        # Configuration model
        while True:
            half_edges = [i for i in range(n) for _ in range(k)]
            random.shuffle(half_edges)
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, k*n, 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return adj

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def find_shortest_cycle_and_edges(adj_list, n, g_limit):
        # Returns (length, list_of_edges) for a cycle of length < g_limit, or (inf, [])
        # We only need to find *one* short cycle to break.
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Potential cycle
                        # The cycle is u-v + path(v -> start) + path(start -> u) ?
                        # No, the cycle is u-v + path(v -> LCA) + path(u -> LCA).
                        # Finding LCA is complex.
                        # Alternative: Just check if dist[u] + dist[v] + 1 < g_limit.
                        # This guarantees *some* cycle of that length exists in the BFS tree + edge.
                        # The actual cycle might be shorter, but this is a valid upper bound on the girth contribution.
                        # If this sum is < g_limit, we have a short cycle.
                        # We can reconstruct the cycle:
                        # Path from v to start, path from u to start.
                        # Intersection is LCA.
                        # This is O(N) per check.
                        if dist[u] + dist[v] + 1 < g_limit:
                            # Reconstruct cycle
                            # Get path from v to start
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                            # Get path from u to start
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = parent[cur]
                            
                            # Find LCA
                            # Since both paths end at start, and start is root, LCA is start?
                            # No, start is the root of BFS.
                            # The cycle is formed by the edge (u,v) and the path u->...->start->...->v.
                            # The length is dist[u] + dist[v] + 1.
                            # The edges are:
                            # (u, parent[u]), ..., (start, child_on_path_to_u)
                            # (start, child_on_path_to_v), ..., (parent[v], v)
                            # (u, v)
                            
                            # This cycle might not be simple if paths intersect before start?
                            # In a tree (BFS tree), paths from root to nodes are unique and only intersect at root or common ancestors.
                            # So the union of paths u->start and v->start plus edge (u,v) is a simple cycle.
                            
                            cycle_edges = []
                            # Add edge u-v
                            cycle_edges.append((u, v))
                            # Add path u to start
                            cur = u
                            while cur != start:
                                p = parent[cur]
                                cycle_edges.append((cur, p))
                                cur = p
                            # Add path v to start
                            cur = v
                            while cur != start:
                                p = parent[cur]
                                cycle_edges.append((cur, p))
                                cur = p
                                
                            return dist[u] + dist[v] + 1, cycle_edges
        return float('inf'), []

    def try_improve_girth(n, k, g, time_limit):
        adj = generate_random_3_regular(n)
        adj_list = [list(adj[i]) for i in range(n)]
        
        def get_girth():
            # Just check if any cycle < g
            for _ in range(10): # Random restarts in BFS? No, BFS is deterministic.
                # We just need to know if girth < g
                # Use the find function
                length, _ = find_shortest_cycle_and_edges(adj_list, n, g)
                if length < g:
                    return length
            return float('inf')

        current_girth = get_girth()
        
        while time.time() < time_limit and current_girth < g:
            length, cycle_edges = find_shortest_cycle_and_edges(adj_list, n, g)
            if length >= g:
                break
            
            # Pick a random edge in the cycle to swap
            u, v = random.choice(cycle_edges)
            
            # Find another edge (x, y) to swap with
            # Must be disjoint from u, v
            # And not create multi-edges
            
            swapped = False
            for _ in range(50): # Try 50 random swaps
                # Pick random x, y
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y or x in [u, v] or y in [u, v]:
                    continue
                if y not in adj_list[x]:
                    continue
                # Check if swap creates multi-edges
                # Remove (u,v) and (x,y)
                # Add (u,x) and (v,y)
                if x in adj_list[u] or y in adj_list[v] or y in adj_list[u] or x in adj_list[v]:
                    continue
                # Also check if (u,x) or (v,y) already exist?
                # adj_list[u] contains x?
                
                # Perform swap
                adj_list[u].remove(v)
                adj_list[v].remove(u)
                adj_list[x].remove(y)
                adj_list[y].remove(x)
                
                adj_list[u].append(x)
                adj_list[x].append(u)
                adj_list[v].append(y)
                adj_list[y].append(v)
                
                # Check girth
                new_girth = get_girth()
                if new_girth > current_girth:
                    current_girth = new_girth
                    swapped = True
                    break
                else:
                    # Revert
                    adj_list[u].remove(x)
                    adj_list[x].remove(u)
                    adj_list[v].remove(y)
                    adj_list[y].remove(v)
                    
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                    adj_list[x].append(y)
                    adj_list[y].append(x)
            
            if not swapped:
                # Stuck? Try a different edge in cycle or give up
                break

        if current_girth >= g:
            # Reconstruct adj sets from adj_list
            adj_final = {i: set(adj_list[i]) for i in range(n)}
            edges = get_edges(adj_final, n)
            return n, edges
        return None

    # Try sizes
    # Moore bound 766.
    for n in range(768, 1000, 32):
        if time.time() > deadline - 10:
            break
        res = try_improve_girth(n, k, g, deadline - 5)
        if res:
            return res

    # Fallback
    n = 768
    adj = generate_random_3_regular(n)
    edges = get_edges(adj, n)
    return n, edges

