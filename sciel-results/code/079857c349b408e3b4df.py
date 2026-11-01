<<<<<<< SEARCH
import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using BFS from each vertex (or optimized non-backtracking)
        # For small graphs, BFS from each vertex is fine.
        # Girth is the length of the shortest cycle.
        # A cycle of length L implies a path of length L-1 from a vertex to one of its neighbors.
        # So we look for the shortest path from u to v where v is a neighbor of u, with length >= 2.
        # Actually, shortest cycle through u is the min over neighbors v of (dist(u, v) + 1) where dist is shortest path not using the edge (u,v) directly?
        # Simpler: BFS from u, but don't traverse back to u.
        # Or: BFS from all vertices, find shortest cycle.
        
        # Optimized check:
        # For each vertex u, run BFS to find the shortest cycle containing u.
        # We can do this by doing BFS from u, but when we reach a neighbor v of u, the cycle length is dist[u][v] + 1.
        # We must ensure the path doesn't use the edge (u,v) as the first step?
        # Actually, if we start BFS from u, and we visit a neighbor v at distance d, the cycle is d+1.
        # But if d=1, that's just the edge, not a cycle. So we need d >= 2.
        # However, BFS finds shortest paths. If the shortest path from u to v has length 1, that's the edge.
        # If there is a cycle of length 3, u-v-w-u, then dist(u, v) via w is 2. So cycle length is 3.
        # So we just need to find the minimum (dist(u, v) + 1) for all u, and all neighbors v of u, where dist(u, v) >= 2.
        # Note: dist(u, v) is the standard shortest path distance.
        
        # Let's just do BFS from each u.
        for u in range(n):
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= g:  # If we've already found a path of length g, no shorter cycle can exist through this level
                    break
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        # The cycle length is dist[curr] + 1 + dist[v] ? No.
                        # This is for general graph. 
                        # For cycle containing u:
                        # We are looking for a path from u to a neighbor v of u, of length L.
                        # Then cycle length is L + 1.
                        # In BFS from u, if we encounter an edge (curr, v) where v is a neighbor of u:
                        # Then we have a path u -> ... -> curr -> v -> u.
                        # Length = dist[curr] + 1 + 1 = dist[curr] + 2.
                        # Wait, if v is a neighbor of u, and we are at curr, and curr is connected to v.
                        # Path: u ... curr - v - u.
                        # Length: dist[u][curr] + 1 (curr to v) + 1 (v to u).
                        # So cycle length = dist[curr] + 2.
                        
                        # But we only care if v is a neighbor of u.
                        if v in adj[u] and v != curr: # v != curr is redundant if no loops, but safe
                            cycle_len = dist[curr] + 2
                            if cycle_len < g:
                                return False
                        # Also, if v is already visited and is not the parent, it might form a cycle not containing u.
                        # But we are checking girth, which is the minimum cycle in the whole graph.
                        # If any cycle is shorter than g, we fail.
                        # The standard BFS from a single source finds shortest cycles through that source.
                        # To find the global girth, we need to check all sources?
                        # Yes.
                        
                        # Actually, the check `if v in adj[u]` only catches cycles through u.
                        # What about a cycle not through u?
                        # It will be caught when we run BFS from a vertex on that cycle.
                        # So checking all u is correct.
                        
                        # However, there is a subtle bug. 
                        # If we find a shorter path to v, we don't update dist. 
                        # But if we find a cycle, we should check if it's shorter than g.
                        # The cycle found by edge (curr, v) where dist[v] != -1 has length dist[curr] + 1 + dist[v].
                        # This cycle might not contain u.
                        # So we should check ALL back edges.
                        
                        # Let's refine:
                        # When we see an edge (curr, v) where dist[v] != -1:
                        # We have a cycle of length dist[curr] + 1 + dist[v].
                        # If this length < g, return False.
                        
                        # BUT, this only works if the path from u to curr and u to v are disjoint except at u?
                        # No, BFS tree paths are disjoint except at u? Not necessarily.
                        # However, the length dist[curr] + 1 + dist[v] is a valid upper bound on the cycle length.
                        # If it's < g, then girth < g.
                        # So this is a valid check.
                        
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True
=======
import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        Uses a time limit to prevent hanging on large graphs.
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Optimized Girth Check:
        # We perform BFS from each vertex u.
        # We stop BFS at depth g/2 (floor) because any cycle shorter than g would be detected.
        # Specifically, a cycle of length L < g implies there is a path of length L-1 < g-1.
        # If we find an edge (u, v) such that dist(u, v) + 1 < g, we have a short cycle?
        # No, dist(u, v) is shortest path.
        # If we do BFS from u, and we find an edge (curr, v) where v is already visited:
        # Cycle length <= dist[curr] + 1 + dist[v].
        # If this is < g, girth < g.
        # To be safe and fast, we limit BFS depth to g//2.
        # If no cycle < g is found within this depth from any node, girth >= g.
        # Proof: If there is a cycle C of length L < g, pick a node u on C.
        # The distance from u to any other node on C is at most L/2.
        # So BFS from u to depth L/2 will encounter a back-edge or cross-edge forming C.
        
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        q.append(v)
                    else:
                        # Found a cycle.
                        # The cycle length is dist[curr] + 1 + dist[v].
                        # We only care if it's < g.
                        # Note: If v is the parent, dist[v] = dist[curr] - 1, cycle len = 2 (edge), but we ignore self-loops and multi-edges.
                        # Actually, if v is parent, dist[curr] + 1 + dist[v] = dist[curr] + 1 + dist[curr] - 1 = 2 * dist[curr].
                        # But the edge (curr, v) is the tree edge. This doesn't form a cycle in the simple graph sense unless there's another path.
                        # Wait, BFS tree edges don't form cycles. Only non-tree edges do.
                        # If dist[v] != -1, (curr, v) is either a tree edge (v is parent or child) or a non-tree edge.
                        # If v is parent: dist[v] = dist[curr] - 1. Cycle len = 2*dist[curr]. This is not a simple cycle formed by this edge alone.
                        # If v is child: dist[v] = dist[curr] + 1. Cycle len = 2*dist[curr] + 2.
                        # If v is cross edge: dist[v] <= dist[curr]. Cycle len = dist[curr] + 1 + dist[v].
                        
                        # The formula dist[curr] + 1 + dist[v] gives the length of the cycle formed by the path u->...->curr, edge (curr,v), path v->...->u.
                        # This is a valid cycle length.
                        # If this length < g, we fail.
                        
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True
>>>>>>> REPLACE

