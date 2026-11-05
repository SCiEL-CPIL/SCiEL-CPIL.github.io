import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses random generation and girth verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # The seed program likely produces n = 2 * k ** ((g-1)//2 + 1) or similar.
    # For k=4, g=10, seed n is likely 2048.
    # We search for smaller n.
    
    def generate_random_regular_graph(n, k):
        """
        Generates a random simple k-regular graph on n vertices.
        Returns adjacency list or None if failed.
        """
        if n < k or (n * k) % 2 != 0:
            return None
        
        # Configuration model
        # Each vertex has k stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        for _ in range(100): # Retry limit for this n
            random.shuffle(stubs)
            edges = []
            adj = [[] for _ in range(n)]
            valid = True
            
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]: # Multi-edge
                    valid = False
                    break
                edges.append((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            
            if count == n:
                return adj, edges
        
        return None

    def has_short_cycle(adj, n, max_girth):
        """
        Checks if the graph has a cycle of length < max_girth.
        Uses BFS from each vertex up to depth max_girth // 2 - 1.
        """
        depth_limit = max_girth // 2 - 1
        
        for start in range(n):
            # BFS
            # dist[v] = (distance, parent)
            # We need to detect if we reach a node already visited by a different path
            # or if we reach the start node within depth_limit.
            
            visited = {}
            queue = deque()
            queue.append((start, -1, 0)) # node, parent, dist
            visited[start] = (-1, 0)
            
            while queue:
                u, parent, dist = queue.popleft()
                
                if dist > depth_limit:
                    continue
                
                for v in adj[u]:
                    if v == parent:
                        continue
                    
                    if v in visited:
                        # Found a cycle
                        # The cycle length is dist + 1 + dist_to_v
                        # Actually, if v is visited, and v is not parent,
                        # we found a cycle of length dist + 1 + visited[v][1]?
                        # No, standard BFS cycle detection:
                        # If we see v, and v is not the parent of u,
                        # and v was visited at distance d_v,
                        # then we have a cycle of length dist + 1 + d_v?
                        # Wait, BFS tree.
                        # If v is in the current level or previous level,
                        # and v is not parent, it's a cycle.
                        
                        # Simpler: Just check if we can reach start within depth_limit?
                        # No, that only detects cycles through start.
                        # We need to detect ANY cycle.
                        
                        # Correct logic:
                        # If v is already visited, and v is not the parent of u,
                        # then there is a cycle.
                        # The length of the cycle is dist + 1 + (dist of v)?
                        # No, the path from start to v is not necessarily part of the cycle.
                        # The cycle is formed by the edge (u,v) and the paths from start to u and start to v.
                        # This logic is tricky.
                        
                        # Alternative:
                        # Just run BFS from start, and if we reach a node that is already in the queue
                        # or visited, we need to be careful.
                        
                        # Let's use a simpler check:
                        # For each vertex, check if there is a cycle of length <= max_girth - 1.
                        # We can just check for cycles of length 4, 6, 8 explicitly?
                        # Or use the property: Girth >= g iff no two neighbors of any vertex are connected by a path of length < g-2?
                        
                        # Let's stick to BFS from each vertex, but check for back-edges.
                        # If we encounter a visited node v (v != parent),
                        # the cycle length is dist(u) + 1 + dist(v)? No.
                        # The cycle is the path from start to u, edge u-v, and path from v to start?
                        # No, the path from start to v is unique in BFS tree.
                        # So the cycle is (start ... u - v ... start).
                        # Length = dist(u) + 1 + dist(v).
                        # If this length < max_girth, we have a short cycle.
                        
                        d_v = visited[v][1]
                        cycle_len = dist + 1 + d_v
                        if cycle_len < max_girth:
                            return True
                    else:
                        visited[v] = (u, dist + 1)
                        queue.append((v, u, dist + 1))
                        
            # If we reached start again with dist > 0, that's a cycle too.
            # But the BFS above handles it if start is visited.
            # Actually, if v == start, and dist + 1 < max_girth, it's a cycle.
            # The logic above: if v is visited, check cycle_len.
            # If v == start, visited[start] is (-1, 0).
            # cycle_len = dist + 1 + 0 = dist + 1.
            # If dist + 1 < max_girth, return True.
            # This is correct.
            
        return False

    best_n = None
    best_edges = None
    
    # Search range
    # Seed is likely 2048.
    # We try to find something smaller.
    # Random graphs have girth ~ log_k(n).
    # For k=4, g=10, n ~ 4^5 = 1024.
    # We start from 500 and go up.
    
    for n in range(500, 1500, 20):
        if time.time() > deadline:
            break
            
        # Try a few random graphs for this n
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            result = generate_random_regular_graph(n, k)
            if result is None:
                continue
                
            adj, edges = result
            
            if not has_short_cycle(adj, n, g):
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            return best_n, best_edges

    # If we didn't find any in the loop (unlikely), generate a large one to be safe
    # Or just return a known valid one if we have one.
    # If best_n is still None, we must return *something* valid.
    # A random graph with n=2048 is almost certainly valid.
    if best_n is None:
        # Fallback: Generate a large random graph
        n = 2048
        while time.time() < deadline:
            result = generate_random_regular_graph(n, k)
            if result:
                adj, edges = result
                if not has_short_cycle(adj, n, g):
                    return n, edges
        # If even that fails, return empty? No, must return valid.
        # This should not happen.
        
    return best_n, best_edges
