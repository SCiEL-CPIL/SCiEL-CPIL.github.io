"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 16}

def _has_path_len_leq_2(n, adj, u, v):
    """Check if there is a path of length <= 2 between u and v in adjacency list."""
    if u in adj[v]:
        return True
    # Check common neighbors
    for w in adj[u]:
        if w in adj[v]:
            return True
    return False

def _check_diameter_two(n, adj):
    """Check if graph has diameter exactly 2. Returns True if valid."""
    # Check connectivity and diameter <= 2
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False  # disconnected
            if i != start and dist[i] > 2:
                return False  # diameter > 2
    return True

def _is_critical(n, adj, edge):
    """Check if removing this edge from adj makes diameter > 2 or disconnects."""
    u, v = edge
    
    # If u and v have a common neighbor in the current graph (excluding the edge u-v),
    # then removing u-v still leaves a path of length 2, so u and v are still dist 2.
    # We need to check if the *diameter* increases.
    # If u and v have a common neighbor, dist(u,v) remains 2.
    # If not, dist(u,v) becomes > 2 (or disconnected), so diameter increases.
    
    # Quick check: do u and v have a common neighbor?
    # Note: adj[u] contains v, and adj[v] contains u.
    # We need to check if there exists w != u, w != v such that w in adj[u] and w in adj[v].
    has_common = False
    for w in adj[u]:
        if w != v and w in adj[v]:
            has_common = True
            break
    
    if has_common:
        # dist(u,v) is still 2.
        # But could the diameter increase elsewhere?
        # In a dense graph, removing an edge rarely affects other pairs unless the graph is sparse.
        # However, we must be rigorous.
        # If the graph is "minimally diameter 2", usually every edge is a unique shortest path for its endpoints.
        # If there's an alternative path of length 2, the edge is NOT critical for the distance between its endpoints.
        # But the task says "Deleting EACH individual edge must increase its diameter above two or disconnect it."
        # This implies that for every edge (u,v), dist(u,v) must become > 2 or disconnected.
        # Therefore, if there is a common neighbor, the edge is NOT critical.
        return False
    
    # If no common neighbor, dist(u,v) becomes > 2 or disconnected.
    # This automatically increases the diameter (since it was 2, and now a pair is > 2).
    # So the edge is critical.
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (evaluator kills at 90s, but let's be safe)
    
    n = P['n']
    
    # Strategy: Start with Complete Graph K_n (120 edges)
    # Remove edges that are NOT critical.
    # An edge is critical if removing it increases diameter > 2.
    # In K_n, every pair has multiple common neighbors.
    # So no edge in K_n is critical?
    # Wait. If we remove an edge from K_n, the diameter remains 2 because there are many other paths.
    # So we need to remove edges until the graph becomes "minimally diameter 2".
    # This is equivalent to finding a maximal subgraph of K_n with diameter 2 such that every edge is essential.
    
    # Alternative Strategy: Start with K_{8,8} (64 edges).
    # Is K_{8,8} minimally diameter 2?
    # In K_{8,8}, any edge (u,v) with u in A, v in B.
    # If we remove it, is there a path of length 2?
    # u is connected to all other 7 in A? No, u is in A, connected to all in B.
    # Neighbors of u in K_{8,8} are all 8 vertices in B.
    # Neighbors of v in K_{8,8} are all 8 vertices in A.
    # Common neighbors of u and v: None. (A and B are disjoint, u in A, v in B. N(u) subset B, N(v) subset A. Intersection is empty).
    # So removing any edge from K_{8,8} makes dist(u,v) infinite (disconnected? No, graph is still connected, but dist(u,v) becomes 3? No, bipartite graph. Path must be odd length. u->B->A->v? No, u in A, v in B. Path u->b->a->v. Length 3.
    # So diameter becomes 3.
    # Thus K_{8,8} is minimally diameter 2.
    # Score = 64.
    
    # Can we do better than 64?
    # Max edges in a minimally diameter 2 graph on n vertices?
    # This is related to "minimally 2-connected" or similar extremal graph theory.
    # For n=16, can we have > 64 edges?
    
    # Let's try to add edges to K_{8,8}.
    # If we add an edge within A (say (a1, a2)), does it remain minimally diameter 2?
    # New graph G = K_{8,8} + E_A.
    # Edge (a1, a2) is critical?
    # If we remove (a1, a2), is there a path of length 2?
    # a1 is connected to all B. a2 is connected to all B.
    # Common neighbors: All 8 vertices in B.
    # So dist(a1, a2) is still 2.
    # So (a1, a2) is NOT critical.
    # Thus we cannot simply add edges to K_{8,8} and keep them.
    
    # We need to remove some edges from B to make the new edges critical?
    # This suggests a trade-off.
    
    # Let's use the greedy approach from inspiration 3, but optimized.
    # Start with empty graph? No, start with K_n and remove non-critical edges?
    # Or start with K_{8,8} and try to "upgrade" it?
    
    # Let's try the greedy addition from empty, but with a good heuristic order.
    # Or start with K_n and greedily remove non-critical edges?
    # Removing an edge is cheaper to check?
    # Check if edge is critical: O(n).
    # If we start with K_n (120 edges), and remove edges that are not critical.
    # In K_n, no edge is critical (many common neighbors).
    # So we would remove all edges? That leads to empty graph.
    # This direction is wrong. We need to *keep* edges that are critical.
    # So we should start with a graph where many edges are critical, and add more if possible.
    
    # K_{8,8} is a good base.
    # Can we find a graph with > 64 edges?
    # Consider a "wheel" or other structures?
    # Actually, the maximum number of edges in a minimally diameter 2 graph is a known problem.
    # For n=16, let's run a search.
    
    # Strategy:
    # 1. Start with K_{8,8}.
    # 2. Try to add random edges (within A or within B).
    # 3. When adding an edge, it becomes non-critical (due to common neighbors).
    # 4. To make it critical, we must remove some of its common neighbor edges?
    #    Specifically, to make (u,v) critical, we need N(u) \cap N(v) = empty (in the graph without u-v).
    #    This means we need to remove edges such that u and v share no common neighbor.
    #    This is a complex constraint satisfaction problem.
    
    # Let's stick to the greedy search from inspiration 3, but optimize the check.
    # The check `_is_critical` in inspiration 3 was O(E) or O(n^2).
    # We can maintain adjacency sets.
    
    # Let's try a different angle:
    # Is there a graph with 65+ edges?
    # Let's try to construct one.
    # Take K_{8,8}. Add edge (a1, a2).
    # To make (a1, a2) critical, we need no common neighbor.
    # Currently, all 8 in B are common neighbors.
    # We must remove at least one edge from a1 to each b in B, OR one edge from a2 to each b in B.
    # If we remove (a1, b1), then b1 is no longer a common neighbor of a1 and a2?
    # No, a2 is still connected to b1. a1 is not. So b1 is not in N(a1) \cap N(a2).
    # We need to break ALL common neighbors.
    # So we must remove edges (a1, b) for all b in B, OR (a2, b) for all b in B.
    # If we remove all (a1, b), we lose 8 edges to gain 1. Net -7. Bad.
    
    # What if we add multiple edges?
    # This suggests K_{8,8} might be optimal or close to it.
    
    # However, inspiration 3 scored 3.2, which is worse than 4.26.
    # Wait, the score is value/seed. Seed is likely 15 (or something).
    # 64 / 15 = 4.266.
    # So the current score 4.2667 corresponds to 64 edges.
    # Inspiration 3 scored 3.2, which is ~48 edges.
    # So K_{8,8} is better than the greedy search found.
    
    # Can we beat 64?
    # Let's try a randomized local search starting from K_{8,8}.
    # Try to swap edges?
    
    # Given the time limit and the difficulty of proving optimality, and the fact that K_{8,8} is a strong candidate,
    # I will implement a fast verification of K_{8,8} and a quick randomized search to see if we can find something better.
    # If not, return K_{8,8}.
    
    # Actually, let's just return K_{8,8} efficiently.
    # The previous programs did this and got 4.2667.
    # Is there a known graph with more edges?
    # For n=6, K_{3,3} has 9 edges. K_6 has 15.
    # Minimally diameter 2 graphs are also called "2-minimally connected" in some contexts? No.
    
    # Let's try to run a quick search for a better graph if time permits.
    # But the prompt says "Keep the program's entry-point function name(s)...".
    # And "COMBINE the strategy...".
    
    # I will implement the greedy search from inspiration 3 but with a much faster criticality check and a better starting point (K_{8,8} + random perturbations).
    
    # Actually, the most robust and fast solution is to return K_{8,8}.
    # The inspirations didn't beat it.
    # I will add a small optimization: verify K_{8,8} is valid (it is) and return it.
    
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
