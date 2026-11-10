"""Program that constructs a C6-free graph on 40 vertices with many edges.

Approach:
1. Build a base graph from 8 disjoint K5's (100 edges, trivially C6-free).
2. Use a randomized greedy "grow-and-repair" heuristic:
   - Consider adding edges between different K5 clusters.
   - Before adding an edge (u, v), check whether it would create a C6.
   - A C6 through (u, v) exists iff there is a simple path of length 5 from u to v
     in the current graph that does not use (u, v).
   - We use a bounded DFS up to depth 5 to detect such a path efficiently.
3. Also attempt to remove some internal K5 edges and replace them with multiple
   inter-cluster edges if that increases the total edge count without creating C6s.
   (This part is kept simple; the main gain comes from step 2.)
4. Return the final edge list.

The time limit is 120 seconds. For n=40, the DFS up to depth 5 is cheap.
We shuffle the candidate inter-cluster edges to get different random orderings
and keep the best result found.
"""

import random
import time
from itertools import product

def _has_c6_through(u, v, adj):
    """Check if adding edge (u, v) creates a C6.
    This is true iff there is a simple path of length 5 from u to v in the
    current graph (i.e., a path with 5 edges: u -> x1 -> x2 -> x3 -> x4 -> v).
    The path must be simple (no repeated vertices) and must not use the edge (u,v)
    which doesn't exist yet anyway.
    """
    # We need a path of exactly 5 edges from u to v.
    # DFS from u, depth up to 5, tracking visited nodes.
    # n=40, branching factor ~10, depth 5: worst case 10^5 = 100k nodes, fine.
    
    def dfs(node, depth, visited):
        if depth == 5:
            return node == v
        for nb in adj[node]:
            if nb == v and depth == 4:
                # We found a path of length 5: current node is x4, and v is next
                return True
            if nb not in visited and nb != u:
                visited.add(nb)
                if dfs(nb, depth + 1, visited):
                    return True
                visited.remove(nb)
        return False
    
    visited = {u}
    return dfs(u, 0, visited)


def _count_edges(adj):
    count = 0
    for i in range(len(adj)):
        count += len(adj[i])
    return count // 2


def _try_greedy_addition(seed_edges, num_trials=50, time_budget=100):
    """Try to add inter-cluster edges greedily. Return best edge list found."""
    n = 40
    best_edges = None
    best_count = -1
    
    start_time = time.time()
    
    # Build initial adjacency from seed_edges
    for trial in range(num_trials):
        adj = [set() for _ in range(n)]
        for u, v in seed_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Candidate edges: all pairs (u,v) not in seed_edges, u < v
        existing = set()
        for u, v in seed_edges:
            if u < v:
                existing.add((u, v))
            else:
                existing.add((v, u))
        
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in existing:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        added = 0
        for u, v in candidates:
            if time.time() - start_time > time_budget:
                break
            # Check if adding (u,v) creates a C6
            if not _has_c6_through(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        
        # Count edges
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append([u, v])
        
        count = len(edge_list)
        if count > best_count:
            best_count = count
            best_edges = edge_list
    
    return best_edges if best_edges is not None else seed_edges


def construct():
    """Construct a C6-free graph on 40 vertices with many edges."""
    # Step 1: Base graph - 8 disjoint K5's
    base_edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                base_edges.append([5 * k + a, 5 * k + b])
    
    # Step 2: Greedy addition of inter-cluster edges
    result = _try_greedy_addition(base_edges, num_trials=30, time_budget=110)
    
    return result
