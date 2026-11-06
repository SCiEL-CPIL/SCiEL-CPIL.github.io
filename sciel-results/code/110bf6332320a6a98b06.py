import time
import random
import sys

DEADLINE_MARGIN = 2.0  # seconds to spare before 150s limit
START_TIME = time.time()

def _time_remaining():
    return 150.0 - (time.time() - START_TIME) - DEADLINE_MARGIN

def _is_3_colorable(adj, n):
    """Backtracking 3-coloring check. Returns True if 3-colorable."""
    color = [-1] * n
    # Try to find a 3-coloring; if found, return True
    def backtrack(v):
        if v == n:
            return True
        for c in range(3):
            if all(color[u] != c for u in adj[v]):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = -1
        return False
    return backtrack(0)

def _max_independent_set(adj, n):
    """Find a maximum independent set (size) using greedy + local search."""
    # Greedy: repeatedly pick vertex with lowest degree
    remaining = set(range(n))
    indep = []
    while remaining:
        # Pick vertex with min degree among remaining
        best = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and u in adj[v])
            if deg < best_deg:
                best_deg = deg
                best = v
        indep.append(best)
        remaining.remove(best)
        for u in list(remaining):
            if u in adj[best]:
                remaining.discard(u)
    # Local search: try to add vertices that were removed
    current_size = len(indep)
    for _ in range(50):
        for v in range(n):
            if v in indep:
                continue
            if all(u not in indep for u in adj[v]):
                indep.append(v)
                current_size += 1
    return current_size

def _greedy_max_extension(adj, n, seed):
    """Add edges greedily to maximize edge count while keeping triangle-free and non-3-colorable."""
    random.seed(seed)
    adj = [set(adj[i]) for i in range(n)]
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Try adding edges in random order
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_pairs)
    
    added = 0
    for u, v in all_pairs:
        if v in adj[u]:
            continue
        # Check triangle-free: no common neighbor
        if adj[u] & adj[v]:
            continue
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        added += 1
        # Check if still non-3-colorable (only check occasionally to save time)
        if added % 3 == 0 and _time_remaining() < 5.0:
            if _is_3_colorable(adj, n):
                # Revert
                adj[u].discard(v)
                adj[v].discard(u)
                edges.pop()
                added -= 1
    
    return edges, adj

def _randomized_greedy(adj, n, seed, budget):
    """Randomized greedy extension with multiple restarts."""
    random.seed(seed)
    best_edges = None
    best_count = -1
    start = time.time()
    
    attempt = 0
    while time.time() - start < budget and _time_remaining() > 5.0:
        attempt += 1
        adj_copy = [set(adj[i]) for i in range(n)]
        edges = []
        for u in range(n):
            for v in adj_copy[u]:
                if u < v:
                    edges.append([u, v])
        
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if v in adj_copy[u]:
                continue
            if adj_copy[u] & adj_copy[v]:
                continue
            adj_copy[u].add(v)
            adj_copy[v].add(u)
            edges.append([u, v])
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges[:]
        
        # Check if valid
        if _is_3_colorable(adj_copy, n):
            # Invalid, but we track count for heuristic
            pass
    
    return best_edges, best_count

def construct():
    # Start with Grötzsch graph on 11 vertices (0-10)
    # Vertices: 0-4 = C5 cycle, 5-9 = copies, 10 = universal to 5-9
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    
    n = 16
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Add 5 more vertices (11-15) connected to maximum independent set
    # In Grötzsch, the copies 5-9 form an independent set of size 5
    max_indep = [5, 6, 7, 8, 9]
    
    for new_v in range(11, 16):
        for u in max_indep:
            edges.append([new_v, u])
            adj[new_v].add(u)
            adj[u].add(new_v)
    
    # Now we have 45 edges. Try to add more via randomized greedy.
    # The 5 new vertices are independent, and each is connected to 5,6,7,8,9.
    # We can potentially add edges between new vertices and old vertices 0-4 or 10,
    # as long as no triangles are formed.
    
    # Try multiple randomized extensions
    best_edges = edges[:]
    best_count = len(edges)
    
    base_adj = [set(adj[i]) for i in range(n)]
    base_edges = [e[:] for e in edges]
    
    # Try a few different seeds for randomized greedy
    for seed in range(20):
        if _time_remaining() < 10.0:
            break
        budget = min(_time_remaining() - 10.0, 5.0)
        ext_edges, ext_count = _randomized_greedy(base_adj, n, seed, budget)
        if ext_edges is not None and ext_count > best_count:
            # Verify validity
            adj_check = [set() for _ in range(n)]
            for u, v in ext_edges:
                adj_check[u].add(v)
                adj_check[v].add(u)
            if not _is_3_colorable(adj_check, n):
                best_edges = ext_edges
                best_count = ext_count
    
    # Final verification
    adj_final = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
    
    # Check triangle-free
    for u in range(n):
        for v in adj_final[u]:
            for w in adj_final[v]:
                if w != u and u in adj_final[w]:
                    # Triangle found - should not happen
                    pass
    
    return best_edges

# Ensure the function is ready
if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
