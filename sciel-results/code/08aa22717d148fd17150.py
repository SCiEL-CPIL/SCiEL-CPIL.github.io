import random
import time
import sys

def construct():
    n = 20
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    # ---- New approach: Iterative greedy with local search ----
    # Strategy: Build graph greedily, then try to improve by adding/removing edges
    
    def greedy_build(seed_rng):
        """Build a K2,3-free graph greedily with random ordering."""
        adj = [set() for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        
        def can_add(u, v):
            if u == v or v in adj[u]:
                return False
            # Check if adding u-v would create K2,3
            # For each w in adj[u], w and v would have common neighbors
            # including u, so we need common[w][v] + 1 <= 2, i.e., common[w][v] <= 1
            for w in adj[u]:
                if common[w][v] > 1:
                    return False
            for w in adj[v]:
                if common[u][w] > 1:
                    return False
            return True
        
        def do_add(u, v):
            adj[u].add(v)
            adj[v].add(u)
            # Update common counts
            for w in adj[u]:
                if w != v:
                    common[w][v] += 1
                    common[v][w] += 1
            for w in adj[v]:
                if w != u:
                    common[u][w] += 1
                    common[w][u] += 1
        
        # Try adding edges in random order
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        seed_rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add(u, v):
                do_add(u, v)
        
        return [[u, v] for u in range(n) for v in adj[u] if u < v]
    
    def local_search(edges, max_iters=500):
        """Try to improve by removing edges and trying to add others."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[u]:
                        if w != v and w in adj[v]:
                            common[u][w] += 1
                            common[w][u] += 1
        
        def can_add(u, v):
            if u == v or v in adj[u]:
                return False
            for w in adj[u]:
                if common[w][v] > 1:
                    return False
            for w in adj[v]:
                if common[u][w] > 1:
                    return False
            return True
        
        def do_add(u, v):
            adj[u].add(v)
            adj[v].add(u)
            for w in adj[u]:
                if w != v:
                    common[w][v] += 1
                    common[v][w] += 1
            for w in adj[v]:
                if w != u:
                    common[u][w] += 1
                    common[w][u] += 1
        
        def do_remove(u, v):
            adj[u].remove(v)
            adj[v].remove(u)
            for w in adj[u]:
                if w != v:
                    common[w][v] -= 1
                    common[v][w] -= 1
            for w in adj[v]:
                if w != u:
                    common[u][w] -= 1
                    common[w][u] -= 1
        
        current_edges = len(edges)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Try to add an edge
            all_pairs = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            rng.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if can_add(u, v):
                    do_add(u, v)
                    improved = True
                    break
            
            # If we couldn't add, try removing a random edge to free up space
            if not improved and len(edges) > 20:
                edge_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if edge_list:
                    u, v = rng.choice(edge_list)
                    do_remove(u, v)
                    improved = True
        
        return [[u, v] for u in range(n) for v in adj[u] if u < v]
    
    # Run multiple greedy builds with different seeds
    for trial in range(20):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = greedy_build(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            # Try local search to improve
            improved = local_search(edges, max_iters=100)
            improved = verify(improved)
            
            if improved is not None and len(improved) > best_e:
                best_e = len(improved)
                best = improved
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
