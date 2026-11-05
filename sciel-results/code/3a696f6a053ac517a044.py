import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)

    def is_c4_free(n, edges):
        """Check if graph has no C4 using adjacency sets."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & adj[j]:
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0

    def validate(n, edges):
        if not is_c4_free(n, edges):
            return False
        return min_degree(n, edges) >= n - 39

    def polarity_graph(n):
        """Construct a polarity graph from finite field GF(q) where q = n-1.
        For q a prime power, the Hermitian polarity graph on q^2+1 points is C4-free.
        Here we use a simpler construction: take q = n-1, and for each vertex i,
        connect to vertices based on a quadratic form.
        """
        # Use the standard polarity graph construction for q = n-1
        # For q being a prime, we can construct a graph with q^2+1 vertices
        # But we need exactly n vertices, so let's use a different approach.
        
        # Simpler: use a random-like deterministic construction with structure
        edges = []
        adj = [set() for _ in range(n)]
        
        # Use a greedy construction with structured ordering
        # Start with a base structure: partition into groups
        # Group 0: vertices 0..k-1 (hub group)
        # Group 1: vertices k..n-1 (leaf group)
        # Connect hubs to leaves in a pattern that avoids C4
        
        # Actually, let's use the known construction:
        # For n = q^2 + 1, the polarity graph of PG(2,q) is C4-free with degree q+1
        # For n = q^2, we can use a subgraph
        
        # Let's try: for each pair (i,j), include edge if some condition holds
        # Use a bipartite-like structure with cross-connections
        
        # Better approach: use the incidence structure of a finite projective plane
        # If n-1 is a prime power q, then q^2+1 = n gives a perfect construction
        # For n=47: q^2+1=47 -> q^2=46, not a perfect square
        # For n=46: q^2+1=46 -> q^2=45, not a perfect square
        
        # Let's use a different well-known C4-free construction:
        # The Erdos-Renyi construction: take vertices as (x,y) in F_q x F_q
        # Connect (x1,y1) to (x2,y2) if x1*x2 + y1*y2 = 1 (or similar)
        # This gives a C4-free graph
        
        # For our purposes, let's use a simple greedy with backtracking
        # but with a good initial structure
        
        return None  # We'll use the search approach instead

    def fast_greedy_c4_free(n, min_deg_target, seed_val=42):
        """Fast greedy construction with C4 avoidance."""
        rng = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Create edge list and shuffle
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        # Precompute for fast C4 check
        # For adding edge (u,v): C4 exists if there exist a in adj[u], b in adj[v]
        # such that a != b and b in adj[a] (i.e., a-b is an edge)
        # Equivalently: |{a in adj[u] : adj[a] & adj[v]}| > 0 where a not in adj[v]
        # Simpler: check if there exist a,b with a in adj[u], b in adj[v], a!=b, and {a,b} in E
        
        def would_create_c4(u, v):
            # Check if adding (u,v) creates a C4
            # C4: u-a-v-b-u where a in adj[u], b in adj[v], a!=b, and a-b is an edge
            # So: exists a in adj[u], b in adj[v], a != b, a != v, b != u, and b in adj[a]
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u or b == a:
                        continue
                    if b in adj[a]:
                        return True
            return False
        
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            if v not in adj[u]:
                if not would_create_c4(u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return edges
        return None

    def local_search_improve(n, edges, min_deg_target, max_time=20.0):
        """Try to improve minimum degree by swapping edges."""
        if time.time() > deadline - 5:
            return None
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return [[u, v] for u, v in edges]
        
        start_ls = time.time()
        
        def would_create_c4(u, v):
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u or b == a:
                        continue
                    if b in adj[a]:
                        return True
            return False
        
        # Try to add edges to low-degree vertices
        for _ in range(1000):
            if time.time() - start_ls > max_time:
                break
            if time.time() > deadline - 5:
                break
            
            # Find vertex with minimum degree
            min_deg = min(degs)
            if min_deg >= min_deg_target:
                return [[u, v] for u, v in edges]
            
            low_v = degs.index(min_deg)
            
            # Find candidates to connect to
            candidates = [v for v in range(n) if v != low_v and v not in adj[low_v]]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if would_create_c4(low_v, v):
                    continue
                # Check if removing some edge from v would help
                adj[low_v].add(v)
                adj[v].add(low_v)
                edges.append((low_v, v))
                degs[low_v] += 1
                degs[v] += 1
                added = True
                break
            
            if not added:
                # Try removing an edge from a high-degree vertex and adding to low
                high_v = degs.index(max(degs))
                if degs[high_v] > min_deg_target + 2:
                    # Remove a random edge from high_v
                    if adj[high_v]:
                        neighbor = random.choice(list(adj[high_v]))
                        # Check if removing this edge is safe (doesn't violate C4-free)
                        # Removing edges never creates C4, so it's always safe
                        adj[high_v].discard(neighbor)
                        adj[neighbor].discard(high_v)
                        edges = [(u, v) for u, v in edges if not (u == high_v and v == neighbor) and not (u == neighbor and v == high_v)]
                        degs[high_v] -= 1
                        degs[neighbor] -= 1
        
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return [[u, v] for u, v in edges]
        return None

    # Try deterministic constructions first
    for n in [47, 46, 45]:
        if time.time() > deadline - 20:
            break
        
        min_deg_target = n - 39
        
        # Try multiple seeds for greedy
        found = None
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            edges = fast_greedy_c4_free(n, min_deg_target, seed_val=seed)
            if edges is not None:
                if validate(n, edges):
                    found = edges
                    break
        
        if found is not None:
            return {'n': n, 'edges': found}
        
        # Try local search improvement
        if time.time() < deadline - 15:
            # Start from best greedy result
            best_edges = None
            best_min_deg = -1
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                edges = fast_greedy_c4_free(n, min_deg_target - 2, seed_val=100 + seed)
                if edges is not None:
                    md = min_degree(n, edges)
                    if md > best_min_deg:
                        best_min_deg = md
                        best_edges = edges
            
            if best_edges is not None:
                improved = local_search_improve(n, best_edges, min_deg_target, max_time=15.0)
                if improved is not None:
                    if validate(n, improved):
                        return {'n': n, 'edges': improved}

    # Fallback: try smaller n
    for n in range(44, 38, -1):
        if time.time() > deadline - 5:
            break
        min_deg_target = n - 39
        if min_deg_target < 1:
            return {'n': n, 'edges': []}
        
        for seed in range(5):
            if time.time() > deadline - 3:
                break
            edges = fast_greedy_c4_free(n, min_deg_target, seed_val=seed)
            if edges is not None:
                if validate(n, edges):
                    return {'n': n, 'edges': edges}

    # Absolute fallback
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
