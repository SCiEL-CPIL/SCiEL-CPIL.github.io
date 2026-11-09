import random
import time
import sys
from itertools import combinations

def construct():
    start = time.time()
    deadline = 145  # Safety margin before 150s kill

    def can_add_edge_fast(n, adj, u, v):
        """
        Check if adding (u,v) creates a C4.
        Creates C4 iff ∃ path of length 3 from u to v: u-a-b-v
        i.e., ∃ a∈N(u), b∈N(v), a≠b, a~b
        Since b∈N(a) implies b≠a, just check: ∃ a∈N(u) s.t. N(a)∩N(v)≠∅
        """
        for a in adj[u]:
            if adj[a] & adj[v]:
                return False
        return True

    def is_c4_free(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if len(adj[u] & adj[v]) > 1:
                        return False
        return True

    def min_degree(n, adj):
        return min(len(adj[i]) for i in range(n))

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def try_local_search(adj, n, target_deg, max_iters=500):
        """Local search to improve minimum degree while maintaining C4-free property."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= target_deg:
                return True
            
            # Find all vertices below target degree
            low_verts = [i for i in range(n) if len(adj[i]) < target_deg]
            
            # Try to add edges to low-degree vertices
            improved = False
            for u in low_verts:
                # Get candidates that would not create C4
                candidates = []
                for v in range(n):
                    if v == u or v in adj[u]:
                        continue
                    if can_add_edge_fast(n, adj, u, v):
                        candidates.append(v)
                
                # Shuffle and try to add edges
                random.shuffle(candidates)
                for v in candidates:
                    if len(adj[u]) >= target_deg:
                        break
                    # Verify this addition doesn't break C4-free for all pairs
                    adj[u].add(v)
                    adj[v].add(u)
                    if is_c4_free(n, adj):
                        improved = True
                        if len(adj[u]) >= target_deg:
                            break
                    else:
                        # Revert
                        adj[u].remove(v)
                        adj[v].remove(u)
                
                if len(adj[u]) >= target_deg:
                    continue
                if time.time() > deadline:
                    break
            
            if not improved:
                break
        
        return min_degree(n, adj) >= target_deg

    def construct_polar_graph(n, target_deg):
        """
        Try to construct a C4-free graph using polar decomposition approach.
        Split vertices into two parts and use structured connections.
        """
        adj = [set() for _ in range(n)]
        
        # Try different split sizes
        for split in range(n // 4, 3 * n // 4 + 1):
            if time.time() > deadline:
                break
            
            part_a = list(range(split))
            part_b = list(range(split, n))
            
            # Randomly assign edges within parts (sparse to avoid C4)
            random.shuffle(part_a)
            random.shuffle(part_b)
            
            # Add some edges within part_a (keep sparse)
            max_inner_a = max(1, len(part_a) // 5)
            for i in range(min(max_inner_a, len(part_a) - 1)):
                u, v = part_a[i], part_a[(i + 1) % len(part_a)]
                if can_add_edge_fast(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Add some edges within part_b (keep sparse)
            max_inner_b = max(1, len(part_b) // 5)
            for i in range(min(max_inner_b, len(part_b) - 1)):
                u, v = part_b[i], part_b[(i + 1) % len(part_b)]
                if can_add_edge_fast(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Add cross edges between part_a and part_b
            # Each vertex in part_a connects to a subset of part_b
            for u in part_a:
                # Connect to random subset of part_b
                k = min(target_deg, len(part_b))
                subset = random.sample(part_b, k)
                for v in subset:
                    if can_add_edge_fast(n, adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Each vertex in part_b connects to a subset of part_a
            for v in part_b:
                if len(adj[v]) < target_deg:
                    # Connect to more vertices in part_a
                    k = target_deg - len(adj[v])
                    available = [u for u in part_a if u not in adj[v]]
                    if len(available) > k:
                        subset = random.sample(available, k)
                        for u in subset:
                            if can_add_edge_fast(n, adj, u, v):
                                adj[u].add(v)
                                adj[v].add(u)
            
            # Check if we achieved target degree
            if min_degree(n, adj) >= target_deg and is_c4_free(n, adj):
                return adj
        
        return None

    def greedy_random(n, target_deg, seed=None):
        """Greedy random construction."""
        if seed is not None:
            random.seed(seed)
        
        adj = [set() for _ in range(n)]
        
        # Generate random order of edges
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        
        # Greedy addition
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if can_add_edge_fast(n, adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        # Try local search to improve
        if min_degree(n, adj) < target_deg:
            try_local_search(adj, n, target_deg, max_iters=200)
        
        if min_degree(n, adj) >= target_deg and is_c4_free(n, adj):
            return adj
        
        return None

    best = {'n': 39, 'edges': []}

    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        target_deg = n - 39
        
        # Strategy 1: Try polar graph construction
        if time.time() < deadline - 10:
            adj = construct_polar_graph(n, target_deg)
            if adj is not None:
                best = {'n': n, 'edges': to_edges(adj, n)}
                found = True
                break
        
        # Strategy 2: Multiple greedy random attempts
        if not found:
            attempts = 0
            max_attempts = 200
            
            while attempts < max_attempts and time.time() < deadline - 5 and not found:
                attempts += 1
                
                adj = greedy_random(n, target_deg, seed=attempts * 1000)
                
                if adj is not None:
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
        
        # Strategy 3: If still not found, try more aggressive local search
        if not found and time.time() < deadline - 2:
            # Start with a random sparse graph and do intensive local search
            adj = [set() for _ in range(n)]
            
            # Add random edges
            edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(edges_to_try)
            
            # Add about 30% of edges
            for u, v in edges_to_try[:len(edges_to_try) // 3]:
                if can_add_edge_fast(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Intensive local search
            if try_local_search(adj, n, target_deg, max_iters=1000):
                best = {'n': n, 'edges': to_edges(adj, n)}
                found = True
        
        if found:
            break

    return best

# EVOLVE-BLOCK-END
