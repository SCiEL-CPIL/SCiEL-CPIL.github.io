"""Optimized graph construction for 16 vertices, 65 edges, minimizing max triangle degree sum."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 16
    target_edges = 65
    
    def compute_score(edges):
        """Compute the max triangle degree sum."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        max_sum = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def edge_set_to_list(es):
        return [[u, v] for u, v in es if u < v]
    
    def make_seed(a, b):
        """Create K_{a,b} plus (target_edges - a*b) edges within the b-part."""
        edges = set()
        # Bipartite edges: part A = {0..a-1}, part B = {a..a+b-1}
        for u in range(a):
            for v in range(a, a + b):
                edges.add((u, v))
        
        # Add edges within part B
        needed = target_edges - a * b
        b_part = list(range(a, a + b))
        internal_edges = list(combinations(b_part, 2))
        random.shuffle(internal_edges)
        for e in internal_edges[:needed]:
            edges.add(e)
        
        return edges
    
    def local_search(edges_set, max_iter=50000):
        """Try edge swaps to reduce max triangle degree sum."""
        edges = set(edges_set)
        current_score = compute_score(edge_set_to_list(edges))
        best_edges = set(edges)
        best_score = current_score
        
        # Precompute degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all non-edges
        all_pairs = set((i, j) for i in range(n) for j in range(i+1, n))
        non_edges = all_pairs - edges
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            edge_list = list(edges)
            if not edge_list:
                break
            e_remove = random.choice(edge_list)
            if e_remove[0] > e_remove[1]:
                e_remove = (e_remove[1], e_remove[0])
            
            # Pick a random non-edge to add
            non_edge_list = list(non_edges)
            if not non_edge_list:
                break
            e_add = random.choice(non_edge_list)
            if e_add[0] > e_add[1]:
                e_add = (e_add[1], e_add[0])
            
            # Apply swap
            edges.discard(e_remove)
            edges.add(e_add)
            non_edges.discard(e_add)
            non_edges.add(e_remove)
            
            # Update degrees
            u1, v1 = e_remove
            u2, v2 = e_add
            deg[u1] -= 1
            deg[v1] -= 1
            deg[u2] += 1
            deg[v2] += 1
            
            # Update adjacency
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            # Recompute score (expensive but necessary for accuracy)
            new_score = compute_score(edge_set_to_list(edges))
            
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
            else:
                # Revert
                edges.discard(e_add)
                edges.add(e_remove)
                non_edges.discard(e_remove)
                non_edges.add(e_add)
                deg[u1] += 1
                deg[v1] += 1
                deg[u2] -= 1
                deg[v2] -= 1
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].discard(v2)
                adj[v2].discard(u2)
        
        return best_edges, best_score
    
    # Try multiple seed configurations
    best_edges = None
    best_score = float('inf')
    
    # Try different bipartite splits
    for a in [5, 6, 7]:
        b = n - a
        if a * b > target_edges:
            continue
        needed = target_edges - a * b
        if needed < 0 or needed > b * (b - 1) // 2:
            continue
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            random.seed(42 + trial * 100 + a)
            edges_set = make_seed(a, b)
            
            # Local search
            ls_edges, ls_score = local_search(edges_set, max_iter=20000)
            
            if ls_score < best_score:
                best_score = ls_score
                best_edges = ls_edges
            
            if best_score <= 23:
                break
        
        if best_score <= 23:
            break
    
    if best_edges is None:
        # Fallback
        best_edges = make_seed(6, 10)
        best_score = compute_score(edge_set_to_list(best_edges))
    
    return {'n': n, 'edges': edge_set_to_list(best_edges)}
