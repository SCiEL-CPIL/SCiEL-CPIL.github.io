import time
import random

def construct():
    deadline = time.time() + 140
    n = 31
    target_edges = 91
    
    # Pre-compute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4s_created(u, v, adj):
        """
        Count C4s created by adding edge (u, v) to graph with adjacency sets adj.
        A C4 is u-a-b-v-u where a in N(u), b in N(v), a~b, a!=v, b!=u, a!=b.
        """
        count = 0
        # Optimization: iterate over the smaller neighborhood
        if len(adj[u]) < len(adj[v]):
            for a in adj[u]:
                if a == v: continue
                for b in adj[v]:
                    if b == u or b == a: continue
                    if a in adj[b]:
                        count += 1
        else:
            for b in adj[v]:
                if b == u: continue
                for a in adj[u]:
                    if a == v or a == b: continue
                    if a in adj[b]:
                        count += 1
        return count

    def is_c4_free_addition(u, v, adj):
        return count_c4s_created(u, v, adj) == 0

    best_edges = None
    best_score = float('inf')
    
    # Try multiple random seeds/orders to ensure we reach 90 C4-free edges
    for seed in range(10):
        if time.time() > deadline - 20:
            break
            
        random.seed(seed)
        shuffled_pairs = all_pairs[:]
        random.shuffle(shuffled_pairs)
        
        adj = [set() for _ in range(n)]
        edges = []
        edge_set = set()
        
        # Phase 1: Build C4-free graph with up to 90 edges
        for u, v in shuffled_pairs:
            if len(edges) >= 90:
                break
            if is_c4_free_addition(u, v, adj):
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
        
        if len(edges) < 90:
            # Failed to reach 90, try next seed
            continue
        
        # Phase 2: Add the 91st edge to minimize C4s
        # We have 90 C4-free edges.
        # Find the pair (u, v) not in edge_set that minimizes count_c4s_created(u, v, adj)
        min_c4s = float('inf')
        best_pair = None
        
        # Check all non-edges
        for u, v in all_pairs:
            if (min(u, v), max(u, v)) in edge_set:
                continue
            c4s = count_c4s_created(u, v, adj)
            if c4s < min_c4s:
                min_c4s = c4s
                best_pair = (u, v)
            if min_c4s == 0:
                break # Can't get better than 0
        
        if best_pair is None:
            # Should not happen, but just in case
            continue
            
        u, v = best_pair
        edges.append([u, v])
        
        # Calculate total C4s for this graph
        # Since the first 90 edges form a C4-free graph, the total C4s is exactly min_c4s
        # (All C4s must include the new edge, as the base is C4-free)
        score = min_c4s
        
        if score < best_score:
            best_score = score
            best_edges = edges[:]
            
        if best_score == 0:
            break # Perfect score found
            
    if best_edges is None:
        # Fallback: return any 91 edges (should not happen if seeds work)
        return all_pairs[:91]
        
    return best_edges
