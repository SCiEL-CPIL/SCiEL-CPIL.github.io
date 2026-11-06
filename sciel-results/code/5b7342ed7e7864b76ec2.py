import random
import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    def build_adj(edges_set):
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_max_book(adj, edges_set):
        max_b = 0
        for u, v in edges_set:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    # Structural construction:
    # K_{20,20} has 400 edges. We need 401 edges total.
    # Instead of adding 1 intra-edge (book=20), we remove k bipartite edges
    # and add k+1 intra-edges arranged to minimize max book.
    #
    # Strategy: partition into A=0..19, B=20..39
    # Remove k edges from bipartite, add k+1 intra-edges
    # The intra-edges should be spread so no vertex has too many intra-neighbors
    
    best_edges = None
    best_max = 999
    
    # Try different numbers of intra-edges
    for num_intra in range(1, 15):
        # Remove (num_intra - 1) bipartite edges to compensate
        num_remove = num_intra - 1
        
        # Build base K_{20,20}
        base = set()
        for a in range(20):
            for b in range(20, 40):
                base.add((a, b))
        
        # Choose which bipartite edges to remove (spread out)
        # Remove edges to reduce triangle formation potential
        all_bip = list(base)
        random.seed(42)
        random.shuffle(all_bip)
        remove_set = set(all_bip[:num_remove])
        
        # Choose intra-edges: spread within A and B
        # Use a matching-like structure to minimize shared neighbors
        intra_edges = []
        
        # Within A: add edges that don't share vertices too much
        # Within B: add edges that don't share vertices too much
        # Alternate between A and B intra-edges
        
        # For intra-A: pick pairs (i, i+10) type spread
        a_vertices = list(range(20))
        b_vertices = list(range(20, 40))
        random.shuffle(a_vertices)
        random.shuffle(b_vertices)
        
        # Create intra edges as a matching to minimize common neighbors
        # A matching means no vertex appears in more than 1 intra edge
        # This limits book size for intra edges to at most the number of 
        # common bipartite neighbors minus removed ones
        
        half = num_intra // 2
        rem = num_intra - half
        
        # Intra-A edges: matching
        for i in range(half):
            u = a_vertices[2*i]
            v = a_vertices[2*i+1]
            intra_edges.append((u, v))
        
        # Intra-B edges: matching
        for i in range(rem):
            u = b_vertices[2*i]
            v = b_vertices[2*i+1]
            intra_edges.append((u, v))
        
        # Build final edge set
        edges_set = set(base)
        edges_set.difference_update(remove_set)
        for e in intra_edges:
            edges_set.add(e)
        
        if len(edges_set) != target_edges:
            continue
        
        adj = build_adj(edges_set)
        cur_max = compute_max_book(adj, edges_set)
        
        if cur_max < best_max:
            best_max = cur_max
            best_edges = list(edges_set)
        
        # Local search to improve
        if cur_max <= best_max and time.time() < deadline - 10:
            # Try edge swaps to reduce max book
            for _iter in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Find an edge with high book
                high_edges = []
                for u, v in edges_set:
                    b = len(adj[u] & adj[v])
                    if b >= best_max:
                        high_edges.append((u, v, b))
                
                if not high_edges:
                    break
                
                random.shuffle(high_edges)
                u, v, _ = high_edges[0]
                uk = edge_key(u, v)
                
                # Try to swap this edge with a non-edge
                # Find a non-edge that would reduce max book
                found = False
                for _ in range(50):
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    if a == b:
                        continue
                    ak = edge_key(a, b)
                    if ak in edges_set or ak == uk:
                        continue
                    
                    # Test swap
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges_set.discard(uk)
                    
                    adj[a].add(b)
                    adj[b].add(a)
                    edges_set.add(ak)
                    
                    new_max = compute_max_book(adj, edges_set)
                    
                    if new_max < best_max:
                        best_max = new_max
                        best_edges = list(edges_set)
                        # Keep the swap
                        found = True
                        break
                    else:
                        # Revert
                        adj[a].discard(b)
                        adj[b].discard(a)
                        edges_set.discard(ak)
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        edges_set.add(uk)
                
                if not found:
                    break
        
        if best_max <= 8:
            break
    
    # Final local search on best solution
    if best_edges and time.time() < deadline - 5:
        edges_set = set(tuple(e) for e in best_edges)
        adj = build_adj(edges_set)
        cur_max = compute_max_book(adj, edges_set)
        
        for _iter in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Find max book edge
            max_b = 0
            max_edge = None
            for u, v in edges_set:
                b = len(adj[u] & adj[v])
                if b > max_b:
                    max_b = b
                    max_edge = (u, v)
            
            if max_edge is None:
                break
            
            u, v = max_edge
            uk = edge_key(u, v)
            
            improved = False
            # Try swapping with random non-edges
            for _ in range(100):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b:
                    continue
                ak = edge_key(a, b)
                if ak in edges_set or ak == uk:
                    continue
                
                # Test
                adj[u].discard(v)
                adj[v].discard(u)
                edges_set.discard(uk)
                
                adj[a].add(b)
                adj[b].add(a)
                edges_set.add(ak)
                
                new_max = compute_max_book(adj, edges_set)
                
                if new_max < cur_max:
                    cur_max = new_max
                    best_edges = [list(e) for e in edges_set]
                    if cur_max < best_max:
                        best_max = cur_max
                    improved = True
                    break
                else:
                    # Revert
                    adj[a].discard(b)
                    adj[b].discard(a)
                    edges_set.discard(ak)
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    edges_set.add(uk)
            
            if not improved:
                # Try 2-opt: remove one edge, add two edges, remove one more
                # Or just break
                break
    
    return best_edges
