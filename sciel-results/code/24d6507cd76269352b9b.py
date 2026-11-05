import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # 1. Deterministic Initial Construction using Paley-like structure
    # Paley graph on q vertices (q = 4k+1 prime power) is strongly regular.
    # For n=20, we can use a subgraph or a cyclic construction.
    # A known dense K2,3-free graph on 20 vertices has 100 edges (ex(20, K2,3) >= 100).
    # The complement of a strongly regular graph (19, 9, 4, 5) is (19, 9, 4, 5).
    # Let's try to construct a graph close to the Hoffman-Singleton graph parameters or similar.
    # Actually, ex(20, K2,3) is known to be 100.
    # A (20, 10) regular graph that is K2,3-free would be ideal.
    
    # Strategy: Start with a random graph and optimize, or start with a structured graph.
    # The current program scores 1.9667 which corresponds to ~ 190 * 0.98? No, score is value/seed.
    # Seed is likely a basic random graph or empty? "1.0 equals the seed program".
    # If seed is random, it's probably low. 1.96 * seed_value.
    # Let's assume the goal is to get as many edges as possible. Max is 190.
    # ex(20, K2,3) is likely around 100-110.
    
    # We will use a fast local search starting from multiple random initializations 
    # but with a much faster constraint check using bitmasks and vectorized operations.
    
    def validate_and_optimize(adj_masks, edges):
        """
        Given an initial graph, remove edges to satisfy K2,3-free constraint,
        then try to add edges back to maximize count.
        """
        n = 20
        # Precompute common neighbor counts for all pairs
        # common[u][v] = number of common neighbors of u and v
        # If common[u][v] >= 3, we have a K2,3.
        
        # Fast check and repair loop
        adj = adj_masks[:]
        changed = True
        while changed:
            changed = False
            # Find all violations
            violations = []
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = bin(common).count('1')
                    if cnt >= 3:
                        violations.append((u, v, common, cnt))
            
            if not violations:
                break
            
            # To fix a violation (u, v) with common neighbors C (|C|>=3),
            # we must remove at least one edge incident to u or v connecting to C.
            # Removing edge (u, w) where w in C reduces common(u, v) by 1.
            # It also affects other pairs.
            
            # Heuristic: Remove the edge (u, w) or (v, w) that is part of the most other violations.
            # Or simply remove a random one from the violation set.
            # For speed, let's remove an edge that appears in many violations.
            
            edge_freq = {}
            for u, v, common, cnt in violations:
                # Edges (u, w) for w in common, and (v, w) for w in common
                # Note: u and v are not in common.
                mask = common
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    mask ^= lsb
                    
                    e1 = (min(u, w), max(u, w))
                    e2 = (min(v, w), max(v, w))
                    
                    edge_freq[e1] = edge_freq.get(e1, 0) + 1
                    edge_freq[e2] = edge_freq.get(e2, 0) + 1
            
            if not edge_freq:
                break
                
            # Pick the edge with highest frequency
            best_edge = max(edge_freq, key=edge_freq.get)
            u, v = best_edge
            
            # Remove it
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Update edges list
                edges = [e for e in edges if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
                changed = True
        
        # Now graph is K2,3-free. Try to add edges back.
        # Greedy addition
        improved = True
        while improved:
            improved = False
            # Check all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    
                    # Check if adding (u, v) creates a K2,3
                    # New common neighbors of (x, y) will increase if x or y is u or v.
                    # Specifically:
                    # 1. Common(u, v) must be <= 2. (It is, since u, v not adjacent, but wait, adding edge doesn't change common neighbors of u and v directly, it changes common neighbors of pairs involving u or v).
                    # Let's just check all pairs (u, w) and (v, w) for w != u, v.
                    
                    valid = True
                    # Check pair (u, v) itself? No, u and v are not neighbors of each other in the "common neighbor" sense for the pair (u,v) unless we consider the edge being added.
                    # Wait, definition: K2,3 has two vertices u, v with 3 common neighbors.
                    # If we add edge (u, v), does u and v have common neighbors?
                    # The common neighbors of u and v are vertices w such that w~u and w~v.
                    # Adding edge (u,v) does NOT change the set of common neighbors of u and v.
                    # It DOES change the common neighbors of (u, w) for all w.
                    # New common neighbors of (u, w) = Old common neighbors of (u, w) + {v if v~w}.
                    # So we need: Old common(u, w) + (1 if v~w else 0) <= 2.
                    # And: Old common(v, w) + (1 if u~w else 0) <= 2.
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        
                        # Check pair (u, w)
                        cnt_uw = bin(adj[u] & adj[w]).count('1')
                        if (adj[v] >> w) & 1: # v is neighbor of w
                            if cnt_uw >= 2:
                                valid = False
                                break
                        
                        # Check pair (v, w)
                        cnt_vw = bin(adj[v] & adj[w]).count('1')
                        if (adj[u] >> w) & 1: # u is neighbor of w
                            if cnt_vw >= 2:
                                valid = False
                                break
                    
                    if valid:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        improved = True
        
        return adj, edges

    best_edges = []
    best_count = -1
    
    # Multiple restarts with random initializations
    # Start with dense random graphs and prune
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization: keep each edge with probability p
        # To get close to max, start with p=0.7 or 0.8
        p = 0.7
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Optimize
        adj, edges = validate_and_optimize(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [e[:] for e in edges]
            
            # If we found a very good solution, we might want to do more local search
            # But the greedy add in validate_and_optimize is already doing some work.
            
            # Try a "swap" local search step: remove one edge, try to add two
            # This is expensive, so only do it if we have time and haven't hit a high score
            if len(edges) < 105 and time.time() < deadline - 10.0:
                # Simple swap search
                improved_swap = True
                while improved_swap and time.time() < deadline - 5.0:
                    improved_swap = False
                    for i in range(len(edges)):
                        if time.time() > deadline - 5.0: break
                        u, v = edges[i]
                        # Temporarily remove
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        temp_edges = [e for j, e in enumerate(edges) if j != i]
                        
                        # Try to add any edge
                        added = False
                        for x in range(n):
                            for y in range(x + 1, n):
                                if (adj[x] >> y) & 1:
                                    continue
                                # Check validity of adding (x, y)
                                valid = True
                                for w in range(n):
                                    if w == x or w == y:
                                        continue
                                    cnt_xw = bin(adj[x] & adj[w]).count('1')
                                    if (adj[y] >> w) & 1:
                                        if cnt_xw >= 2:
                                            valid = False
                                            break
                                    cnt_yw = bin(adj[y] & adj[w]).count('1')
                                    if (adj[x] >> w) & 1:
                                        if cnt_yw >= 2:
                                            valid = False
                                            break
                                if valid:
                                    adj[x] |= (1 << y)
                                    adj[y] |= (1 << x)
                                    temp_edges.append([x, y])
                                    added = True
                                    break
                            if added:
                                break
                        
                        if added and len(temp_edges) > len(edges):
                            edges = temp_edges
                            improved_swap = True
                        else:
                            # Revert
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            # Note: temp_edges was a copy, so we don't need to revert edges list explicitly if we didn't assign it
                            # But we modified adj. We reverted adj.
                            # We didn't modify edges.
                            pass
                            
    return best_edges
