"""Construct a C4-free graph with high minimum degree using finite geometry / combinatorial designs."""
import time
import random
from itertools import combinations

def _is_c4_free(n, edges):
    """Check if graph is C4-free using bitsets for speed."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common - if >= 2, C4 exists
            c = common
            if c & (c - 1):  # More than one bit set
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        # Random greedy: start with empty graph, add edges randomly
        for seed in range(500):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            edges = []
            adj = [0] * target_n
            deg = [0] * target_n
            
            # Generate all possible edges
            all_edges = list(combinations(range(target_n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                
                # Check if adding (u,v) creates a C4: u-v-w-x-u
                # Need w in N(v), x in N(u), w~x, w!=x, w!=u, x!=v
                creates_c4 = False
                nv = adj[v]
                nu = adj[u]
                
                # Iterate over neighbors of v
                w_mask = nv
                while w_mask:
                    w_bit = w_mask & (-w_mask)
                    w = w_bit.bit_length() - 1
                    w_mask ^= w_bit
                    
                    if w == u:
                        continue
                    
                    # Check if any neighbor of u (that isn't v or w) is adjacent to w
                    x_mask = nu & ~w_bit & ~(1 << v)
                    if x_mask & adj[w]:
                        creates_c4 = True
                        break
                
                if not creates_c4:
                    edges.append([u, v])
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            # Check minimum degree
            if min(deg) >= min_deg_needed:
                # Verify C4-free
                if _is_c4_free(target_n, edges):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges
                    break  # Found a valid graph for this n, move to next n
        
        # If we found a valid graph for this target_n, we can stop trying smaller n
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
