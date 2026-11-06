"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = 40
    
    def time_ok():
        return time.time() < deadline
    
    # Bitset-based adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6 using fast bitset ops."""
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na_excl = adj[a] & ~(1 << u)
            
            # d in N(v)\{u, a}
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                Nd_excl = adj[d] & ~(1 << v)
                
                # candidates for b: N(a) \ {u, d, v}
                # candidates for c: N(d) \ {v, a, u}
                B = Na_excl & ~(1 << d) & ~(1 << v)
                C = Nd_excl & ~(1 << a) & ~(1 << u)
                
                # For each b in B, check if adj[b] intersects C (and b not in C)
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    if adj[b] & C & ~(1 << b):
                        return True
        
        return False

    def greedy_fill(edges, adj, edge_set, non_edges, max_time=None):
        """Greedily add edges from non_edges if they don't create C6."""
        if max_time is None:
            max_time = deadline
        remaining = non_edges[:]
        random.shuffle(remaining)
        added_any = True
        while added_any and time.time() < max_time:
            added_any = False
            new_remaining = []
            for u, v in remaining:
                if (u, v) in edge_set:
                    continue
                if would_create_c6(adj, u, v):
                    new_remaining.append((u, v))
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    added_any = True
            remaining = new_remaining
        return edges, adj, edge_set

    best_edges = []
    best_count = 0
    
    # Initial structure: 8 disjoint K5s
    base_edges = []
    for k in range(8):
        base_edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    # Multiple restarts
    for restart in range(50):
        if not time_ok():
            break
        
        current_edges = base_edges[:]
        current_edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)
        
        # Define non-edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        non_edges = [e for e in all_edges if e not in current_edge_set]
        
        # Greedy fill
        current_edges, current_adj, current_edge_set = greedy_fill(
            current_edges, current_adj, current_edge_set, non_edges
        )
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # If we hit a high score, try to polish with more restarts or just break if time is low
            if best_count >= 105:
                # We have a good candidate. Try a few more quick restarts to see if we can beat it
                # or just continue to next restart if time allows
                pass

    return best_edges
# EVOLVE-BLOCK-END
