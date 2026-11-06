"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 20
    
    # Start with a random dense-ish graph or a structured one
    # Let's start with a random graph with p=0.5, then optimize
    random.seed(42)
    
    # Build adjacency as bitmasks for fast common-neighbor checks
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def is_valid_addition(u, v):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        # Check if adding u-v creates a K2,3
        # For any common neighbor w of u and v, we need to ensure
        # that u and v don't already have 2 other common neighbors
        # Actually, the constraint is: no two vertices have 3 common neighbors.
        # Adding edge (u,v) doesn't change common neighbors of any pair except
        # pairs involving u or v.
        # For pair (u,v): their common neighbors are (adj[u] & adj[v]).
        # Before adding, they had some common neighbors. After adding, u is a neighbor of v and vice versa,
        # but common neighbors of u and v are vertices w such that w~u and w~v.
        # Adding edge (u,v) does NOT change the set of common neighbors of (u,v).
        # Wait, it does affect pairs (u, w) for w in V\{u,v}.
        # If we add (u,v), then for any w, the common neighbors of (u,w) might change.
        # Specifically, v becomes a common neighbor of (u,w) if w~v.
        # So we need to check: for each w in N(u) ∪ N(v) (excluding u,v),
        # does adding (u,v) cause (u,w) or (v,w) to have 3 common neighbors?
        
        # More precisely:
        # After adding (u,v), for any vertex w != u,v:
        # - Common neighbors of (u,w) = (adj[u] & adj[w]) but now v is in adj[u].
        #   So new common neighbors of (u,w) = old_common(u,w) ∪ ({v} if w~v else {})
        # - Similarly for (v,w).
        # - Common neighbors of (u,v) = adj[u] & adj[v] (unchanged by adding the edge itself)
        
        # So we need to check:
        # 1. Common neighbors of (u,v) < 3: (adj[u] & adj[v]).bit_count() < 3
        # 2. For each w in N(u) (old): if w~v, then new common neighbors of (u,w) = old_common(u,w) + 1.
        #    So we need old_common(u,w) < 2 for all w in N(u) ∩ N(v).
        # 3. For each w in N(v) (old): if w~u, then new common neighbors of (v,w) = old_common(v,w) + 1.
        #    So we need old_common(v,w) < 2 for all w in N(v) ∩ N(u).
        
        # Note that N(u) ∩ N(v) = common neighbors of u and v.
        # So condition 2 and 3 are: for each w in (adj[u] & adj[v]), 
        #   (adj[u] & adj[w]).bit_count() < 2 and (adj[v] & adj[w]).bit_count() < 2
        
        # And condition 1: (adj[u] & adj[v]).bit_count() < 3
        
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() >= 3:
            return False
        
        # Check for each w in common_uv
        w_mask = common_uv
        while w_mask:
            lsb = w_mask & (-w_mask)
            w = lsb.bit_length() - 1
            w_mask ^= lsb
            if (adj[u] & adj[w]).bit_count() >= 2:
                return False
            if (adj[v] & adj[w]).bit_count() >= 2:
                return False
        
        return True
    
    def edge_count():
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    # Initialize with random graph
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.4:
                add_edge(u, v)
    
    # Local search: try adding edges
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for u in range(n):
            if time.time() > deadline - 20:
                break
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)) and is_valid_addition(u, v):
                    add_edge(u, v)
                    improved = True
    
    # Edge swap local search
    best_edges = edge_count()
    for iteration in range(500000):
        if time.time() > deadline - 10:
            break
        u1, v1 = random.randint(0, n-1), random.randint(0, n-1)
        if u1 == v1:
            continue
        if not (adj[u1] & (1 << v1)):
            continue
        u2, v2 = random.randint(0, n-1), random.randint(0, n-1)
        if u2 == v2:
            continue
        if adj[u2] & (1 << v2):
            continue
        # Check if removing u1-v1 and adding u2-v2 is valid
        # Temporarily remove u1-v1
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        if is_valid_addition(u2, v2):
            add_edge(u2, v2)
            # Keep the swap
            pass
        else:
            # Revert
            remove_edge(u2, v2) if (adj[u2] & (1 << v2)) else None
            # Actually, we didn't add it, so just restore u1-v1
            add_edge(u1, v1)
            continue
        
        # Now try to add back u1-v1
        if is_valid_addition(u1, v1):
            add_edge(u1, v1)
            # Net effect: swapped u1-v1 for u2-v2, but now we have both? No.
            # We removed u1-v1, added u2-v2. Then we tried to add u1-v1 back.
            # If we can add u1-v1 back, then we've effectively added u2-v2 without removing u1-v1.
            # That's a pure addition!
            improved = True
        else:
            # We've swapped: removed u1-v1, added u2-v2
            # This maintains edge count
            pass
    
    # Final polish: try all possible additions again
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)) and is_valid_addition(u, v):
                add_edge(u, v)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            mask ^= lsb
            if u < v:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
