import time
import random
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free.
    Returns True if it IS K3,3-free (valid), False otherwise.
    Uses bit_count() for speed.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        for j in range(i + 1, n):
            common_ij = adj_bitmasks[i] & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Only need to check triples that include u or v (since only their neighborhoods change).
    """
    n = len(adj_bitmasks)
    # After adding (u,v):
    # - u gains neighbor v
    # - v gains neighbor u
    # New K3,3 can only involve triples where at least one of the vertices is u or v
    # AND at least one of the common neighbors is the new edge endpoint.
    
    # Case 1: Triple includes u and v and some k
    # N(u) ∩ N(v) ∩ N(k) must have < 3
    # N(u) now includes v, N(v) now includes u
    # For triple (u, v, k): common = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(k)
    # = (N(u) ∩ N(v) ∩ N(k)) ∪ (N(u) ∩ {u} ∩ N(k)) ∪ ({v} ∩ N(v) ∩ N(k)) ∪ ... 
    # Simpler: just check the new common neighbors
    # The only new common neighbors for triples involving u and v are:
    # - For triple (u,v,k): v is in N(u), u is in N(v), so u∩v∩N(k) includes k if k~u and k~v
    #   Actually let's just do a targeted check
    
    # After adding edge (u,v), the only new K3,3s must involve the new edge.
    # A K3,3 with bipartition {a,b,c} and {x,y,z} where (u,v) is one of the 9 edges.
    # WLOG, suppose u is in {a,b,c} and v is in {x,y,z}.
    # Then we need 2 more vertices from {a,b}\{u} and 2 more from {x,y,z}\{v}
    # such that all cross-edges exist.
    
    # Alternative simpler approach: check all triples that include u or v
    # For triples containing u: (u, i, j) where i < j, i≠u, j≠u
    #   Check if |N(u) ∩ N(i) ∩ N(j)| >= 3 with new N(u)
    # For triples containing v: (v, i, j) where i < j, i≠v, j≠v  
    #   Check if |N(v) ∩ N(i) ∩ N(j)| >= 3 with new N(v)
    # But we can optimize: only triples where the new neighbor matters
    # For triple (u, i, j): new common neighbors come from v being in N(u)
    #   So we need v ∈ N(i) and v ∈ N(j), i.e., i~v and j~v
    #   Then common = (N(u) ∩ N(i) ∩ N(j)) ∪ {v} if v~i and v~j
    #   So we check: if v~i and v~j, then |N(u) ∩ N(i) ∩ N(j) ∪ {v}| >= 3
    #   Which is: |N(u) ∩ N(i) ∩ N(j)| >= 3 OR (v~i AND v~j AND |N(u) ∩ N(i) ∩ N(j)| >= 2)
    
    # Let me just do the direct check for affected triples
    new_u_mask = adj_bitmasks[u] | (1 << v)
    new_v_mask = adj_bitmasks[v] | (1 << u)
    
    # Check triples containing u (but not v, since those are covered when we check v's side)
    for i in range(n):
        if i == u:
            continue
        for j in range(i + 1, n):
            if j == u or j == v:
                continue
            # Triple (u, i, j) - but need u < i < j or handle ordering
            # Actually let's just check all pairs (i,j) with i≠u, j≠u, i<j
            pass
    
    # Simpler: just check all triples that include u or v
    # Triple (u, i, j) for all i < j, i≠u, j≠u
    for i in range(n):
        if i == u:
            continue
        for j in range(i + 1, n):
            if j == u:
                continue
            # Triple is {u, i, j} - check common neighbors
            # With new adjacency: N(u) = new_u_mask, N(i) = adj[i], N(j) = adj[j]
            common = new_u_mask & adj_bitmasks[i] & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False
    
    # Triple (v, i, j) for all i < j, i≠v, j≠v
    for i in range(n):
        if i == v:
            continue
        for j in range(i + 1, n):
            if j == v:
                continue
            common = new_v_mask & adj_bitmasks[i] & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False
    
    return True

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = 0
    best_adj_bitmasks = None
    
    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def edges_to_bitmasks(edges):
        bitmasks = [0] * n
        for u, v in edges:
            bitmasks[u] |= (1 << v)
            bitmasks[v] |= (1 << u)
        return bitmasks

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (bitmasks[i] & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def local_search(initial_bitmasks):
        bitmasks = initial_bitmasks[:]
        current_edges = sum(b.bit_count() for b in bitmasks) // 2
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]
        
        stuck_count = 0
        while time.time() < deadline:
            moved = False
            
            # Try to add an edge
            non_edges = get_non_edges(bitmasks)
            
            if non_edges:
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]
                        break
            
            if not moved:
                # Try swap: remove edge (a,b), add edge (c,d)
                all_edges = get_all_edges(bitmasks)
                if all_edges and non_edges:
                    # Try a few random swaps
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        # Check if removing (a,b) and adding (c,d) is valid
                        # First remove (a,b)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Then check if adding (c,d) is valid in bm2
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            # Verify full validity
                            if has_k33_fast(bm2):
                                bitmasks = bm2
                                current_edges = sum(b.bit_count() for b in bitmasks) // 2
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_bitmasks = bitmasks[:]
                                break
                
                if not moved:
                    # Try removing a random edge to escape
                    if all_edges:
                        u, v = random.choice(all_edges)
                        bitmasks[u] &= ~(1 << v)
                        bitmasks[v] &= ~(1 << u)
                        current_edges -= 1
                        moved = True
            
            if not moved:
                stuck_count += 1
                if stuck_count > 30:
                    break
            else:
                stuck_count = 0
                
        return best_local_bitmasks, best_local_edges

    # Run multiple restarts
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random sparse graph
        init_bitmasks = [0] * n
        p = 0.15 + 0.03 * r
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    init_bitmasks[i] |= (1 << j)
                    init_bitmasks[j] |= (1 << i)
        
        # Repair: Remove random edges until valid
        valid = has_k33_fast(init_bitmasks)
        attempts = 0
        while not valid and attempts < 500:
            all_edges = get_all_edges(init_bitmasks)
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            init_bitmasks[u] &= ~(1 << v)
            init_bitmasks[v] &= ~(1 << u)
            valid = has_k33_fast(init_bitmasks)
            attempts += 1
        
        if not valid:
            continue
            
        # Run local search
        local_best_bitmasks, local_best_edges = local_search(init_bitmasks)
        
        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks

    # If no valid graph found (should not happen), return empty
    if best_adj_bitmasks is None:
        return []
        
    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Verify K3,3 free
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
