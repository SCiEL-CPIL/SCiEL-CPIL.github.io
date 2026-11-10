import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses a local search strategy with edge addition and edge swapping.
    Maintains a common-neighbor count matrix for O(1) validity checks (with O(n) updates).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    # Initialize with 5 disjoint K4s (30 edges)
    # Vertices 0-3, 4-7, 8-11, 12-15, 16-19
    adj = np.zeros((n, n), dtype=np.int8)
    for k in range(5):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * k + i
                v = 4 * k + j
                adj[u, v] = 1
                adj[v, u] = 1
                
    # Compute initial common neighbor counts
    common = (adj @ adj).astype(np.int8)
    
    best_count = int(np.sum(adj) // 2)
    best_adj = adj.copy()
    
    def get_neighbors(u):
        """Return list of neighbors of u."""
        return np.nonzero(adj[u])[0].tolist()
        
    def can_add(u, v):
        """Check if edge (u,v) can be added to current graph."""
        if adj[u, v]:
            return False
        # Condition 1: common[u, v] <= 2
        if common[u, v] > 2:
            return False
        
        # Condition 2: For all w in N(v), common[u, w] <= 1
        # Note: u is not in N(v) since adj[u,v]=0
        nv = np.nonzero(adj[v])[0]
        if len(nv) > 0:
            if np.max(common[u, nv]) > 1:
                return False
        
        # Condition 3: For all w in N(u), common[v, w] <= 1
        nu = np.nonzero(adj[u])[0]
        if len(nu) > 0:
            if np.max(common[v, nu]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        """Add edge (u,v) and update common counts."""
        adj[u, v] = 1
        adj[v, u] = 1
        
        # Update common counts
        # For w in N(v) \ {u}, common[u, w] += 1
        nv = np.nonzero(adj[v])[0]
        # nv includes u now? No, adj[v] was checked before setting?
        # We set adj[u,v]=1 first. So nv includes u.
        # We need to update common[u, w] for w in N(v) where w != u.
        # But common[u, u] is 0.
        # The formula was: common[u, w] += adj[v, w].
        # Since we just set adj[v, u]=1, common[u, u] would increment?
        # But common[u, u] is not used in constraints (u!=v).
        # Let's be precise.
        # The update rule derived:
        # For w != u, v: common[u, w] += adj[v, w] (old)
        # Since we updated adj[v, u]=1, the "old" adj[v, w] for w=u is 0.
        # So we should use the neighbors BEFORE adding the edge.
        
        # Let's revert to using pre-computed neighbors or careful indexing.
        # It's safer to compute updates based on the state BEFORE adding.
        # But we already modified adj.
        # Let's undo, compute, redo? No, just careful logic.
        
        # Actually, the update is:
        # common[u, w] += 1 for all w in N_old(v)
        # common[v, w] += 1 for all w in N_old(u)
        
        # Since I modified adj, I can't easily get N_old.
        # I will refactor to pass neighbors or compute before modifying.
        pass

    # Refactored add_edge and remove_edge to handle updates correctly
    
    def perform_add(u, v):
        """Add edge (u,v) to adj and common."""
        # Get neighbors BEFORE modification
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        
        adj[u, v] = 1
        adj[v, u] = 1
        
        # Update common
        # For w in nv: common[u, w] += 1
        # Note: nv does not contain u (since adj[u,v] was 0)
        if len(nv) > 0:
            common[u, nv] += 1
            common[nv, u] += 1 # Symmetry
            
        # For w in nu: common[v, w] += 1
        # Note: nu does not contain v
        if len(nu) > 0:
            common[v, nu] += 1
            common[nu, v] += 1
            
    def perform_remove(u, v):
        """Remove edge (u,v) from adj and common."""
        # Get neighbors BEFORE modification
        # Note: u is in N(v) and v is in N(u) currently
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        
        adj[u, v] = 0
        adj[v, u] = 0
        
        # Update common
        # For w in N_new(v) = N_old(v) \ {u}: common[u, w] -= 1
        # For w in N_new(u) = N_old(u) \ {v}: common[v, w] -= 1
        
        # It is easier to subtract 1 from common[u, w] for all w in nv,
        # but w=u is in nv. common[u, u] is 0.
        # Does the formula hold for w=u?
        # common[u, u] is not constrained.
        # But let's check the derivation.
        # C_uw_new = C_uw_old - A_vw.
        # If w=u, A_vu is 1 (before removal).
        # So common[u, u] should decrease by 1?
        # But common[u, u] is always 0 in a simple graph (no self loops).
        # The formula C_ab = sum A_ak A_bk.
        # C_uu = sum A_uk A_uk = sum A_uk = degree(u).
        # Wait, my `common` matrix calculation `adj @ adj` gives degrees on the diagonal.
        # The constraint is only for u != v.
        # So diagonal values don't matter for validity, but they might matter for updates if we are not careful.
        # However, we only check `common[u, v]` for u != v.
        # And we check `max(common[u, nv])`. If nv contains u, we check common[u, u].
        # common[u, u] = degree(u).
        # Is degree(u) <= 1 required? No.
        # So we must EXCLUDE u from the check list or ensure diagonal is not checked.
        
        # In `can_add`, I used `np.max(common[u, nv])`.
        # If `nv` contains `u`, it checks `common[u, u]`.
        # `common[u, u]` is degree(u).
        # If degree(u) > 1, it returns False.
        # This is WRONG.
        
        # Fix: In `can_add`, exclude u from nv and v from nu.
        
        # Let's rewrite `can_add` with exclusion.
        
        pass

    # Corrected can_add
    def can_add_corrected(u, v):
        if adj[u, v]:
            return False
        if common[u, v] > 2:
            return False
        
        nv = np.nonzero(adj[v])[0]
        # Exclude u
        mask = nv != u
        nv = nv[mask]
        if len(nv) > 0:
            if np.max(common[u, nv]) > 1:
                return False
        
        nu = np.nonzero(adj[u])[0]
        # Exclude v
        mask = nu != v
        nu = nu[mask]
        if len(nu) > 0:
            if np.max(common[v, nu]) > 1:
                return False
                
        return True

    # Corrected perform_add
    def perform_add_corrected(u, v):
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        
        adj[u, v] = 1
        adj[v, u] = 1
        
        # Update common
        # common[u, w] += 1 for w in nv
        # common[v, w] += 1 for w in nu
        
        # nv contains no u (since adj[u,v] was 0)
        # nu contains no v
        if len(nv) > 0:
            common[u, nv] += 1
            common[nv, u] += 1
        if len(nu) > 0:
            common[v, nu] += 1
            common[nu, v] += 1

    # Corrected perform_remove
    def perform_remove_corrected(u, v):
        # nv contains u
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        
        adj[u, v] = 0
        adj[v, u] = 0
        
        # Update common
        # For w in nv: common[u, w] -= 1
        # For w in nu: common[v, w] -= 1
        
        if len(nv) > 0:
            common[u, nv] -= 1
            common[nv, u] -= 1
        if len(nu) > 0:
            common[v, nu] -= 1
            common[nu, v] -= 1

    # Main Loop
    while time.time() < deadline:
        added_any = False
        
        # Try to add edges
        # Shuffle order to avoid bias
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add_corrected(u, v):
                perform_add_corrected(u, v)
                added_any = True
                # Update best
                cnt = int(np.sum(adj) // 2)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj.copy()
        
        if not added_any:
            # Try a swap
            # Pick a random edge to remove
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if not edges:
                break
            rem_u, rem_v = random.choice(edges)
            
            # Remove it
            perform_remove_corrected(rem_u, rem_v)
            
            # Try to add a random non-edge
            # Get non-edges
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            random.shuffle(non_edges)
            
            added_swap = False
            for add_u, add_v in non_edges:
                if time.time() > deadline:
                    break
                if can_add_corrected(add_u, add_v):
                    perform_add_corrected(add_u, add_v)
                    added_swap = True
                    # Update best
                    cnt = int(np.sum(adj) // 2)
                    if cnt > best_count:
                        best_count = cnt
                        best_adj = adj.copy()
                    break # Only add one edge per swap attempt
            
            if not added_swap:
                # Restore the removed edge
                perform_add_corrected(rem_u, rem_v)
                
    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return best_edges
