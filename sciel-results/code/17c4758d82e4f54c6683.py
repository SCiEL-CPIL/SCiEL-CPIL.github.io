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
    
    # Precompute neighbor lists as lists of indices for faster iteration in can_add
    # We will update these lists on add/remove
    
    def get_neighbors_list(u):
        return np.nonzero(adj[u])[0].tolist()

    # To speed up can_add, we can maintain a list of neighbors for each vertex
    # But updating lists is O(n). Updating adj matrix is O(1).
    # Given n=20, computing neighbors on the fly with np.nonzero is acceptable,
    # but let's try to optimize the check itself.
    
    def can_add(u, v):
        """Check if edge (u,v) can be added to current graph."""
        if adj[u, v]:
            return False
        # Condition 1: common[u, v] <= 2
        if common[u, v] > 2:
            return False
        
        # Condition 2: For all w in N(v), common[u, w] <= 1
        # We need to check max(common[u, w]) for w in N(v)
        # Using numpy is fast for max, but we need indices of N(v)
        # Let's use a loop for n=20, it's very fast in Python if optimized
        # Or use np.max on the subset
        
        # Fast path: check if any common[u, w] > 1 for w in N(v)
        # We can precompute a boolean mask of neighbors? No, adj changes.
        
        # Let's stick to the numpy approach but minimize overhead
        nv = np.nonzero(adj[v])[0]
        if len(nv) > 0:
            if np.max(common[u, nv]) > 1:
                return False
        
        nu = np.nonzero(adj[u])[0]
        if len(nu) > 0:
            if np.max(common[v, nu]) > 1:
                return False
                
        return True

    def perform_add(u, v):
        """Add edge (u,v) to adj and common."""
        # Get neighbors BEFORE modification
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

    def perform_remove(u, v):
        """Remove edge (u,v) from adj and common."""
        # Get neighbors BEFORE modification
        # Note: u is in N(v) and v is in N(u) currently
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
    # We will use a multi-restart strategy or a single long run with kicks.
    # The inspiration used a single run with kicks.
    
    current_count = best_count
    no_improve_steps = 0
    kick_threshold = 100
    
    # Pre-generate a list of all pairs for random selection
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_pairs)
    
    while time.time() < deadline:
        # Try to add a random edge
        # To avoid bias, we can cycle through pairs or pick randomly
        # Picking randomly from non-edges is better
        
        # Efficiently pick a random non-edge
        # We can just pick a random pair and check if it's an edge
        # Or maintain a list of non-edges? Updating that list is costly.
        # Just pick random u, v.
        
        u, v = random.sample(range(n), 2)
        
        if not adj[u, v] and can_add(u, v):
            perform_add(u, v)
            current_count += 1
            no_improve_steps = 0
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj.copy()
        else:
            no_improve_steps += 1
            
            # Kick mechanism: remove a random edge to allow re-adding
            if no_improve_steps > kick_threshold:
                # Find a random edge to remove
                # Pick random u, v until we find an edge
                removed = False
                for _ in range(10):
                    ru, rv = random.sample(range(n), 2)
                    if adj[ru, rv]:
                        perform_remove(ru, rv)
                        current_count -= 1
                        removed = True
                        no_improve_steps = 0
                        break
                
                if not removed:
                    # Graph is empty? Unlikely.
                    break

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])

    return best_edges
