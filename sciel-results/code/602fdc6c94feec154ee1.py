import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    def numpy_greedy_build(seed_rng):
        """
        Build a K2,3-free graph using vectorized NumPy common-neighbor checks.
        adj: n x n binary matrix
        common: n x n matrix where common[u,v] = |N(u) & N(v)|
        """
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            # Check if adding (u, v) violates K2,3
            # 1. For all w in N(u), common[w, v] must be <= 1
            # 2. For all w in N(v), common[u, w] must be <= 1
            # 3. common[u, v] must be <= 2 (which is always true since we check before adding)
            
            # Get neighbors
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            ok = True
            
            if len(nu) > 0:
                # Check common[nu, v]
                if np.max(common[nu, v]) > 1:
                    ok = False
            
            if ok and len(nv) > 0:
                # Check common[u, nv]
                if np.max(common[u, nv]) > 1:
                    ok = False
            
            if ok:
                adj[u, v] = 1
                adj[v, u] = 1
                
                # Update common counts
                # For each w in N(u), common[w, v] += 1
                # For each w in N(v), common[u, w] += 1
                # Note: u and v are not in N(u) or N(v) initially, but after adding, 
                # we need to be careful. 
                # Actually, the update logic:
                # When adding edge (u,v), for any w in N(u), w and v now have one more common neighbor (u).
                # So common[w, v] += 1 for all w in N(u).
                # Similarly, common[u, w] += 1 for all w in N(v).
                
                if len(nu) > 0:
                    common[nu, v] += 1
                    common[v, nu] += 1
                
                if len(nv) > 0:
                    common[u, nv] += 1
                    common[nv, u] += 1
                
                # Also, u and v themselves:
                # common[u, v] increases? No, common[u,v] counts common neighbors.
                # Adding edge (u,v) does not change the set of common neighbors of u and v.
                # It only changes the common neighbor counts of pairs involving u or v.
        
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def can_add_numpy(adj, common, u, v):
        if adj[u, v]:
            return False
        
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        if len(nu) > 0:
            if np.max(common[nu, v]) > 1:
                return False
        
        if len(nv) > 0:
            if np.max(common[u, nv]) > 1:
                return False
        
        return True

    def add_edge_numpy(adj, common, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # Exclude u and v from the lists if they appear (they shouldn't for simple graph, but safe)
        # Actually, adj[u,u] is 0, so u is not in nu.
        
        if len(nu) > 0:
            common[nu, v] += 1
            common[v, nu] += 1
        
        if len(nv) > 0:
            common[u, nv] += 1
            common[nv, u] += 1

    def remove_edge_numpy(adj, common, u, v):
        # Must be careful: we need the neighbor lists BEFORE removal
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        adj[u, v] = 0
        adj[v, u] = 0
        
        if len(nu) > 0:
            common[nu, v] -= 1
            common[v, nu] -= 1
        
        if len(nv) > 0:
            common[u, nv] -= 1
            common[nv, u] -= 1

    def greedy_fill_numpy(adj, common):
        added = False
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    if can_add_numpy(adj, common, i, j):
                        add_edge_numpy(adj, common, i, j)
                        added = True
        return added

    def local_search_numpy(adj, common, time_limit):
        best_adj = adj.copy()
        best_common = common.copy()
        best_count = count_edges(adj)
        
        temperature = 5.0
        cooling = 0.99
        min_temp = 0.1
        
        while time.time() < time_limit and temperature > min_temp:
            # 1. Try to add any possible edge
            added = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i, j]:
                        if can_add_numpy(adj, common, i, j):
                            add_edge_numpy(adj, common, i, j)
                            added = True
            
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj.copy()
                best_common = common.copy()
            
            if not added:
                # 2. Try swap moves: remove (u1,v1), add (u2,v2)
                move_made = False
                
                # Get existing edges
                existing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            existing.append((i, j))
                
                if not existing:
                    break
                
                # Try a few random swaps
                for _ in range(5):
                    u1, v1 = rng.choice(existing)
                    # Pick a random non-edge
                    non_edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not adj[i, j]:
                                non_edges.append((i, j))
                    
                    if not non_edges:
                        break
                    
                    u2, v2 = rng.choice(non_edges)
                    
                    # Try swap
                    remove_edge_numpy(adj, common, u1, v1)
                    if can_add_numpy(adj, common, u2, v2):
                        add_edge_numpy(adj, common, u2, v2)
                        move_made = True
                        # Accept with some probability if count didn't increase (it stays same)
                        # Actually, swap keeps count same. We accept to explore.
                        # But we need to ensure we don't get stuck in bad swaps.
                        # Let's just accept if it's valid.
                        break
                    else:
                        # Restore
                        add_edge_numpy(adj, common, u1, v1)
                
                if not move_made:
                    # 3. Random edge removal to escape local minimum
                    u1, v1 = rng.choice(existing)
                    remove_edge_numpy(adj, common, u1, v1)
                    # Try to add something
                    greedy_fill_numpy(adj, common)
            
            temperature *= cooling
            
            if time.time() > time_limit:
                break

        return best_adj, best_common, best_count

    best_edges = fallback()
    best_count = len(best_edges)
    
    # Multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with random density
        p = rng.choice([0.3, 0.5, 0.7])
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        # Random initialization
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if rng.random() < p:
                if can_add_numpy(adj, common, u, v):
                    add_edge_numpy(adj, common, u, v)
        
        # Greedy fill
        greedy_fill_numpy(adj, common)
        
        current_count = count_edges(adj)
        
        # Local search
        adj, common, current_count = local_search_numpy(adj, common, deadline - 1.0)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = adj_to_edges(adj)
    
    # Final verification
    verified = verify(best_edges)
    if verified is not None:
        return verified
    else:
        return fallback()
