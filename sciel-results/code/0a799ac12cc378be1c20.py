import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = set()
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) in edges: return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((u, v))
        
    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) not in edges: return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.remove((u, v))
        
    def get_book(u, v):
        return np.sum(adj[u] & adj[v])
        
    def get_cost():
        cost = 0
        for u, v in edges:
            b = get_book(u, v)
            cost += b * b
        return cost

    # Initialize
    random.seed(42)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    for u, v in all_possible[:target_edges]:
        add_edge(u, v)
        
    current_cost = get_cost()
    
    # Simulated Annealing
    T = 1000.0
    T_min = 1.0
    cooling = 0.9995
    steps = 100000
    
    for step in range(steps):
        T *= cooling
        if T < T_min:
            break
            
        # Random Move
        r = random.random()
        
        if r < 0.33:
            # Add edge
            if len(edges) >= target_edges: continue
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if (u, v) in edges or (v, u) in edges: continue
                break
            
            # Calculate delta
            # We need to compute cost change
            # To avoid full re-calc, we compute delta
            # But for simplicity in this draft, let's just do full re-calc if n is small?
            # 40 vertices, 401 edges. Full re-calc is 401 * (40+40) ops ~ 32000 ops.
            # 100000 steps * 32000 = 3.2e9 ops. Too slow in Python.
            
            # Must use incremental delta.
            
            # Let's implement delta_cost for add/remove/swap properly.
            
            # For now, let's use a simpler heuristic: 
            # Just use the previous code's logic but with SA.
            # The previous code's delta logic was correct.
            
            pass # Placeholder
            
    # This implementation is getting complex. Let's stick to the incremental delta approach from the thought process.
    
    # Re-implementing with incremental delta
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = set()
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((u, v))
        
    def remove_edge(u, v):
        if u > v: u, v = v, u
        adj[u][v] = 0
        adj[v][u] = 0
        edges.remove((u, v))
        
    def get_book(u, v):
        return np.sum(adj[u] & adj[v])
        
    def delta_add(u, v):
        if (u,v) in edges or (v,u) in edges: return 0
        common_mask = adj[u] & adj[v]
        common_idx = np.where(common_mask > 0)[0]
        
        dc = 0
        # New edge (u,v)
        b_new = len(common_idx)
        dc += b_new * b_new
        
        # Affected edges (u,w) and (v,w)
        for w in common_idx:
            # (u,w)
            b_uw_old = get_book(u, w)
            b_uw_new = b_uw_old + 1
            dc += (b_uw_new * b_uw_new - b_uw_old * b_uw_old)
            
            # (v,w)
            b_vw_old = get_book(v, w)
            b_vw_new = b_vw_old + 1
            dc += (b_vw_new * b_vw_new - b_vw_old * b_vw_old)
            
        return dc

    def delta_remove(u, v):
        if (u,v) not in edges and (v,u) not in edges: return 0
        common_mask = adj[u] & adj[v]
        common_idx = np.where(common_mask > 0)[0]
        
        dc = 0
        # Old edge (u,v)
        b_old = get_book(u, v)
        dc -= b_old * b_old
        
        # Affected edges (u,w) and (v,w)
        for w in common_idx:
            # (u,w)
            b_uw_old = get_book(u, w)
            b_uw_new = b_uw_old - 1
            dc += (b_uw_new * b_uw_new - b_uw_old * b_uw_old)
            
            # (v,w)
            b_vw_old = get_book(v, w)
            b_vw_new = b_vw_old - 1
            dc += (b_vw_new * b_vw_new - b_vw_old * b_vw_old)
            
        return dc

    # Initialize
    random.seed(42)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    for u, v in all_possible[:target_edges]:
        add_edge(u, v)
        
    current_cost = 0
    for u, v in edges:
        b = get_book(u, v)
        current_cost += b * b

    T = 500.0
    T_min = 0.1
    cooling = 0.999
    steps = 50000
    
    for step in range(steps):
        T *= cooling
        if T < T_min:
            break
            
        r = random.random()
        
        if r < 0.3:
            # Add
            if len(edges) == target_edges: continue
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if (u, v) in edges or (v, u) in edges: continue
                break
            dc = delta_add(u, v)
            if dc < 0 or random.random() < np.exp(-dc / T):
                add_edge(u, v)
                current_cost += dc
                
        elif r < 0.6:
            # Remove
            if len(edges) == 0: continue
            u, v = random.choice(list(edges))
            dc = delta_remove(u, v)
            if dc < 0 or random.random() < np.exp(-dc / T):
                remove_edge(u, v)
                current_cost += dc
                
        else:
            # Swap
            if len(edges) == 0: continue
            u1, v1 = random.choice(list(edges))
            
            # Find u2, v2
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                if (u2, v2) in edges or (v2, u2) in edges: continue
                if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1): continue
                break
            
            # Calculate delta for swap
            # Remove (u1,v1) then Add (u2,v2)
            # We can't just add deltas because the state changes.
            # However, we can simulate:
            # 1. Temporarily remove (u1,v1)
            # 2. Calculate delta_add for (u2,v2) on the new state
            # 3. Total delta = delta_remove(u1,v1) + delta_add_new(u2,v2)
            # But delta_add_new depends on the state after removal.
            
            # To do this correctly without modifying state twice:
            # We can calculate the cost of the graph with (u1,v1) removed and (u2,v2) added.
            # This is expensive.
            
            # Alternative: Just do remove, then add.
            # If remove is accepted, then try add.
            # This is not a single SA move, but it works.
            
            dc_rem = delta_remove(u1, v1)
            # Acceptance of remove
            if dc_rem < 0 or random.random() < np.exp(-dc_rem / T):
                remove_edge(u1, v1)
                current_cost += dc_rem
                
                # Now try to add (u2, v2)
                # It might be that (u2,v2) is now a valid edge (it was not in edges)
                dc_add = delta_add(u2, v2)
                if dc_add < 0 or random.random() < np.exp(-dc_add / T):
                    add_edge(u2, v2)
                    current_cost += dc_add
                else:
                    # Revert remove? No, we already removed it.
                    # If we don't add, we have 400 edges.
                    # We should probably revert the remove to keep edge count constant.
                    # Let's revert.
                    add_edge(u1, v1)
                    current_cost -= dc_rem

    # Finalize
    final_edges = list(edges)
    return [list(e) for e in final_edges]
