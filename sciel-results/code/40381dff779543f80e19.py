import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    def get_paley_83():
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def has_dominator(a, b, c, d, adj):
        if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)):
            return True
        if (adj[b] & (1 << a)) and (adj[b] & (1 << c)) and (adj[b] & (1 << d)):
            return True
        if (adj[c] & (1 << a)) and (adj[c] & (1 << b)) and (adj[c] & (1 << d)):
            return True
        if (adj[d] & (1 << a)) and (adj[d] & (1 << b)) and (adj[d] & (1 << c)):
            return True
        return False

    def count_bad_sets(adj, n):
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not has_dominator(a, b, c, d, adj):
                            count += 1
        return count

    def greedy_init(n):
        # Initialize adj
        adj = [0] * n
        # List of all edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        # Track decided edges
        # Using a set of tuples is slow. 
        # We can use a 2D list of booleans or just check adj bits?
        # If adj[i] has bit j, then i->j is decided.
        # If adj[j] has bit i, then j->i is decided.
        # So edge (i, j) is decided if (adj[i] >> j) & 1 or (adj[j] >> i) & 1.
        
        for u, v in edges:
            score_uv = 0
            score_vu = 0
            
            # Check all pairs w, x
            # To speed up, we can precompute lists of vertices, but n is small.
            for w in range(n):
                if w == u or w == v: continue
                for x in range(w + 1, n):
                    if x == u or x == v: continue
                    
                    # Check if the other 4 edges are decided
                    # Edges: (u, w), (v, w), (u, x), (v, x), (w, x)
                    
                    # Helper to check if edge (a, b) is decided
                    # Returns 1 if a->b, -1 if b->a, 0 if not decided
                    def edge_state(a, b):
                        if (adj[a] >> b) & 1: return 1
                        if (adj[b] >> a) & 1: return -1
                        return 0

                    e_uw = edge_state(u, w)
                    e_vw = edge_state(v, w)
                    e_ux = edge_state(u, x)
                    e_vx = edge_state(v, x)
                    e_wx = edge_state(w, x)
                    
                    if e_uw == 0 or e_vw == 0 or e_ux == 0 or e_vx == 0 or e_wx == 0:
                        continue
                        
                    # All 5 other edges are decided.
                    # Now check validity for u->v and v->u
                    
                    # State 1: u -> v
                    # Set bits temporarily? No, just check conditions.
                    # We need to check if {u, v, w, x} has a dominator.
                    # Dominator can be u, v, w, x.
                    
                    # Check u
                    # u -> v is True.
                    # u -> w is (e_uw == 1)
                    # u -> x is (e_ux == 1)
                    if e_uw == 1 and e_ux == 1:
                        score_uv += 1
                        
                    # Check v
                    # v -> u is False (since u -> v).
                    # So v cannot be dominator.
                    
                    # Check w
                    # w -> u is (e_uw == -1)
                    # w -> v is (e_vw == -1)
                    # w -> x is (e_wx == 1)
                    if e_uw == -1 and e_vw == -1 and e_wx == 1:
                        score_uv += 1
                        
                    # Check x
                    # x -> u is (e_ux == -1)
                    # x -> v is (e_vx == -1)
                    # x -> w is (e_wx == -1)
                    if e_ux == -1 and e_vx == -1 and e_wx == -1:
                        score_uv += 1
                        
                    # State 2: v -> u
                    # Check u
                    # u -> v is False. So u cannot be dominator.
                    
                    # Check v
                    # v -> u is True.
                    # v -> w is (e_vw == 1)
                    # v -> x is (e_vx == 1)
                    if e_vw == 1 and e_vx == 1:
                        score_vu += 1
                        
                    # Check w
                    # w -> u is (e_uw == -1)
                    # w -> v is (e_vw == -1)
                    # w -> x is (e_wx == 1)
                    if e_uw == -1 and e_vw == -1 and e_wx == 1:
                        score_vu += 1
                        
                    # Check x
                    # x -> u is (e_ux == -1)
                    # x -> v is (e_vx == -1)
                    # x -> w is (e_wx == -1)
                    if e_ux == -1 and e_vx == -1 and e_wx == -1:
                        score_vu += 1
                        
            # Choose orientation
            if score_uv > score_vu:
                adj[u] |= (1 << v)
            elif score_vu > score_uv:
                adj[v] |= (1 << u)
            else:
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                else:
                    adj[v] |= (1 << u)
                    
        return adj

    def local_search(n, time_budget, initial_adj=None):
        if initial_adj is None:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                    else:
                        adj[j] |= (1 << i)
        else:
            adj = initial_adj[:]
            
        current_cost = count_bad_sets(adj, n)
        if current_cost == 0:
            return adj
        
        best_adj = adj[:]
        best_cost = current_cost
        
        end_time = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < end_time:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            u_to_v = (adj[u] & (1 << v)) != 0
            
            delta = 0
            others = [k for k in range(n) if k != u and k != v]
            
            for i in range(len(others)):
                w = others[i]
                for j in range(i + 1, len(others)):
                    x = others[j]
                    
                    old_good = has_dominator(u, v, w, x, adj)
                    
                    if u_to_v:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                    
                    new_good = has_dominator(u, v, w, x, adj)
                    
                    if u_to_v:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    
                    if old_good and not new_good:
                        delta += 1
                    elif not old_good and new_good:
                        delta -= 1
            
            if delta < 0:
                if u_to_v:
                    adj[u] &= ~(1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] &= ~(1 << u)
                
                current_cost += delta
                no_improve_count = 0
                
                if current_cost == 0:
                    return adj
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    
            elif delta == 0:
                if random.random() < 0.1:
                    if u_to_v:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart with greedy init
                adj = greedy_init(n)
                current_cost = count_bad_sets(adj, n)
                if current_cost == 0:
                    return adj
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                no_improve_count = 0

        return best_adj if best_cost == 0 else None

    for n in range(31, 83):
        if time.time() > start_time + time_limit - 10:
            break
        
        time_slice = 2.0 if n < 50 else 1.0
        if time.time() + time_slice > start_time + time_limit - 5:
            break
            
        # Try a few attempts with greedy init
        attempts = 3 if n < 50 else 1
        success = False
        for _ in range(attempts):
            if time.time() > start_time + time_limit - 5:
                break
            init_adj = greedy_init(n)
            res = local_search(n, time_slice / attempts, init_adj)
            if res is not None:
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (res[i] & (1 << j)):
                            orientation.append(0)
                        else:
                            orientation.append(1)
                return {'n': n, 'orientation': orientation}
                success = True
        if success:
            break

    return get_paley_83()
