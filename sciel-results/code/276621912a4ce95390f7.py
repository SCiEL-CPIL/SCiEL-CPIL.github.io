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
                # 0 means i -> j, 1 means j -> i
                # i -> j iff (j - i) % p is a square
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def adj_to_orientation(adj, n):
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] & (1 << j)):
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

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

    def local_search(adj, n, time_budget):
        # adj is a list of bitmasks
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
            else:
                no_improve_count += 1
                if delta == 0 and random.random() < 0.1:
                    if u_to_v:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                
            if no_improve_count > max_no_improve:
                # Restart local search with a perturbation? Or just break and let outer loop handle?
                # For shrinking, we want to fix the specific bad sets.
                # If stuck, break.
                break

        return best_adj if best_cost == 0 else None

    def shrink_tournament(adj, n, time_per_step):
        # Try to remove vertices one by one
        current_adj = adj[:]
        current_n = n
        
        while current_n > 31: # Stop at lower bound
            if time.time() > start_time + time_limit - 5:
                break
            
            # Try to remove a random vertex
            # Maybe try all vertices? No, too slow.
            # Pick a few random candidates
            
            improved = False
            for _ in range(10): # Try 10 random vertices
                if time.time() > start_time + time_limit - 5:
                    break
                    
                v_to_remove = random.randint(0, current_n - 1)
                
                # Construct new adj without v_to_remove
                new_n = current_n - 1
                new_adj = [0] * new_n
                
                # Mapping from old index to new index
                # Or just build directly
                idx_map = {}
                new_idx = 0
                for i in range(current_n):
                    if i == v_to_remove:
                        continue
                    idx_map[i] = new_idx
                    new_idx += 1
                
                for i in range(current_n):
                    if i == v_to_remove:
                        continue
                    ni = idx_map[i]
                    mask = 0
                    for j in range(current_n):
                        if i == j or j == v_to_remove:
                            continue
                        if (current_adj[i] & (1 << j)):
                            nj = idx_map[j]
                            mask |= (1 << nj)
                    new_adj[ni] = mask
                
                # Check if valid
                cost = count_bad_sets(new_adj, new_n)
                
                if cost == 0:
                    # Success!
                    current_adj = new_adj
                    current_n = new_n
                    improved = True
                    break
                else:
                    # Try to fix with local search
                    # Time budget for this repair
                    budget = min(time_per_step, max(0.5, (start_time + time_limit - 5 - time.time()) / 10))
                    res = local_search(new_adj, new_n, budget)
                    if res is not None:
                        current_adj = res
                        current_n = new_n
                        improved = True
                        break
            
            if not improved:
                # Could not remove any vertex in this round?
                # Maybe try different vertices in next round?
                # Or give up on shrinking further?
                # Let's try a few more rounds
                # If we fail 3 times in a row, break?
                # For simplicity, if we can't remove one, we might be stuck.
                # But maybe a different vertex works.
                # We tried 10 random. Let's try to be smarter or just stop.
                # Given time constraints, stopping is safer.
                break
        
        return {'n': current_n, 'orientation': adj_to_orientation(current_adj, current_n)}

    # Initial solution: Paley 83
    paley = get_paley_83()
    
    # Convert Paley orientation to adj
    n0 = paley['n']
    adj0 = [0] * n0
    k = 0
    for i in range(n0):
        for j in range(i + 1, n0):
            bit = paley['orientation'][k]
            k += 1
            if bit == 0:
                adj0[i] |= (1 << j)
            else:
                adj0[j] |= (1 << i)
    
    # Try to shrink
    # Allocate time for shrinking
    # If we run out of time, return the best we have (which is at least Paley 83)
    
    result = shrink_tournament(adj0, n0, 5.0)
    
    # If result is worse than Paley (shouldn't happen), return Paley
    if result['n'] > n0:
        return paley
        
    return result
