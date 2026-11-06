import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    def time_left():
        return deadline - time.time()

    # 1. Build Paley Graph P(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append((i, j))
    
    n = q

    # Precompute K4s and Ind4s
    k4s = []
    ind4s = []
    
    # Enumerate all 4-sets
    # O(n^4) is fine for n=37
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)): continue
            for k in range(j+1, n):
                if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                for l in range(k+1, n):
                    if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        k4s.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        ind4s.append((1 << i) | (1 << j) | (1 << k) | (1 << l))

    # Convert to lists of ints for faster bitwise ops
    k4_masks = k4s
    ind4_masks = ind4s

    def count_violations(mask):
        viol = 0
        for k4 in k4_masks:
            if (mask & k4) == k4:
                viol += 1
        for i4 in ind4_masks:
            if (mask & i4) == 0:
                viol += 1
        return viol

    def try_add_vertex():
        # Simulated Annealing
        # Initial mask: Random
        best_mask = random.getrandbits(n)
        best_viol = count_violations(best_mask)
        
        if best_viol == 0:
            return best_mask

        # Parameters
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        # Try multiple restarts if stuck
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time_left() < 5:
                return None
            
            # Reset or continue? Let's restart with new random mask if stuck
            if restart > 0:
                best_mask = random.getrandbits(n)
                best_viol = count_violations(best_mask)
                if best_viol == 0:
                    return best_mask
                T = 1.0
            
            for step in range(2000):
                if time_left() < 5:
                    return None
                
                # Pick a random bit to flip
                bit_idx = random.randint(0, n - 1)
                bit = 1 << bit_idx
                new_mask = best_mask ^ bit
                new_viol = count_violations(new_mask)
                
                delta = new_viol - best_viol
                
                if delta < 0:
                    best_mask = new_mask
                    best_viol = new_viol
                    if best_viol == 0:
                        return best_mask
                else:
                    # Accept with probability exp(-delta/T)
                    # Since delta is integer >= 1, and T is small, this is rare
                    if random.random() < (2.718281828 ** (-delta / T)):
                        best_mask = new_mask
                        best_viol = new_viol
                
                T *= cooling
                if T < min_T:
                    break
        
        return None

    # Loop to add vertices
    added = 0
    # We aim to reach 42 (Exoo's bound) or higher
    # R(5,5) is between 43 and 46. So max n is 45.
    # Let's try to add up to 9 vertices (37+9=46)
    while added < 9:
        if time_left() < 20:
            break
        
        mask = try_add_vertex()
        if mask is None:
            # Try harder? Or break?
            # Let's try a few more times with lower temperature or different strategy
            # For now, break if we can't find one quickly
            break
        
        # Add vertex n
        new_adj = [0] * (n + 1)
        for i in range(n):
            new_adj[i] = adj[i]
            if mask & (1 << i):
                new_adj[i] |= 1 << n
                new_adj[n] |= 1 << i
                edges.append((i, n))
        
        adj = new_adj
        old_n = n
        n += 1
        added += 1
        
        # Update K4s and Ind4s
        # New K4s must involve the new vertex (n-1)
        new_k4s = []
        new_ind4s = []
        
        # Iterate over triples (i, j, k) in the old graph
        # We need to check if {i, j, k, n-1} is a K4 or Ind4
        for i in range(old_n):
            for j in range(i+1, old_n):
                for k in range(j+1, old_n):
                    # Check K4
                    if (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        if (adj[i] & (1 << (n-1))) and (adj[j] & (1 << (n-1))) and (adj[k] & (1 << (n-1))):
                            new_k4s.append((1 << i) | (1 << j) | (1 << k) | (1 << (n-1)))
                            
                    # Check Ind4
                    if not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        if not (adj[i] & (1 << (n-1))) and not (adj[j] & (1 << (n-1))) and not (adj[k] & (1 << (n-1))):
                            new_ind4s.append((1 << i) | (1 << j) | (1 << k) | (1 << (n-1)))
        
        k4_masks.extend(new_k4s)
        ind4_masks.extend(new_ind4s)

    return n, edges
