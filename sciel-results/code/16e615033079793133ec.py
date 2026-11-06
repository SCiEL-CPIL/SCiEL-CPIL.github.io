import time
import itertools
import random

# EVOLVE-BLOCK-START
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 320  # 320s safety margin before 330s kill
    
    def time_left():
        return deadline - time.time()
    
    # Build Paley graph on 37 vertices as fallback/seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    n_paley = q
    adj_mask_paley = [0] * n_paley
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            if (j - i) % q in residues:
                adj_mask_paley[i] |= (1 << j)
                adj_mask_paley[j] |= (1 << i)
    
    def build_edges(adj_mask, n):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            while m:
                lb = m & (-m)
                j = lb.bit_length() - 1
                if i < j:
                    edges.append((i, j))
                m ^= lb
        return edges

    def try_find_circulant_43():
        n = 43
        half_n = n // 2
        
        def count_k4_and_ind4(S_mask):
            adj = [0] * n
            for i in range(n):
                m = S_mask
                while m:
                    lb = m & (-m)
                    offset = lb.bit_length() - 1
                    j = (i + offset) % n
                    k = (i - offset) % n
                    adj[i] |= (1 << j)
                    adj[i] |= (1 << k)
                    m ^= lb
            
            k4_count = 0
            ind4_count = 0
            
            for comb in itertools.combinations(range(n), 4):
                v0, v1, v2, v3 = comb
                if (adj[v0] & (1 << v1)) and (adj[v0] & (1 << v2)) and (adj[v0] & (1 << v3)) and \
                   (adj[v1] & (1 << v2)) and (adj[v1] & (1 << v3)) and \
                   (adj[v2] & (1 << v3)):
                    k4_count += 1
                else:
                    if not (adj[v0] & (1 << v1)) and not (adj[v0] & (1 << v2)) and not (adj[v0] & (1 << v3)) and \
                       not (adj[v1] & (1 << v2)) and not (adj[v1] & (1 << v3)) and \
                       not (adj[v2] & (1 << v3)):
                        ind4_count += 1
            
            return k4_count + ind4_count

        def has_k5(S_mask):
            adj = [0] * n
            for i in range(n):
                m = S_mask
                while m:
                    lb = m & (-m)
                    offset = lb.bit_length() - 1
                    j = (i + offset) % n
                    k = (i - offset) % n
                    adj[i] |= (1 << j)
                    adj[i] |= (1 << k)
                    m ^= lb
            
            for comb in itertools.combinations(range(n), 5):
                v0, v1, v2, v3, v4 = comb
                if (adj[v0] & (1 << v1)) and (adj[v0] & (1 << v2)) and (adj[v0] & (1 << v3)) and (adj[v0] & (1 << v4)) and \
                   (adj[v1] & (1 << v2)) and (adj[v1] & (1 << v3)) and (adj[v1] & (1 << v4)) and \
                   (adj[v2] & (1 << v3)) and (adj[v2] & (1 << v4)) and \
                   (adj[v3] & (1 << v4)):
                    return True
            return False

        def has_ind5(S_mask):
            adj = [0] * n
            for i in range(n):
                m = S_mask
                while m:
                    lb = m & (-m)
                    offset = lb.bit_length() - 1
                    j = (i + offset) % n
                    k = (i - offset) % n
                    adj[i] |= (1 << j)
                    adj[i] |= (1 << k)
                    m ^= lb
            
            for comb in itertools.combinations(range(n), 5):
                v0, v1, v2, v3, v4 = comb
                if not (adj[v0] & (1 << v1)) and not (adj[v0] & (1 << v2)) and not (adj[v0] & (1 << v3)) and not (adj[v0] & (1 << v4)) and \
                   not (adj[v1] & (1 << v2)) and not (adj[v1] & (1 << v3)) and not (adj[v1] & (1 << v4)) and \
                   not (adj[v2] & (1 << v3)) and not (adj[v2] & (1 << v4)) and \
                   not (adj[v3] & (1 << v4)):
                    return True
            return False

        best_S = 0
        best_cost = float('inf')
        
        random.seed(42)
        # Limit restarts and steps to ensure time safety
        max_restarts = 5
        max_steps = 200
        
        for _ in range(max_restarts):
            if time_left() < 50: # Reserve time for final check
                break
                
            S = 0
            offsets = random.sample(range(1, half_n + 1), 10)
            for o in offsets:
                S |= (1 << o)
            
            cost = count_k4_and_ind4(S)
            if cost < best_cost:
                best_cost = cost
                best_S = S
                if cost == 0:
                    break
            
            for step in range(max_steps):
                if time_left() < 50:
                    break
                    
                flip_bit = random.randint(1, half_n)
                new_S = best_S ^ (1 << flip_bit)
                new_cost = count_k4_and_ind4(new_S)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_S = new_S
                    if best_cost == 0:
                        break
                elif new_cost == best_cost and random.random() < 0.5:
                    best_S = new_S
        
        if time_left() < 20:
            return None

        if has_k5(best_S) or has_ind5(best_S):
            return None
        
        adj = [0] * n
        for i in range(n):
            m = best_S
            while m:
                lb = m & (-m)
                offset = lb.bit_length() - 1
                j = (i + offset) % n
                k = (i - offset) % n
                adj[i] |= (1 << j)
                adj[i] |= (1 << k)
                m ^= lb
        
        edges = build_edges(adj, n)
        return n, edges

    result = try_find_circulant_43()
    if result:
        return result
    
    return n_paley, build_edges(adj_mask_paley, n_paley)

# EVOLVE-BLOCK-END
