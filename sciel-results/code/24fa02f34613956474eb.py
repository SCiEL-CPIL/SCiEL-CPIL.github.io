import time
import random
import numpy as np
from itertools import combinations

def count_violations(adj, n):
    """Count K5 cliques and independent sets of size 5 using bitmasks."""
    if n < 5:
        return 0
    
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i, j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    all_mask = (1 << n) - 1
    
    def count_k_in_induced(mask, k):
        if k == 1:
            return bin(mask).count('1')
        if k == 2:
            cnt = 0
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                cnt += bin(neigh[v] & mask).count('1')
            return cnt // 2
        if k == 3:
            cnt = 0
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                sub = neigh[v] & mask
                sm = sub
                while sm:
                    lsb2 = sm & (-sm)
                    w = lsb2.bit_length() - 1
                    sm ^= lsb2
                    cnt += bin(neigh[w] & sub).count('1')
                cnt //= 2
            return cnt // 3
        if k == 4:
            cnt = 0
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                sub = neigh[v] & mask
                tri = 0
                sm = sub
                while sm:
                    lsb2 = sm & (-sm)
                    w = lsb2.bit_length() - 1
                    sm ^= lsb2
                    tri += bin(neigh[w] & sub).count('1')
                tri //= 2
                cnt += tri
            return cnt // 4
        return 0
    
    total = 0
    for i in range(n):
        total += count_k_in_induced(neigh[i], 4)
        non_neigh = all_mask & ~(neigh[i] | (1 << i))
        total += count_k_in_induced(non_neigh, 4)
    
    return total // 5

def construct():
    start_time = time.time()
    deadline = start_time + 270
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    for target_n in range(q + 1, 43):
        if time.time() > deadline - 20:
            break
        
        found_valid = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            adj = np.zeros((target_n, target_n), dtype=bool)
            
            for i, j in seed_edges:
                adj[i, j] = True
                adj[j, i] = True
            
            for i in range(q, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            for i in range(q):
                for j in range(q, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            current_violations = count_violations(adj, target_n)
            
            if current_violations == 0:
                found_valid = True
                break
            
            temperature = 1.0
            cooling_rate = 0.9995
            steps = 5000
            best_violations = current_violations
            best_adj = adj.copy()
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                
                if current_violations == 0:
                    found_valid = True
                    break
                
                i, j = np.random.randint(0, target_n, 2)
                if i == j:
                    continue
                
                old_val = adj[i, j]
                adj[i, j] = not old_val
                adj[j, i] = not old_val
                
                new_violations = count_violations(adj, target_n)
                delta = new_violations - current_violations
                
                if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_adj = adj.copy()
                else:
                    adj[i, j] = old_val
                    adj[j, i] = old_val
                
                temperature *= cooling_rate
            
            if current_violations == 0:
                found_valid = True
                break
            
            # Use best found so far for next restart
            if best_violations < current_violations:
                adj = best_adj
                current_violations = best_violations
        
        if found_valid:
            best_n = target_n
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
        else:
            break
    
    return best_n, best_edges
