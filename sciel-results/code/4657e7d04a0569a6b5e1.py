import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_adj(adj, degrees=None):
        if degrees is None:
            degrees = compute_degrees(adj)
        
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_adj = [0] * n
    best_count = 0
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    
    # Phase 1: Greedy construction with multiple random orderings
    for trial in range(200):
        if time.time() - start_time > deadline - 10:
            break
        
        trial_adj = [0] * n
        trial_deg = [0] * n
        trial_count = 0
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            new_adj = trial_adj[:]
            new_deg = trial_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                trial_adj = new_adj
                trial_deg = new_deg
                trial_count += 1
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = trial_adj[:]
        
        if best_count >= 45:
            break
    
    # Phase 2: Local Search
    current_adj = best_adj[:]
    current_deg = compute_degrees(current_adj)
    current_count = best_count
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() - start_time < deadline:
        no_improve += 1
        if no_improve > max_no_improve:
            break
        
        added = False
        # Try adding random edges
        for _ in range(200):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or u > v:
                continue
            if current_adj[u] & (1 << v):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                added = True
                no_improve = 0
                break
        
        if added:
            continue
        
        # Try swapping edges
        swapped = False
        for _ in range(200):
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1:
                continue
            if u1 > v1:
                u1, v1 = v1, u1
            if not (current_adj[u1] & (1 << v1)):
                continue
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1) or (current_adj[u2] & (1 << v2)):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                swapped = True
                no_improve = 0
                break
                
        if not swapped and not added:
             # Occasionally restart from best
             if random.random() < 0.1:
                 current_adj = best_adj[:]
                 current_deg = compute_degrees(current_adj)
                 current_count = best_count
                 no_improve = 0

    final_edges = adj_to_edges(best_adj)
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    return final_edges
