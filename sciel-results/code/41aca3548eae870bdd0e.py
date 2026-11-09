import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Precompute complement masks for B&B
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ conflict_mask[i]
        
    recip = [1.0 / numbers[i] for i in range(n)]
    
    def elapsed():
        return time.time() - start_time
        
    # Heuristic phase to get a good incumbent
    best_sum = 0.0
    best_mask = 0
    
    def greedy_order(order, selected_mask, current_sum):
        for i in order:
            if not (conflict_mask[i] & selected_mask):
                selected_mask |= (1 << i)
                current_sum += recip[i]
        return selected_mask, current_sum
        
    def fill_set(selected_mask, current_sum):
        improved = True
        while improved and elapsed() < DEADLINE - 5:
            improved = False
            for i in range(n):
                if not (selected_mask & (1 << i)):
                    if not (conflict_mask[i] & selected_mask):
                        selected_mask |= (1 << i)
                        current_sum += recip[i]
                        improved = True
        return selected_mask, current_sum

    def local_search(selected_mask, current_sum):
        improved = True
        while improved and elapsed() < DEADLINE - 5:
            improved = False
            for i in range(n):
                if not (selected_mask & (1 << i)):
                    continue
                new_mask = selected_mask & ~(1 << i)
                new_sum = current_sum - recip[i]
                
                best_j = -1
                best_gain = 0.0
                for j in range(n):
                    if new_mask & (1 << j):
                        continue
                    if conflict_mask[j] & new_mask:
                        continue
                    gain = recip[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                
                if best_j >= 0 and new_sum + best_gain > current_sum + 1e-12:
                    selected_mask = new_mask | (1 << best_j)
                    current_sum = new_sum + best_gain
                    improved = True
                    break
            
            if not improved:
                new_mask, new_sum = fill_set(selected_mask, current_sum)
                if new_sum > current_sum + 1e-12:
                    selected_mask = new_mask
                    current_sum = new_sum
                    improved = True
        return selected_mask, current_sum

    # Run a few heuristic restarts
    random.seed(42)
    # Ascending
    m, s = greedy_order(list(range(n)), 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Descending
    m, s = greedy_order(list(range(n-1, -1, -1)), 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m

    # Randomized
    for _ in range(50):
        if elapsed() > 10: break
        weights = [recip[i] ** 2 for i in range(n)]
        order = sorted(range(n), key=lambda i: random.random() / (weights[i] + 1e-10))
        m, s = greedy_order(order, 0, 0.0)
        m, s = local_search(m, s)
        if s > best_sum:
            best_sum = s
            best_mask = m

    # B&B Phase
    def color_bound(candidates_mask):
        if candidates_mask == 0:
            return 0.0
        verts = [i for i in range(n) if (candidates_mask >> i) & 1]
        verts.sort(key=lambda i: recip[i], reverse=True)
        
        colors = []
        for v in verts:
            v_bit = 1 << v
            placed = False
            for c_idx, c_mask in enumerate(colors):
                if (comp_mask[v] & c_mask) == 0:
                    colors[c_idx] |= v_bit
                    placed = True
                    break
            if not placed:
                colors.append(v_bit)
        
        bound = 0.0
        for c_mask in colors:
            max_w = 0.0
            m = c_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if recip[idx] > max_w:
                    max_w = recip[idx]
                m ^= lsb
            bound += max_w
        return bound

    def bron_kerbosch(P_mask, X_mask, W, current_mask):
        nonlocal best_sum, best_mask
        
        if elapsed() > DEADLINE - 2:
            return
            
        if W + color_bound(P_mask) <= best_sum + 1e-12:
            return
            
        if P_mask == 0:
            if W > best_sum:
                best_sum = W
                best_mask = current_mask
            return
        
        candidates = [i for i in range(n) if (P_mask >> i) & 1]
        candidates.sort(key=lambda i: recip[i], reverse=True)
        
        for v in candidates:
            if not (P_mask >> v) & 1:
                continue
                
            new_P = P_mask & comp_mask[v]
            new_X = X_mask & comp_mask[v]
            new_W = W + recip[v]
            new_mask = current_mask | (1 << v)
            
            bron_kerbosch(new_P, new_X, new_W, new_mask)
            
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            if elapsed() > DEADLINE - 2:
                return

    bron_kerbosch(full_mask, 0, 0.0, 0)
    
    result = [numbers[i] for i in range(n) if (best_mask >> i) & 1]
    return result
