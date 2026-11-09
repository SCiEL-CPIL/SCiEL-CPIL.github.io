import time
import random
import sys

def construct():
    deadline = time.time() + 85  # 90s limit, 5s safety margin
    
    N = 256
    n = N
    
    # Build adjacency as bitmask: adj[a] has bit b set if a and b conflict
    adj = [0] * n
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a-1] |= (1 << (b-1))
                adj[b-1] |= (1 << (a-1))
    
    def can_add(v, set_mask):
        return (adj[v] & set_mask) == 0
    
    def get_set_bits(mask):
        bits = []
        temp = mask
        while temp:
            v_bit = temp & (-temp)
            bits.append(v_bit.bit_length() - 1)
            temp &= temp - 1
        return bits
    
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy initial solution with various strategies
    def greedy_solution(strategy=0):
        selected = 0
        remaining = (1 << n) - 1
        while remaining:
            if strategy == 0:
                # Minimum degree greedy
                best_v = -1
                best_deg = n + 1
                temp = remaining
                while temp:
                    v_bit = temp & (-temp)
                    v_idx = v_bit.bit_length() - 1
                    deg = popcount(adj[v_idx] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v_idx
                    temp &= temp - 1
            elif strategy == 1:
                # Maximum value greedy
                best_v = -1
                temp = remaining
                while temp:
                    v_bit = temp & (-temp)
                    v_idx = v_bit.bit_length() - 1
                    if best_v == -1 or v_idx > best_v:
                        best_v = v_idx
                    temp &= temp - 1
            elif strategy == 2:
                # Minimum degree with random tie-breaking
                best_v = -1
                best_deg = n + 1
                candidates = []
                temp = remaining
                while temp:
                    v_bit = temp & (-temp)
                    v_idx = v_bit.bit_length() - 1
                    deg = popcount(adj[v_idx] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        candidates = [v_idx]
                    elif deg == best_deg:
                        candidates.append(v_idx)
                    temp &= temp - 1
                best_v = random.choice(candidates) if candidates else -1
            else:
                # Random
                bits = get_set_bits(remaining)
                best_v = random.choice(bits)
            
            if best_v < 0:
                break
            selected |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return selected
    
    # Try multiple greedy strategies
    best_mask = 0
    best_size = 0
    for strat in range(15):
        mask = greedy_solution(strat % 4)
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Local search with multiple move types
    def local_search(mask, max_time=0.5):
        current = mask
        current_size = popcount(current)
        start = time.time()
        improved = True
        
        while improved and time.time() - start < max_time:
            improved = False
            
            # 1-move: add a vertex
            for v in range(n):
                if not (current >> v) & 1:
                    if can_add(v, current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2-move: remove one, add two (net +1)
            set_bits = get_set_bits(current)
            found = False
            for v_remove in set_bits:
                if found:
                    break
                new_set = current & ~(1 << v_remove)
                candidates = []
                for v in range(n):
                    if (new_set >> v) & 1:
                        continue
                    if can_add(v, new_set):
                        candidates.append(v)
                
                for i in range(len(candidates)):
                    if found:
                        break
                    for j in range(i+1, len(candidates)):
                        if found:
                            break
                        v1, v2 = candidates[i], candidates[j]
                        if can_add(v2, new_set | (1 << v1)):
                            current = new_set | (1 << v1) | (1 << v2)
                            current_size += 1
                            improved = True
                            found = True
                            break
            
            if improved:
                continue
            
            # 3-move: remove two, add three (net +1)
            set_bits = get_set_bits(current)
            found = False
            for i in range(len(set_bits)):
                if found:
                    break
                for j in range(i+1, len(set_bits)):
                    if found:
                        break
                    v1_remove, v2_remove = set_bits[i], set_bits[j]
                    new_set = current & ~(1 << v1_remove) & ~(1 << v2_remove)
                    candidates = []
                    for v in range(n):
                        if (new_set >> v) & 1:
                            continue
                        if can_add(v, new_set):
                            candidates.append(v)
                    
                    for a_idx in range(len(candidates)):
                        if found:
                            break
                        for b_idx in range(a_idx+1, len(candidates)):
                            if found:
                                break
                            for c_idx in range(b_idx+1, len(candidates)):
                                if found:
                                    break
                                va, vb, vc = candidates[a_idx], candidates[b_idx], candidates[c_idx]
                                if can_add(vb, new_set | (1 << va)) and can_add(vc, new_set | (1 << va) | (1 << vb)):
                                    current = new_set | (1 << va) | (1 << vb) | (1 << vc)
                                    current_size += 1
                                    improved = True
                                    found = True
                                    break
            
            if improved:
                continue
            
            # 4-move: remove three, add four (net +1)
            set_bits = get_set_bits(current)
            found = False
            for i in range(len(set_bits)):
                if found:
                    break
                for j in range(i+1, len(set_bits)):
                    if found:
                        break
                    for k in range(j+1, len(set_bits)):
                        if found:
                            break
                        v1_remove, v2_remove, v3_remove = set_bits[i], set_bits[j], set_bits[k]
                        new_set = current & ~(1 << v1_remove) & ~(1 << v2_remove) & ~(1 << v3_remove)
                        candidates = []
                        for v in range(n):
                            if (new_set >> v) & 1:
                                continue
                            if can_add(v, new_set):
                                candidates.append(v)
                        
                        for a_idx in range(len(candidates)):
                            if found:
                                break
                            for b_idx in range(a_idx+1, len(candidates)):
                                if found:
                                    break
                                for c_idx in range(b_idx+1, len(candidates)):
                                    if found:
                                        break
                                    for d_idx in range(c_idx+1, len(candidates)):
                                        if found:
                                            break
                                        va, vb, vc, vd = candidates[a_idx], candidates[b_idx], candidates[c_idx], candidates[d_idx]
                                        if (can_add(vb, new_set | (1 << va)) and 
                                            can_add(vc, new_set | (1 << va) | (1 << vb)) and
                                            can_add(vd, new_set | (1 << va) | (1 << vb) | (1 << vc))):
                                            current = new_set | (1 << va) | (1 << vb) | (1 << vc) | (1 << vd)
                                            current_size += 1
                                            improved = True
                                            found = True
                                            break
            
            if improved:
                continue
        
        return current, current_size
    
    # Iterated local search with adaptive perturbation
    for iteration in range(500):
        if time.time() > deadline - 2:
            break
        
        # Adaptive perturbation: more aggressive when stuck
        if iteration < 50:
            k = random.randint(1, 5)
        elif iteration < 150:
            k = random.randint(2, 8)
        else:
            k = random.randint(3, 12)
        
        perturbed = best_mask
        for _ in range(k):
            bits = get_set_bits(perturbed)
            if not bits:
                break
            v_remove = random.choice(bits)
            perturbed &= ~(1 << v_remove)
        
        # Local search with time budget
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        max_ls_time = min(0.3, remaining_time * 0.5)
        improved_mask, improved_size = local_search(perturbed, max_time=max_ls_time)
        
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask
    
    # Final polish: try to add any remaining vertices
    for v in range(n):
        if not (best_mask >> v) & 1:
            if can_add(v, best_mask):
                best_mask |= (1 << v)
                best_size += 1
    
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    return result
