import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290

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
    k4_masks = []
    ind4_masks = []
    
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)): continue
            for k in range(j+1, n):
                if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                for l in range(k+1, n):
                    if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        k4_masks.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        ind4_masks.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
    
    # Build membership lists
    k4_containing = [[] for _ in range(n)]
    ind4_containing = [[] for _ in range(n)]
    
    for idx, mask in enumerate(k4_masks):
        for b in range(n):
            if mask & (1 << b):
                k4_containing[b].append(idx)
                
    for idx, mask in enumerate(ind4_masks):
        for b in range(n):
            if mask & (1 << b):
                ind4_containing[b].append(idx)

    def count_violations(mask):
        k4_viol = 0
        for k4m in k4_masks:
            if (mask & k4m) == k4m:
                k4_viol += 1
        ind4_viol = 0
        for i4m in ind4_masks:
            if (mask & i4m) == 0:
                ind4_viol += 1
        return k4_viol + ind4_viol

    def try_add_vertex():
        if time_left() < 10:
            return None
            
        mask = random.getrandbits(n)
        total_viol = count_violations(mask)
        
        if total_viol == 0:
            return mask

        steps = 0
        max_steps = 50000
        no_improve_count = 0
        best_total = total_viol
        best_mask = mask
        
        while steps < max_steps:
            if time_left() < 10:
                return best_mask if count_violations(best_mask) == 0 else None
                
            best_delta = 0
            best_bit = -1
            best_is_add = False
            
            # Try adding bits
            for b in range(n):
                if mask & (1 << b): continue
                
                d_k4 = 0
                for idx in k4_containing[b]:
                    k4m = k4_masks[idx]
                    other_bits = k4m & ~(1 << b)
                    if (mask & other_bits) == other_bits:
                        d_k4 += 1
                        
                d_ind4 = 0
                for idx in ind4_containing[b]:
                    i4m = ind4_masks[idx]
                    other_bits = i4m & ~(1 << b)
                    if (mask & other_bits) == 0:
                        d_ind4 -= 1
                        
                delta = d_k4 + d_ind4
                if delta < best_delta:
                    best_delta = delta
                    best_bit = b
                    best_is_add = True
                    
            # Try removing bits
            for b in range(n):
                if not (mask & (1 << b)): continue
                
                d_k4 = 0
                for idx in k4_containing[b]:
                    k4m = k4_masks[idx]
                    other_bits = k4m & ~(1 << b)
                    if (mask & other_bits) == other_bits:
                        d_k4 -= 1
                
                d_ind4 = 0
                for idx in ind4_containing[b]:
                    i4m = ind4_masks[idx]
                    other_bits = i4m & ~(1 << b)
                    if (mask & other_bits) == 0:
                        d_ind4 += 1
                        
                delta = d_k4 + d_ind4
                if delta < best_delta:
                    best_delta = delta
                    best_bit = b
                    best_is_add = False
            
            if best_bit != -1 and best_delta < 0:
                if best_is_add:
                    mask |= (1 << best_bit)
                else:
                    mask &= ~(1 << best_bit)
                
                total_viol += best_delta
                if total_viol < best_total:
                    best_total = total_viol
                    best_mask = mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if total_viol == 0:
                    return mask
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    # Random restart
                    mask = random.getrandbits(n)
                    total_viol = count_violations(mask)
                    if total_viol < best_total:
                        best_total = total_viol
                        best_mask = mask
                    no_improve_count = 0
                    if total_viol == 0:
                        return mask

            steps += 1
             
        return best_mask if count_violations(best_mask) == 0 else None

    added = 0
    while added < 10:
        if time_left() < 30:
            break
            
        mask = try_add_vertex()
        if mask is None:
            break
        
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
        
        # Extend membership lists
        k4_containing.append([])
        ind4_containing.append([])
        
        # Update K4s and Ind4s
        for i in range(old_n):
            for j in range(i+1, old_n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j+1, old_n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                    
                    if (adj[i] & (1 << (n-1))) and (adj[j] & (1 << (n-1))) and (adj[k] & (1 << (n-1))):
                        new_mask = (1 << i) | (1 << j) | (1 << k) | (1 << (n-1))
                        k4_masks.append(new_mask)
                        k4_idx = len(k4_masks) - 1
                        k4_containing[i].append(k4_idx)
                        k4_containing[j].append(k4_idx)
                        k4_containing[k].append(k4_idx)
                        k4_containing[n-1].append(k4_idx)
                        
                    if not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        if not (adj[i] & (1 << (n-1))) and not (adj[j] & (1 << (n-1))) and not (adj[k] & (1 << (n-1))):
                            new_mask = (1 << i) | (1 << j) | (1 << k) | (1 << (n-1))
                            ind4_masks.append(new_mask)
                            ind4_idx = len(ind4_masks) - 1
                            ind4_containing[i].append(ind4_idx)
                            ind4_containing[j].append(ind4_idx)
                            ind4_containing[k].append(ind4_idx)
                            ind4_containing[n-1].append(ind4_idx)
                            
    return n, edges
