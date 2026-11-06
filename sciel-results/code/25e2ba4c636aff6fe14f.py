import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    # Map 1..256 to bits 0..255.
    def val_to_bit(x):
        return 1 << (x - 1)
    
    # Precompute forbidden masks for 4-APs
    # forbidden_masks[v] = list of masks. If S contains any mask, v is forbidden.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d. Need {v+d, v+2d, v+3d} in S.
        # v is 2nd: v-d, v, v+d, v+2d. Need {v-d, v+d, v+2d} in S.
        # v is 3rd: v-2d, v-d, v, v+d. Need {v-2d, v-d, v+d} in S.
        # v is 4th: v-3d, v-2d, v-d, v. Need {v-3d, v-2d, v-d} in S.
        
        for d in range(1, n // 3 + 1):
            # Case 1
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
            # Case 2
            if v - d >= 1:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
            # Case 3
            if v - 2 * d >= 1:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
            # Case 4
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP contributions for delta
    # For each element u, precompute list of "pairs" (j, k) such that {u, j, k} is a 3-AP.
    # We need to count how many of these pairs are fully in S.
    # Let's store for each u, a list of masks M_u_i.
    # If (S & M_u_i) == M_u_i, then that 3-AP exists.
    
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks
        
    # Helper: Count 3-APs in S (represented as int bitmask)
    def count_3aps_from_bitmask(S_mask):
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    # Helper: Check if v can be added to S (S_mask)
    def can_add(S_mask, v):
        masks = forbidden_masks[v]
        for m in masks:
            if (S_mask & m) == m:
                return False
        return True

    # Helper: Calculate delta 3-APs for swap u -> v
    def calc_delta(S_mask, u, v):
        # S_mask contains u.
        S_mask_without_u = S_mask & ~val_to_bit(u)
        
        # Lost: count masks in ap3_masks[u] that are subset of S_mask.
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        # Gained: count masks in ap3_masks[v] that are subset of S_mask_without_u.
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # 1. Initial Greedy Construction
    # Use bitmask for S
    S_mask = 0
    S_set = set()
    
    for x in range(1, n + 1):
        if can_add(S_mask, x):
            S_mask |= val_to_bit(x)
            S_set.add(x)
            
    current_count = count_3aps_from_bitmask(S_mask)
    best_mask = S_mask
    best_count = current_count
    best_set = set(S_set)
    
    # 2. Simulated Annealing
    T_start = 2.0
    T_end = 0.01
    num_steps = 100000
    
    current_mask = S_mask
    
    for step in range(num_steps):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # Pick random u in S
        u = 0
        while True:
            u = random.randint(1, n)
            if (current_mask >> (u-1)) & 1:
                break
                
        # Pick random v not in S
        v = 0
        while True:
            v = random.randint(1, n)
            if not ((current_mask >> (v-1)) & 1):
                break
        
        # Check validity of v in S \ {u}
        S_mask_without_u = current_mask & ~val_to_bit(u)
        
        valid = True
        for m in forbidden_masks[v]:
            if (S_mask_without_u & m) == m:
                valid = False
                break
        
        if not valid:
            continue
            
        delta = calc_delta(current_mask, u, v)
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            current_mask = S_mask_without_u | val_to_bit(v)
            current_count += delta
            
            if current_count > best_count:
                best_count = current_count
                best_mask = current_mask
                # Update best_set for final return
                best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
                
    return sorted(best_set)

# Test run to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
