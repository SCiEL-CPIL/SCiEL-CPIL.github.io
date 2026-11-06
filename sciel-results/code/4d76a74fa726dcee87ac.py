# EVOLVE-BLOCK-START
import time

N = 256
DEADLINE = 85

def construct():
    start_time = time.time()
    
    # Build conflict graph: conflict[i] is a bitmask where bit j is set if i and j conflict
    # Two numbers a < b conflict if they can form a 3-AP with some third number in [1,256]
    # conflict(a,b) iff: (2b - a <= N) or ((b-a) % 2 == 0) or (2a - b >= 1)
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (2 * b - a <= N) or (d % 2 == 0) or (2 * a - b >= 1):
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
    
    def is_compatible(v, S_mask):
        return (S_mask & conflict[v]) == 0
    
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if (mask & (1 << i))]
    
    # Greedy initialization
    S_mask = 0
    for x in range(1, N + 1):
        if is_compatible(x, S_mask):
            S_mask |= (1 << x)
    
    # Pass 1: Add any compatible missing elements
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for z in range(1, N + 1):
            if not (S_mask & (1 << z)):
                if is_compatible(z, S_mask):
                    S_mask |= (1 << z)
                    improved = True
    
    # Pass 2: Single swaps
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        S_list = mask_to_list(S_mask)
        for y in sorted(S_list, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            S_temp = S_mask & ~(1 << y)
            for z in range(1, y):
                if S_mask & (1 << z):
                    continue
                if is_compatible(z, S_temp):
                    S_mask = S_temp | (1 << z)
                    improved = True
                    break
    
    # Pass 2.5: 2-swaps
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        S_list = mask_to_list(S_mask)
        for y1 in sorted(S_list, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            for y2 in sorted(S_list, reverse=True):
                if y2 >= y1:
                    continue
                if time.time() - start_time > DEADLINE:
                    break
                S_temp = S_mask & ~(1 << y1) & ~(1 << y2)
                base_val = 1.0 / y1 + 1.0 / y2
                best_pair = None
                best_val = base_val
                max_z = max(y1, y2)
                for z1 in range(1, max_z):
                    if S_mask & (1 << z1):
                        continue
                    if not is_compatible(z1, S_temp):
                        continue
                    S_temp2 = S_temp | (1 << z1)
                    for z2 in range(z1 + 1, max_z):
                        if S_mask & (1 << z2):
                            continue
                        if not is_compatible(z2, S_temp2):
                            continue
                        val = 1.0 / z1 + 1.0 / z2
                        if val > best_val + 1e-12:
                            best_val = val
                            best_pair = (z1, z2)
                    if best_pair is not None and best_val > base_val + 0.05:
                        break
                if best_pair is not None:
                    S_mask = S_temp | (1 << best_pair[0]) | (1 << best_pair[1])
                    improved = True
                    break
            if improved:
                break
    
    # Pass 3: Single swaps again
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        S_list = mask_to_list(S_mask)
        for y in sorted(S_list, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            S_temp = S_mask & ~(1 << y)
            for z in range(1, y):
                if S_mask & (1 << z):
                    continue
                if is_compatible(z, S_temp):
                    S_mask = S_temp | (1 << z)
                    improved = True
                    break
    
    # Pass 4: Add pass
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for z in range(1, N + 1):
            if not (S_mask & (1 << z)):
                if is_compatible(z, S_mask):
                    S_mask |= (1 << z)
                    improved = True
    
    # Pass 5: Polish with single swaps
    if time.time() - start_time < DEADLINE - 5:
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            S_list = mask_to_list(S_mask)
            for y in sorted(S_list, reverse=True):
                if time.time() - start_time > DEADLINE:
                    break
                S_temp = S_mask & ~(1 << y)
                for z in range(1, y):
                    if S_mask & (1 << z):
                        continue
                    if is_compatible(z, S_temp):
                        S_mask = S_temp | (1 << z)
                        improved = True
                        break
        
        # Final add
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for z in range(1, N + 1):
                if not (S_mask & (1 << z)):
                    if is_compatible(z, S_mask):
                        S_mask |= (1 << z)
                        improved = True
    
    # Pass 6: 3-swap moves for further improvement
    if time.time() - start_time < DEADLINE - 10:
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            S_list = mask_to_list(S_mask)
            if len(S_list) < 3:
                break
            S_sorted = sorted(S_list, reverse=True)
            y1, y2, y3 = S_sorted[0], S_sorted[1], S_sorted[2]
            S_temp = S_mask & ~(1 << y1) & ~(1 << y2) & ~(1 << y3)
            base_val = 1.0 / y1 + 1.0 / y2 + 1.0 / y3
            max_z = max(y1, y2, y3)
            best_triple = None
            best_val = base_val
            
            for z1 in range(1, max_z):
                if S_mask & (1 << z1):
                    continue
                if not is_compatible(z1, S_temp):
                    continue
                S_temp2 = S_temp | (1 << z1)
                for z2 in range(z1 + 1, max_z):
                    if S_mask & (1 << z2):
                        continue
                    if not is_compatible(z2, S_temp2):
                        continue
                    S_temp3 = S_temp2 | (1 << z2)
                    for z3 in range(z2 + 1, max_z):
                        if S_mask & (1 << z3):
                            continue
                        if not is_compatible(z3, S_temp3):
                            continue
                        val = 1.0 / z1 + 1.0 / z2 + 1.0 / z3
                        if val > best_val + 1e-12:
                            best_val = val
                            best_triple = (z1, z2, z3)
                        break
                    if best_triple is not None:
                        break
                if best_triple is not None:
                    break
            
            if best_triple is not None:
                S_mask = S_temp | (1 << best_triple[0]) | (1 << best_triple[1]) | (1 << best_triple[2])
                improved = True
    
    # Final add pass
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for z in range(1, N + 1):
            if not (S_mask & (1 << z)):
                if is_compatible(z, S_mask):
                    S_mask |= (1 << z)
                    improved = True
    
    return mask_to_list(S_mask)
# EVOLVE-BLOCK-END
