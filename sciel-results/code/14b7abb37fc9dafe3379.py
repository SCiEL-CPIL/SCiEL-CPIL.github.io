import time
import random
import math

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute valid_c_mask
# valid_c_mask[a_idx][b_idx] = bitmask of c in [2..256] such that a | b*c
valid_c_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        mask = 0
        g = math.gcd(a, b)
        req = a // g
        # c must be a multiple of req
        # We can iterate multiples of req
        for c_val in range(2, NMAX + 1):
            if c_val % req == 0:
                mask |= (1 << val_to_idx[c_val])
        valid_c_mask[i][j] = mask

def get_degrees(S_mask):
    degrees = [0] * N
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    for i in a_list:
        a_idx = i
        deg = 0
        for j in a_list:
            if j == i:
                continue
            b_idx = j
            valid_mask = valid_c_mask[a_idx][b_idx]
            count_mask = S_mask & valid_mask & ~(1 << i) & ~(1 << j)
            deg += bin(count_mask).count('1')
        degrees[a_idx] = deg
    return degrees

def greedy_removal(deadline, tie_breaker='largest'):
    S_mask = (1 << N) - 1
    degrees = get_degrees(S_mask)
    
    while time.time() < deadline:
        max_deg = -1
        best_idx = -1
        for i in range(N):
            if not (S_mask >> i) & 1:
                continue
            deg = degrees[i]
            if deg > max_deg:
                max_deg = deg
                best_idx = i
            elif deg == max_deg:
                if best_idx == -1:
                    best_idx = i
                else:
                    val_i = idx_to_val[i]
                    val_best = idx_to_val[best_idx]
                    if tie_breaker == 'largest':
                        if val_i > val_best:
                            best_idx = i
                    elif tie_breaker == 'smallest':
                        if val_i < val_best:
                            best_idx = i
                    else:
                        if random.random() < 0.5:
                            best_idx = i
        
        if best_idx == -1:
            break
        if max_deg == 0:
            break
            
        S_mask &= ~(1 << best_idx)
        degrees = get_degrees(S_mask)
        
    res = []
    for i in range(N):
        if (S_mask >> i) & 1:
            res.append(idx_to_val[i])
    return res

def local_search(S, deadline):
    S_set = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << val_to_idx[v])
    
    best_S = S[:]
    best_size = len(S)
    
    all_elements = set(RANGE)
    removed = list(all_elements - S_set)
    
    while time.time() < deadline:
        improved = False
        random.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline:
                break
            
            x_idx = val_to_idx[x]
            can_add = True
            
            S_list = list(S_set)
            n_s = len(S_list)
            
            # Check if x can be added to S
            # For all distinct b, c in S:
            # 1. x !| b*c
            # 2. b !| x*c
            # 3. c !| x*b
            
            for i in range(n_s):
                b = S_list[i]
                # Check 1: x | b*c
                g_xb = math.gcd(x, b)
                req_x = x // g_xb
                # Check 2: b | x*c
                g_bx = math.gcd(b, x)
                req_b = b // g_bx
                
                for j in range(n_s):
                    if i == j:
                        continue
                    c = S_list[j]
                    
                    # Check 1: x | b*c  =>  (b*c) % x == 0
                    # Equivalent to: c % (x / gcd(x,b)) == 0
                    if c % req_x == 0:
                        can_add = False
                        break
                    
                    # Check 2: b | x*c  =>  (x*c) % b == 0
                    # Equivalent to: c % (b / gcd(b,x)) == 0
                    if c % req_b == 0:
                        can_add = False
                        break
                        
                if not can_add:
                    break
            
            if can_add:
                S_set.add(x)
                S_mask |= (1 << x_idx)
                removed.remove(x)
                best_S = list(S_set)
                best_size = len(S_set)
                improved = True
                break # Restart with new S
            
        if not improved:
            break
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Run greedy removal with different tie breakers
    for tb in ['largest', 'smallest', 'random']:
        if time.time() > deadline - 20:
            break
        c = greedy_removal(deadline - 20, tb)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Local search on best
    if time.time() < deadline - 5:
        c = local_search(best, deadline - 5)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Final check and return
    res = list(best)
    # Verify
    n = len(res)
    valid = True
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(n):
                if i == k or j == k: continue
                if (res[j] * res[k]) % res[i] == 0:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to a known small valid set or empty
        return [256]
        
    return res
