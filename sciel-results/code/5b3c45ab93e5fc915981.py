import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline:
            break
            
        # If n is small, we might have a trivial solution or quick search
        # For n <= 9, the seed solution works or is trivial.
        # But we want max n.
        
        # Initialize state
        # We store colors in a list, but use bitmasks for search
        # red_sets: dict (u,v) -> bitmask of w such that (u,v,w) is RED
        # blue_sets: dict (u,v) -> bitmask of w such that (u,v,w) is BLUE
        
        # Pre-compute list of triples in lexicographic order
        triples = list(combinations(range(n), 3))
        
        # To speed up, we can use a recursive solver
        # State: idx, red_sets, blue_sets
        # We need to pass dictionaries, which is expensive.
        # We can use lists of size n*n for bitmasks.
        # Index u*n + v. Only use u < v.
        
        size = n * n
        red_masks = [0] * size
        blue_masks = [0] * size
        colors = [0] * len(triples)
        
        def get_mask(u, v):
            if u > v: u, v = v, u
            return u * n + v
        
        def solve(idx):
            if time.time() > deadline:
                return None
            if idx == len(triples):
                return colors[:]
            
            i, j, k = triples[idx]
            mask_ij = get_mask(i, j)
            mask_ik = get_mask(i, k)
            mask_jk = get_mask(j, k)
            
            # Try Red (0)
            # Check conflict: red_masks[mask_ij] & red_masks[mask_ik] & red_masks[mask_jk] == 0
            if (red_masks[mask_ij] & red_masks[mask_ik] & red_masks[mask_jk]) == 0:
                colors[idx] = 0
                # Update
                red_masks[mask_ij] |= (1 << k)
                red_masks[mask_ik] |= (1 << j)
                red_masks[mask_jk] |= (1 << i)
                
                res = solve(idx + 1)
                if res is not None:
                    return res
                
                # Backtrack
                red_masks[mask_ij] &= ~(1 << k)
                red_masks[mask_ik] &= ~(1 << j)
                red_masks[mask_jk] &= ~(1 << i)
            
            # Try Blue (1)
            # Check conflict
            C = blue_masks[mask_ij] & blue_masks[mask_ik] & blue_masks[mask_jk]
            valid_blue = True
            # Iterate bits in C
            c_temp = C
            while c_temp:
                # Get lowest set bit
                lsb = c_temp & -c_temp
                v = lsb.bit_length() - 1
                c_temp ^= lsb
                
                # Check if there exists w in C such that (i,v,w), (j,v,w), (k,v,w) are Blue
                # W_v = blue_masks[get_mask(i,v)] & blue_masks[get_mask(j,v)] & blue_masks[get_mask(k,v)]
                # We need W_v & C != 0
                mask_iv = get_mask(i, v)
                mask_jv = get_mask(j, v)
                mask_kv = get_mask(k, v)
                
                W_v = blue_masks[mask_iv] & blue_masks[mask_jv] & blue_masks[mask_kv]
                if W_v & C:
                    valid_blue = False
                    break
            
            if valid_blue:
                colors[idx] = 1
                # Update
                blue_masks[mask_ij] |= (1 << k)
                blue_masks[mask_ik] |= (1 << j)
                blue_masks[mask_jk] |= (1 << i)
                
                res = solve(idx + 1)
                if res is not None:
                    return res
                
                # Backtrack
                blue_masks[mask_ij] &= ~(1 << k)
                blue_masks[mask_ik] &= ~(1 << j)
                blue_masks[mask_jk] &= ~(1 << i)
            
            return None
        
        result = solve(0)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to seed if all fail (should not happen for n=9)
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
