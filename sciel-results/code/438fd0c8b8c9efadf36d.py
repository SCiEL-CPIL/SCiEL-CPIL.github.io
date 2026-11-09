import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 128
    
    # Precompute bad masks as a 3D numpy array: bad_mask[a, b, 0:2] for a<b
    # Index: a in 1..128, b in 1..128, but only a<b and a+b<=128 are meaningful
    bad_mask = np.zeros((129, 129, 2), dtype=np.uint64)
    for a in range(1, 129):
        for b in range(a+1, 129):
            s = a + b
            if s > 128:
                continue
            max_c = 128 - b
            for c in range(1, max_c + 1):
                if c == a or c == b or c == s:
                    continue
                if c <= 64:
                    bad_mask[a, b, 0] |= (1 << (c-1))
                else:
                    bad_mask[a, b, 1] |= (1 << (c-65))
    
    best_size = 0
    best_set = set()
    
    for trial in range(5000):
        if time.time() > deadline - 3:
            break
        
        random.seed(trial)
        elements = list(range(1, 129))
        random.shuffle(elements)
        
        # s_mask: boolean array of length 129, s_mask[i] = True if i in S
        s_mask = np.zeros(129, dtype=bool)
        # s_lo, s_hi: bitmask representation
        s_lo = 0
        s_hi = 0
        s_set = set()
        active_pairs = set()
        
        for v in elements:
            if time.time() > deadline - 3:
                break
            
            # Check 1: v not forbidden by any active pair
            valid = True
            for (a, b) in active_pairs:
                ml = bad_mask[a, b, 0]
                mh = bad_mask[a, b, 1]
                if v <= 64:
                    if ml & (1 << (v-1)):
                        valid = False
                        break
                else:
                    if mh & (1 << (v-65)):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: new active pairs involving v
            # Get all x in s_set as a list
            x_list = list(s_set)
            if x_list:
                x_arr = np.array(x_list)
                # For each x, compute a2, b2, s2
                a2_arr = np.minimum(v, x_arr)
                b2_arr = np.maximum(v, x_arr)
                s2_arr = a2_arr + b2_arr
                # Filter: s2 <= 128 and s2 in s_set
                valid_idx = (s2_arr <= 128) & s_mask[s2_arr.astype(int)]
                if np.any(valid_idx):
                    a2_v = a2_arr[valid_idx].astype(int)
                    b2_v = b2_arr[valid_idx].astype(int)
                    s2_v = s2_arr[valid_idx].astype(int)
                    # For each such pair, check if bad_mask[a2,b2] intersects s_set (minus a2,b2,s2)
                    for i in range(len(a2_v)):
                        a2 = a2_v[i]
                        b2 = b2_v[i]
                        s2 = s2_v[i]
                        ml = bad_mask[a2, b2, 0]
                        mh = bad_mask[a2, b2, 1]
                        # Check intersection with s_set
                        # Build mask for s_set
                        # We can check: is there any y in s_set, y not in {a2,b2,s2}, with bit set in bad_mask?
                        # Equivalent: (s_lo & ml) or (s_hi & mh) has a bit set that's not a2,b2,s2
                        combined_lo = s_lo & ml
                        combined_hi = s_hi & mh
                        # Clear bits for a2, b2, s2
                        for elem in [a2, b2, s2]:
                            if elem <= 64:
                                combined_lo &= ~(1 << (elem-1))
                            else:
                                combined_hi &= ~(1 << (elem-65))
                        if combined_lo or combined_hi:
                            valid = False
                            break
                    if not valid:
                        continue
            
            if valid:
                s_set.add(v)
                s_mask[v] = True
                if v <= 64:
                    s_lo |= (1 << (v-1))
                else:
                    s_hi |= (1 << (v-65))
                # Update active_pairs
                for x in s_set:
                    if x == v:
                        continue
                    a2 = min(v, x)
                    b2 = max(v, x)
                    s2 = a2 + b2
                    if s2 <= 128 and s2 in s_set:
                        active_pairs.add((a2, b2))
        
        if len(s_set) > best_size:
            best_size = len(s_set)
            best_set = s_set
    
    return list(best_set)
