import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 50  # Safety margin
    
    # Precompute compatibility bitsets
    # compat[i] is an integer where bit j is 1 if lcm(i, j) <= 512
    compat = [0] * (limit + 1)
    for i in range(1, limit + 1):
        mask = 0
        for j in range(1, limit + 1):
            if i * j // gcd(i, j) <= limit:
                mask |= (1 << j)
        compat[i] = mask

    # Helper to check if x is compatible with set S (represented as bitset S_mask)
    # x is compatible with S if (compat[x] & S_mask) == S_mask
    def is_compatible_with_set(x, S_mask):
        return (compat[x] & S_mask) == S_mask

    # Initialize with a greedy set
    # Start with 1..22 as a strong core
    S = list(range(1, 23))
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    # Greedily add more if possible
    for x in range(23, limit + 1):
        if time.time() > deadline - 5:
            break
        if is_compatible_with_set(x, S_mask):
            S.append(x)
            S_mask |= (1 << x)

    best_size = len(S)
    best_S = S[:]
    best_S_mask = S_mask

    # Local Search
    # We will try 1-for-1 swaps to improve structure, and 2-for-3 swaps to increase size.
    # To keep it fast, we limit the number of iterations and use randomization.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try 2-for-3 swaps
        # To make this efficient, we iterate through pairs in S and look for 3 candidates
        # This is O(|S|^2 * N^3) which is too slow.
        # Instead, let's try a simpler heuristic:
        # 1. Pick a random pair (u, v) from S.
        # 2. Compute S_temp = S \ {u, v}.
        # 3. Find all candidates C that are compatible with S_temp.
        # 4. Try to find 3 mutually compatible elements in C \ S.
        
        if len(S) < 2:
            break
            
        # Randomly select a pair to remove
        u_idx = random.randint(0, len(S) - 1)
        v_idx = random.randint(0, len(S) - 1)
        if u_idx == v_idx:
            continue
        u = S[u_idx]
        v = S[v_idx]
        
        S_mask_temp = S_mask & ~(1 << u) & ~(1 << v)
        
        # Find candidates compatible with S_mask_temp
        candidates = []
        for x in range(1, limit + 1):
            if (S_mask & (1 << x)): # if x is in S
                continue
            if is_compatible_with_set(x, S_mask_temp):
                candidates.append(x)
                
        # Try to find 3 mutually compatible in candidates
        # Since candidates might be large, we limit the search
        # We can just pick 3 random ones and check? Or greedy pick?
        # Let's try to greedily pick 3 from candidates
        # Sort candidates by "flexibility" (number of compatible elements) might help, but let's keep it simple
        random.shuffle(candidates)
        
        found_three = False
        # Limit candidates to check to avoid slowdown
        check_limit = min(100, len(candidates))
        if len(candidates) >= 3:
            # Try to find a clique of size 3 in the induced subgraph of candidates
            # Simple approach: iterate pairs in candidates, check if a 3rd exists
            c_limit = min(20, len(candidates)) # Only check first 20 to keep it fast
            for i in range(c_limit):
                c1 = candidates[i]
                for j in range(i+1, c_limit):
                    c2 = candidates[j]
                    # Check if c1 and c2 are compatible
                    if not (compat[c1] & (1 << c2)):
                        continue
                    # Find c3 compatible with c1, c2, and S_mask_temp
                    # c3 must be in candidates (so compatible with S_mask_temp)
                    # and compatible with c1 and c2
                    # mask_req = compat[c1] & compat[c2]
                    # We need c3 in candidates such that bit c3 is in mask_req
                    mask_req = compat[c1] & compat[c2]
                    # Iterate candidates again?
                    for k in range(j+1, c_limit):
                        c3 = candidates[k]
                        if (mask_req & (1 << c3)):
                            # We found c1, c2, c3
                            # Check if they are distinct from u, v (already true as they are not in S)
                            # Check if they are distinct from each other (indices i, j, k distinct)
                            # Check if the new set is valid
                            new_S = S.copy()
                            # Remove u, v
                            new_S.remove(u)
                            new_S.remove(v)
                            # Add c1, c2, c3
                            new_S.extend([c1, c2, c3])
                            
                            # Verify validity (safety check)
                            valid = True
                            new_mask = 0
                            for x in new_S:
                                if not is_compatible_with_set(x, new_mask):
                                    valid = False
                                    break
                                new_mask |= (1 << x)
                            
                            if valid and len(new_S) > best_size:
                                S = new_S
                                S_mask = new_mask
                                best_size = len(S)
                                best_S = S[:]
                                best_S_mask = S_mask
                                improved = True
                                no_improve_count = 0
                                found_three = True
                                break
                        if found_three:
                            break
                    if found_three:
                        break
                if found_three:
                    break
                    
        if not improved:
            no_improve_count += 1
            # If stuck for a while, try a 1-for-1 swap or random restart
            if no_improve_count > 50:
                # Try 1-for-1 swap to change structure
                # Pick random u in S, random v not in S
                # If S \ {u} U {v} is valid, do it
                # This doesn't increase size, but might help find a 2-for-3 later
                if len(S) > 0:
                    u = random.choice(S)
                    # Find v not in S compatible with S \ {u}
                    S_mask_temp = S_mask & ~(1 << u)
                    possible_v = []
                    for v in range(1, limit + 1):
                        if not (S_mask & (1 << v)):
                            if is_compatible_with_set(v, S_mask_temp):
                                possible_v.append(v)
                    if possible_v:
                        v = random.choice(possible_v)
                        new_S = S.copy()
                        new_S.remove(u)
                        new_S.append(v)
                        # Check validity
                        valid = True
                        new_mask = 0
                        for x in new_S:
                            if not is_compatible_with_set(x, new_mask):
                                valid = False
                                break
                            new_mask |= (1 << x)
                        if valid:
                            S = new_S
                            S_mask = new_mask
                            no_improve_count = 0 # Reset counter as structure changed
                            
    # Return the best set found
    return best_S
