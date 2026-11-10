import time
import random

N = 256
DEADLINE = 85.0  # 90s limit, keep margin

def _precompute():
    # 3-APs
    aps3 = []
    # Store as (a, b, c) and mask of all 3, and masks of pairs
    # Actually, for score calc, we need to know if the other 2 are in A.
    # Precompute for each element z, list of (ap_index, mask_other_two, ap_tuple)
    
    # 4-APs
    aps4 = []
    # Precompute for each element z, list of (ap_index, mask_other_three, ap_tuple)
    
    # Generate 3-APs
    elem_to_aps3 = [[] for _ in range(N+1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(aps3)
            aps3.append((a, b, c))
            # mask of all 3
            mask_all = (1 << a) | (1 << b) | (1 << c)
            # For each element, store the mask of the other two
            # a: other are b, c
            elem_to_aps3[a].append((idx, (1 << b) | (1 << c)))
            # b: other are a, c
            elem_to_aps3[b].append((idx, (1 << a) | (1 << c)))
            # c: other are a, b
            elem_to_aps3[c].append((idx, (1 << a) | (1 << b)))
            
    # Generate 4-APs
    elem_to_aps4 = [[] for _ in range(N+1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(aps4)
            aps4.append((a, b, c, e))
            mask_all = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            # For each element, store mask of other three
            # a: b,c,e
            elem_to_aps4[a].append((idx, (1 << b) | (1 << c) | (1 << e)))
            # b: a,c,e
            elem_to_aps4[b].append((idx, (1 << a) | (1 << c) | (1 << e)))
            # c: a,b,e
            elem_to_aps4[c].append((idx, (1 << a) | (1 << b) | (1 << e)))
            # e: a,b,c
            elem_to_aps4[e].append((idx, (1 << a) | (1 << b) | (1 << c)))
            
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def construct():
    start_time = time.time()
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute()
    
    # Initial greedy set
    A_mask = 0
    for x in range(1, N + 1):
        # Check if adding x creates 4-AP
        valid = True
        for idx, mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
            
    # Current score
    def calc_score(mask):
        score = 0
        for a, b, c in aps3:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                score += 1
        return score

    current_score = calc_score(A_mask)
    best_score = current_score
    best_mask = A_mask
    
    # Local Search
    # Precompute list of elements in A and not in A for faster iteration?
    # We can just iterate 1..N and check bit.
    
    improved = True
    # To avoid getting stuck in local optima, we might want to accept some neutral moves or random restarts
    # But hill climbing is a good start.
    
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        # Iterate through all x in A
        # To make it deterministic and fast, we can shuffle the order of x and y
        # But full pass is better for finding all improvements
        
        # Get list of x in A
        x_in_A = [i for i in range(1, N+1) if (A_mask >> i) & 1]
        random.shuffle(x_in_A)
        
        # Get list of y not in A
        y_not_in_A = [i for i in range(1, N+1) if not ((A_mask >> i) & 1)]
        random.shuffle(y_not_in_A)
        
        for x in x_in_A:
            if time.time() - start_time > DEADLINE - 5:
                break
            for y in y_not_in_A:
                if x == y: continue
                
                # Check validity: A' = A \ {x} + {y} is 4-AP free
                # Condition: No 4-AP Q containing y such that x not in Q and Q\{y} subset of A
                valid = True
                for idx, mask_other3 in elem_to_aps4[y]:
                    # Check if x is in this 4-AP
                    # We need to know if x is in Q.
                    # We didn't store the full tuple in elem_to_aps4.
                    # We can check: if mask_other3 has bit x set, then x is in Q\{y}, so x is in Q.
                    # If x is in Q, then Q\{y} contains x. Since we remove x, Q\{y} is NOT subset of A\{x}.
                    # So if x is in Q, this Q is safe (cannot form).
                    if (mask_other3 >> x) & 1:
                        continue
                    # x is not in Q. Check if Q\{y} subset of A
                    if (A_mask & mask_other3) == mask_other3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate Delta
                # Loss: 3-APs containing x where other 2 in A
                loss = 0
                for idx, mask_other2 in elem_to_aps3[x]:
                    if (A_mask & mask_other2) == mask_other2:
                        loss += 1
                        
                # Gain: 3-APs containing y where other 2 in A \ {x}
                gain = 0
                for idx, mask_other2 in elem_to_aps3[y]:
                    # Check if other 2 are in A
                    if (A_mask & mask_other2) == mask_other2:
                        # Check if x is in this 3-AP
                        # If x is in mask_other2, then x is in P\{y}.
                        # If x is in P\{y}, then P\{y} is NOT subset of A\{x}.
                        if not ((mask_other2 >> x) & 1):
                            gain += 1
                            # Wait, if x is y? No, x in A, y not in A.
                            # What if x is one of the other 2?
                            # If x is in mask_other2, then one of the other elements is x.
                            # Since we remove x, that element is gone.
                            # So we only count if x is NOT in mask_other2.
                            # This logic is correct.
                            # But wait, if x is the element y? No.
                            # If x is one of the other 2 elements of the 3-AP containing y.
                            # Then that 3-AP is broken.
                            # So gain only if x is not in the other 2.
                            # Correct.
                        
                delta = gain - loss
                if delta > 0:
                    # Apply swap
                    A_mask &= ~(1 << x)
                    A_mask |= (1 << y)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask
                    # Break inner loop? Or continue?
                    # If we break, we restart the x loop?
                    # Usually better to continue checking other y for this x?
                    # Or just break and let the outer while loop handle it.
                    # Breaking allows us to pick the best y for this x?
                    # No, we just pick the first improving y.
                    # Let's break to update state immediately.
                    break
                    
    # Reconstruct set from best_mask
    result = [i for i in range(1, N+1) if (best_mask >> i) & 1]
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the entry point is exactly 'construct'
# The code above defines it.
# EVOLVE-BLOCK-END
