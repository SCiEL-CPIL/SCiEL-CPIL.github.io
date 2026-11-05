import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    n = 256
    S = set()
    sums = set()
    
    # Iterate from largest to smallest
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if x divides any existing sum in sums
        # sums contains b+c for b,c in S with b <= c
        valid = True
        for s in sums:
            if s % x == 0:
                valid = False
                break
                
        if valid:
            S.add(x)
            # Update sums with new pairs involving x
            sums.add(2 * x)
            for y in S:
                if y != x:
                    sums.add(x + y)
    
    # Local improvement: try to replace larger elements with smaller ones
    # that don't conflict, potentially allowing more elements overall
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_sorted = sorted(S, reverse=True)
        
        for i, large in enumerate(S_sorted):
            if time.time() > deadline:
                break
            
            # Try replacing 'large' with some smaller x not in S
            # where x doesn't divide any sum in the modified set
            S_without_large = S - {large}
            
            # Compute sums for S_without_large
            sums_wo_large = set()
            S_list_wo = list(S_without_large)
            for j in range(len(S_list_wo)):
                for k in range(j, len(S_list_wo)):
                    sums_wo_large.add(S_list_wo[j] + S_list_wo[k])
            
            # Try candidate replacements from 1 up to large-1
            for x in range(1, large):
                if x in S:
                    continue
                
                # Check if x can be added to S_without_large
                valid = True
                for s in sums_wo_large:
                    if s % x == 0:
                        valid = False
                        break
                
                if valid:
                    # Check: does any element in S_without_large divide x + y or 2x?
                    # Actually we need to check the full constraint for the new set S_without_large ∪ {x}
                    # For every a < b <= c in new_set, a must NOT divide b+c
                    # Since x is smaller than all in S_without_large (we're trying x < large, 
                    # but x might not be smallest... let's just do full check for safety)
                    
                    new_set = S_without_large | {x}
                    new_set_list = sorted(new_set)
                    
                    # Full validation
                    valid_full = True
                    len_new = len(new_set_list)
                    for idx_a in range(len_new):
                        a = new_set_list[idx_a]
                        for idx_b in range(idx_a + 1, len_new):
                            b = new_set_list[idx_b]
                            for idx_c in range(idx_b, len_new):
                                c = new_set_list[idx_c]
                                if (b + c) % a == 0:
                                    valid_full = False
                                    break
                            if not valid_full:
                                break
                        if not valid_full:
                            break
                    
                    if valid_full:
                        # Perform the swap
                        S = new_set
                        improved = True
                        break
    
    return list(S)
# EVOLVE-BLOCK-END
