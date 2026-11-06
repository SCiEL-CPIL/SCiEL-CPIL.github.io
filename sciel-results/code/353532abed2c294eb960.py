import time
import random

def construct():
    N = 256
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    # 4-AP: (a, a+d, a+2d, a+3d)
    
    three_aps = []
    four_aps = []
    
    # Map element to list of 3-AP indices
    elem_to_3ap = {i: [] for i in range(1, N+1)}
    # Map element to list of 4-AP indices
    elem_to_4ap = {i: [] for i in range(1, N+1)}
    
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            elem_to_3ap[a].append(idx)
            elem_to_3ap[b].append(idx)
            elem_to_3ap[c].append(idx)
            
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            idx = len(four_aps)
            four_aps.append((a, b, c, e))
            elem_to_4ap[a].append(idx)
            elem_to_4ap[b].append(idx)
            elem_to_4ap[c].append(idx)
            elem_to_4ap[e].append(idx)

    # Helper to count 3-APs in a set
    def count_3aps(S):
        count = 0
        for a, b, c in three_aps:
            if a in S and b in S and c in S:
                count += 1
        return count

    # Helper to check if adding x creates a 4-AP
    def creates_4ap(x, S):
        for idx in elem_to_4ap[x]:
            a, b, c, d = four_aps[idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(y in S for y in others):
                return True
        return False

    # Helper to get delta 3-APs when adding x
    def delta_3ap_add(x, S):
        delta = 0
        for idx in elem_to_3ap[x]:
            a, b, c = three_aps[idx]
            others = [y for y in (a, b, c) if y != x]
            if all(y in S for y in others):
                delta += 1
        return delta

    # Helper to get delta 3-APs when removing x
    def delta_3ap_remove(x, S):
        delta = 0
        for idx in elem_to_3ap[x]:
            a, b, c = three_aps[idx]
            others = [y for y in (a, b, c) if y != x]
            if all(y in S for y in others):
                delta -= 1
        return delta

    start_time = time.time()
    best_score = -1
    best_set = set()
    
    # Sort elements by 3-AP potential for greedy init
    potential = {x: len(elem_to_3ap[x]) for x in range(1, N+1)}
    sorted_elems = sorted(range(1, N+1), key=lambda x: -potential[x])
    
    for _ in range(20): # Restart count
        if time.time() - start_time > 50:
            break
            
        # Greedy Init
        S = set()
        for x in sorted_elems:
            if not creates_4ap(x, S):
                S.add(x)
        
        score = count_3aps(S)
        
        # Local Search
        improved = True
        while improved and time.time() - start_time < 55:
            improved = False
            
            # Try additions
            best_add = None
            best_add_delta = 0
            for x in range(1, N+1):
                if x not in S:
                    if not creates_4ap(x, S):
                        d = delta_3ap_add(x, S)
                        if d > best_add_delta:
                            best_add_delta = d
                            best_add = x
            
            if best_add is not None:
                S.add(best_add)
                score += best_add_delta
                improved = True
                continue
            
            # Try removals
            best_rem = None
            best_rem_delta = 0 # We want to remove if it allows better additions later? 
            # Actually, pure hill climbing on score might get stuck.
            # Let's try swaps if no add is possible or beneficial.
            
            # Try swaps
            best_swap = None
            best_swap_delta = 0
            # This is O(|S| * (N-|S|)) which might be slow.
            # Let's sample some swaps.
            for x in list(S):
                for _ in range(10): # Sample 10 candidates for y
                    y = random.randint(1, N)
                    if y in S:
                        continue
                    # Check if S - {x} + {y} is valid
                    # This requires checking 4-APs involving y in S\{x}
                    # And ensuring x removal doesn't break anything (it doesn't)
                    
                    # Efficient check:
                    # 1. Remove x from S temporarily? No, just check y against S\{x}
                    # Creates 4AP if any 4-AP with y has other 3 in S\{x}
                    valid = True
                    for idx in elem_to_4ap[y]:
                        a, b, c, d = four_aps[idx]
                        others = [z for z in (a, b, c, d) if z != y]
                        # Check if all others are in S AND none of them is x? 
                        # Actually, if x is in others, it's not in S\{x}, so that AP is safe.
                        # If x is not in others, we need all others in S.
                        if x in others:
                             # The AP is broken by removing x, so it's safe
                             pass
                        else:
                             if all(z in S for z in others):
                                 valid = False
                                 break
                    if not valid:
                        continue
                    
                    # Calculate delta
                    # Remove x: - count 3APs containing x with others in S
                    # Add y: + count 3APs containing y with others in S\{x}
                    
                    rem_delta = 0
                    for idx in elem_to_3ap[x]:
                        a, b, c = three_aps[idx]
                        others = [z for z in (a, b, c) if z != x]
                        if all(z in S for z in others):
                            rem_delta -= 1
                            
                    add_delta = 0
                    for idx in elem_to_3ap[y]:
                        a, b, c = three_aps[idx]
                        others = [z for z in (a, b, c) if z != y]
                        # Check if all others in S\{x}
                        # If x in others, then one is missing, so no 3AP formed
                        if x in others:
                             continue
                        if all(z in S for z in others):
                             add_delta += 1
                             
                    total_delta = rem_delta + add_delta
                    if total_delta > best_swap_delta:
                        best_swap_delta = total_delta
                        best_swap = (x, y)
            
            if best_swap is not None:
                x, y = best_swap
                S.remove(x)
                S.add(y)
                score += best_swap_delta
                improved = True
                continue

            # If no add, no swap, try a random removal to escape local optimum?
            # Or just break.
            if len(S) > 0:
                # Random removal to allow re-addition
                x = random.choice(list(S))
                S.remove(x)
                score += delta_3ap_remove(x, S)
                improved = True
                continue
                
        if score > best_score:
            best_score = score
            best_set = S.copy()
            
    return sorted(best_set)
