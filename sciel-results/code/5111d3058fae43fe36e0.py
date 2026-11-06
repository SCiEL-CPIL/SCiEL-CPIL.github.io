import time
import random

def construct():
    start_time = time.time()
    deadline = 60  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            ap3s.append((a, a+d, a+2*d))
            
    ap4s = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap4s.append((a, a+d, a+2*d, a+3*d))
            
    # Map element to AP indices
    elem_to_ap3 = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)

    def count_3aps(S):
        count = 0
        for idx in range(len(ap3s)):
            a, b, c = ap3s[idx]
            if a in S and b in S and c in S:
                count += 1
        return count

    def is_valid_4ap_free(S):
        # Check if S contains any 4-AP
        # Optimization: Only check 4-APs where all 4 elements are in S
        # For local search swaps, we will do a more targeted check.
        for idx in range(len(ap4s)):
            a, b, c, d = ap4s[idx]
            if a in S and b in S and c in S and d in S:
                return False
        return True

    def greedy_construct():
        S = set()
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        while candidates:
            if time.time() - start_time > deadline - 10:
                break
                
            best_x = -1
            best_gain = -1
            valid_candidates = []
            
            for x in candidates:
                if x in S:
                    continue
                
                # Check 4-AP constraint
                valid = True
                for idx in elem_to_ap4[x]:
                    a, b, c, d = ap4s[idx]
                    others = [a, b, c, d]
                    others.remove(x)
                    if all(o in S for o in others):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate Gain
                gain = 0
                for idx in elem_to_ap3[x]:
                    a, b, c = ap3s[idx]
                    others = [a, b, c]
                    others.remove(x)
                    if all(o in S for o in others):
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    valid_candidates = [x]
                elif gain == best_gain:
                    valid_candidates.append(x)
                    
            if best_x == -1:
                break
                
            # Pick randomly from best candidates to break symmetry
            if len(valid_candidates) > 1:
                best_x = random.choice(valid_candidates)
            
            S.add(best_x)
            candidates.remove(best_x)
            
        return S

    def local_search(S):
        current_score = count_3aps(S)
        improved = True
        
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            
            # Calculate cost(x) for x in S
            # cost(x) = number of 3-APs in S containing x
            costs = {x: 0 for x in S}
            for x in S:
                for idx in elem_to_ap3[x]:
                    a, b, c = ap3s[idx]
                    if a in S and b in S and c in S:
                        costs[x] += 1
            
            # Calculate benefit(y) for y not in S
            # benefit(y) = number of 3-APs that would be completed by y
            benefits = {}
            not_in_S = [x for x in range(1, N+1) if x not in S]
            for y in not_in_S:
                # Check if y can be added without creating 4-AP
                # (This check is also done in swap logic, but we can pre-filter)
                # Actually, we need to check 4-AP validity for the specific swap (x, y)
                # because removing x might allow y.
                # So we calculate benefit(y) assuming x is removed.
                # benefit_swap(y, x) = number of 3-APs in (S \ {x}) \cup {y} that contain y
                # This is equal to the number of 3-APs containing y where the other 2 are in S \ {x}
                
                # We will iterate swaps directly to be safe and accurate.
                pass
                
            # Find best swap
            best_swap = None
            best_delta = 0
            
            for x in S:
                cx = costs[x]
                for y in not_in_S:
                    # Check 4-AP validity of S' = S \ {x} \cup {y}
                    # A 4-AP is formed if there is a 4-AP containing y such that other 3 are in S \ {x}
                    valid = True
                    for idx in elem_to_ap4[y]:
                        a, b, c, d = ap4s[idx]
                        others = [a, b, c, d]
                        others.remove(y)
                        # Check if all others are in S \ {x}
                        # This means they are in S AND none of them is x
                        all_in_S_minus_x = True
                        for o in others:
                            if o not in S or o == x:
                                all_in_S_minus_x = False
                                break
                        if all_in_S_minus_x:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate benefit(y) given x is removed
                    # Number of 3-APs containing y where other 2 are in S \ {x}
                    by = 0
                    for idx in elem_to_ap3[y]:
                        a, b, c = ap3s[idx]
                        others = [a, b, c]
                        others.remove(y)
                        all_in_S_minus_x = True
                        for o in others:
                            if o not in S or o == x:
                                all_in_S_minus_x = False
                                break
                        if all_in_S_minus_x:
                            by += 1
                    
                    delta = by - cx
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (x, y)
            
            if best_swap:
                x, y = best_swap
                S.remove(x)
                S.add(y)
                current_score += best_delta
                improved = True
                # Optional: Restart local search from new state to avoid local minima
                # But simple hill climbing is a good start.
                
        return S, current_score

    # Run greedy construction multiple times with different seeds/orders
    best_S = set()
    best_score = -1
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 15:
            break
        
        S = greedy_construct()
        S, score = local_search(S)
        
        if score > best_score:
            best_score = score
            best_S = S.copy()
            
    # Final verification and return
    if time.time() - start_time > deadline - 2:
        # Emergency return
        return sorted(best_S)
        
    # One last check to ensure no 4-APs
    # (Should be guaranteed by local search, but just in case)
    for idx in range(len(ap4s)):
        a, b, c, d = ap4s[idx]
        if a in best_S and b in best_S and c in best_S and d in best_S:
            # This shouldn't happen, but if it does, we have a bug.
            # Remove one element to fix?
            best_S.remove(d)
            break

    return sorted(best_S)
