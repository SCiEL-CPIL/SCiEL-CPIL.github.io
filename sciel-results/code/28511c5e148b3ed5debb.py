import time
import random

N = 256
DEADLINE = 85

def construct():
    start_time = time.time()
    
    # Precomputation
    # 3-APs
    elem_to_3aps = {i: [] for i in range(1, N+1)}
    v_to_3ap_others = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            ap3 = (a, b, c)
            elem_to_3aps[a].append(ap3)
            elem_to_3aps[b].append(ap3)
            elem_to_3aps[c].append(ap3)
            
            v_to_3ap_others[a].append((b, c))
            v_to_3ap_others[b].append((a, c))
            v_to_3ap_others[c].append((a, b))
            
    # 4-APs
    elem_to_4aps = {i: [] for i in range(1, N+1)}
    v_to_4ap_others = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            ap4 = (a, b, c, e)
            elem_to_4aps[a].append(ap4)
            elem_to_4aps[b].append(ap4)
            elem_to_4aps[c].append(ap4)
            elem_to_4aps[e].append(ap4)
            
            v_to_4ap_others[a].append((b, c, e))
            v_to_4ap_others[b].append((a, c, e))
            v_to_4ap_others[c].append((a, b, e))
            v_to_4ap_others[e].append((a, b, c))

    # Helper to count 3-APs in a set
    def count_3aps(S):
        count = 0
        for a in S:
            for b, c in v_to_3ap_others[a]:
                # Avoid double counting?
                # If we just count all 3-APs containing a, we count each 3-AP 3 times.
                # So divide by 3 at the end?
                # Or just check if a < b < c and a is the min?
                # Simpler: iterate all 3-APs and check membership.
                pass
        # Let's just use the global list for accuracy in initial count
        global_ap3s = []
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2*d + 1):
                global_ap3s.append((a, a+d, a+2*d))
        for a, b, c in global_ap3s:
            if a in S and b in S and c in S:
                count += 1
        return count

    # Initial construction (Seed)
    A = set()
    for x in range(1, N+1):
        if all(not all(x-j*d in A for j in (1,2,3)) for d in range(1,(x-1)//3+1)):
            A.add(x)
            
    current_score = count_3aps(A)
    
    # Local Search
    while time.time() - start_time < DEADLINE:
        improved = False
        best_gain = 0
        best_swap = None
        
        us = list(A)
        vs = [x for x in range(1, N+1) if x not in A]
        random.shuffle(us)
        random.shuffle(vs)
        
        for u in us:
            for v in vs:
                if u == v: continue
                
                # Check 4-AP constraint
                valid = True
                for others in v_to_4ap_others[v]:
                    # Check if all others are in A and none is u
                    if all(x in A for x in others) and u not in others:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate 3-AP change
                loss = 0
                for ap3 in elem_to_3aps[u]:
                    if all(x in A for x in ap3):
                        loss += 1
                        
                gain = 0
                for others in v_to_3ap_others[v]:
                    if all(x in A for x in others) and u not in others:
                        gain += 1
                
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_swap = (u, v)
                    
        if best_swap:
            u, v = best_swap
            A.remove(u)
            A.add(v)
            current_score += best_gain
            improved = True
        else:
            # Random kick
            if A:
                u = random.choice(list(A))
                vs = [x for x in range(1, N+1) if x not in A]
                if vs:
                    v = random.choice(vs)
                    valid = True
                    for others in v_to_4ap_others[v]:
                        if all(x in A for x in others) and u not in others:
                            valid = False
                            break
                    if valid:
                        loss = 0
                        for ap3 in elem_to_3aps[u]:
                            if all(x in A for x in ap3):
                                loss += 1
                        gain = 0
                        for others in v_to_3ap_others[v]:
                            if all(x in A for x in others) and u not in others:
                                gain += 1
                        A.remove(u)
                        A.add(v)
                        current_score += gain - loss
                        improved = True
            if not improved:
                # Restart
                A = set()
                for x in range(1, N+1):
                    if all(not all(x-j*d in A for j in (1,2,3)) for d in range(1,(x-1)//3+1)):
                        A.add(x)
                current_score = count_3aps(A)

    return sorted(A)
