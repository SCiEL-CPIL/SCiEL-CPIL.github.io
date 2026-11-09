import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192
    
    S = []
    S_set = set()
    
    def is_safe_incremental(a, S_list, S_set_local):
        """Check if adding a to S_list keeps validity. Only checks new constraints involving a."""
        # Check 1: a doesn't divide sum of any 3 distinct elements in S_list
        L = len(S_list)
        if L >= 3:
            # For each pair (u,v) in S_list, check if there exists w in S_list (w != u, w != v)
            # such that u + v + w is divisible by a
            # i.e., w % a == (-u - v) % a
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % a] += 1
            
            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    # Subtract u and v if they match target_rem
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        
        # Check 2: For each existing element b in S_list, b doesn't divide sum of a + 2 distinct others
        # i.e., for each b in S_list, check if there exist u, v in S_list (u != v, u != b, v != b)
        # such that a + u + v is divisible by b
        # i.e., (u + v) % b == (-a) % b
        # We need u, v from S_list \ {b}, distinct from each other
        for b in S_list:
            # Consider elements in S_list excluding b
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue
            
            # We need to find u, v in others (u != v) such that (a + u + v) % b == 0
            # i.e., (u + v) % b == (-a) % b
            target_rem = (-a) % b
            
            # Count remainders of others
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1
            
            # For each pair (u, v) in others, check if (u + v) % b == target_rem
            # But we can do this more efficiently: for each u in others, we need v in others
            # (v != u) such that v % b == (target_rem - u % b) % b
            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                # Subtract u if it matches needed_rem
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    return False
        
        return True
    
    def is_safe_set(S_list):
        """Check if S_list is valid: no element divides sum of 3 distinct others."""
        for idx, a in enumerate(S_list):
            others = [v for i, v in enumerate(S_list) if i != idx]
            L_len = len(others)
            if L_len < 3:
                continue
            
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1
            
            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    # Pass 1: greedy from largest to smallest
    for x in range(n, 0, -1):
        if len(S) >= max_elements:
            break
        if time.time() - start_time > deadline - 5:
            break
        
        if is_safe_incremental(x, S, S_set):
            S.append(x)
            S_set.add(x)
    
    # Pass 2: try to add remaining numbers (from small to large) that fit
    if len(S) < max_elements and time.time() - start_time < deadline - 10:
        for x in range(1, n + 1):
            if len(S) >= max_elements:
                break
            if time.time() - start_time > deadline - 5:
                break
            if x in S_set:
                continue
            if is_safe_incremental(x, S, S_set):
                S.append(x)
                S_set.add(x)
    
    # Pass 3: local search - try to swap elements to increase cardinality or maintain
    if len(S) < max_elements and time.time() - start_time < deadline - 2:
        # Try removing one element and adding two others
        improved = True
        while improved and len(S) < max_elements and time.time() - start_time < deadline - 2:
            improved = False
            S_copy = S[:]
            S_set_copy = S_set.copy()
            
            for i in range(len(S_copy)):
                if time.time() - start_time > deadline - 2:
                    break
                # Try removing S_copy[i]
                S_temp = [v for j, v in enumerate(S_copy) if j != i]
                S_set_temp = set(S_temp)
                
                # Try adding two new elements
                for x1 in range(n, 0, -1):
                    if len(S_temp) >= max_elements:
                        break
                    if x1 in S_set_temp:
                        continue
                    if is_safe_incremental(x1, S_temp, S_set_temp):
                        S_temp2 = S_temp + [x1]
                        S_set_temp2 = S_set_temp | {x1}
                        
                        for x2 in range(n, 0, -1):
                            if len(S_temp2) >= max_elements:
                                break
                            if x2 in S_set_temp2:
                                continue
                            if is_safe_incremental(x2, S_temp2, S_set_temp2):
                                S_temp3 = S_temp2 + [x2]
                                if len(S_temp3) > len(S_copy):
                                    S = S_temp3
                                    S_set = set(S)
                                    S_copy = S[:]
                                    S_set_copy = S_set.copy()
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
    
    return S
