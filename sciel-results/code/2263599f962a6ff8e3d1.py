import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Max n is 30
    A = np.zeros((31, 31), dtype=int)
    current_n = 1
    
    for target_n in range(2, 31):
        if time.time() - start_time > time_limit:
            break
        
        success = False
        # Try multiple random initializations for the new vertex
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            v = target_n - 1
            # Initialize edges from v to 0..v-1
            for u in range(v):
                A[v, u] = random.randint(0, 1)
                A[u, v] = A[v, u]
            
            # Repair
            max_steps = 2000
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                
                # Check for violations involving v
                # We check all u in 0..v-1 as potential partners in the same part
                violation = False
                for u in range(v):
                    for c in [0, 1]:
                        # Find candidates a,b,c,d
                        # Candidates must be neighbors of v and u in color c
                        # and not v or u
                        mask_v = (A[v, :] == c)
                        mask_u = (A[u, :] == c)
                        mask_v[v] = 0
                        mask_u[u] = 0
                        candidates = np.where(mask_v & mask_u)[0]
                        
                        if len(candidates) >= 4:
                            cand_list = candidates.tolist()
                            # Check for K_2,2 in candidates
                            found_k22 = False
                            # Iterate pairs (a,b)
                            for i in range(len(cand_list)):
                                a = cand_list[i]
                                # Pre-compute common neighbors of a in candidates with color c?
                                # Just iterate b
                                for j in range(i+1, len(cand_list)):
                                    b = cand_list[j]
                                    # Find c,d in candidates \ {a,b} such that
                                    # A[a,c]=A[a,d]=A[b,c]=A[b,d]=c
                                    common = []
                                    for k in cand_list:
                                        if k == a or k == b: continue
                                        if A[a, k] == c and A[b, k] == c:
                                            common.append(k)
                                    if len(common) >= 2:
                                        found_k22 = True
                                        # Repair: flip edge (v, a)
                                        A[v, a] = 1 - A[v, a]
                                        A[a, v] = A[v, a]
                                        violation = True
                                        break
                                if found_k22:
                                    break
                            if found_k22:
                                break
                    if violation:
                        break
                if not violation:
                    success = True
                    break
            if success:
                break
        
        if success:
            current_n = target_n
        else:
            # Failed to construct target_n
            break
            
    # Extract colors
    colors = []
    for i in range(current_n):
        for j in range(i+1, current_n):
            colors.append(A[i, j])
            
    return {'n': current_n, 'colors': colors}
