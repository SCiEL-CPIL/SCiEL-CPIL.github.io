import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Current best from previous program
    best_val = 30
    best_list = [224, 240]
    
    # Search for better solutions with max < 240
    # M is the maximum element in the set
    for M in range(211, 240):
        if time.time() > deadline:
            break
        if M - 210 >= best_val:
            break
            
        S = list(range(211, M + 1))
        found = False
        
        # Check k=1
        # x1 = 210 * m^2. Only m=1 gives 210 (invalid). m=2 gives 840 (>M).
        # So no k=1 solution for M < 840.
        
        # Check k=2
        # x1 * x2 = 210 * m^2
        # x1, x2 in S, distinct
        # m^2 = x1*x2/210. So x1*x2 must be divisible by 210 and result must be a square.
        # Iterate pairs in S
        for i in range(len(S)):
            if time.time() > deadline: break
            x1 = S[i]
            for j in range(i + 1, len(S)):
                x2 = S[j]
                prod = x1 * x2
                if prod % 210 != 0:
                    continue
                q = prod // 210
                s = math.isqrt(q)
                if s * s == q:
                    # Valid
                    val = M - 210
                    if val < best_val:
                        best_val = val
                        best_list = [x1, x2]
                    found = True
                    break
            if found:
                break
        
        if found:
            return best_list

        # Check k=3
        # x1 * x2 * x3 = 210 * m^2
        # Iterate triples
        for i in range(len(S)):
            if time.time() > deadline: break
            x1 = S[i]
            for j in range(i + 1, len(S)):
                if time.time() > deadline: break
                x2 = S[j]
                for k in range(j + 1, len(S)):
                    x3 = S[k]
                    prod = x1 * x2 * x3
                    if prod % 210 != 0:
                        continue
                    q = prod // 210
                    s = math.isqrt(q)
                    if s * s == q:
                        val = M - 210
                        if val < best_val:
                            best_val = val
                            best_list = [x1, x2, x3]
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            return best_list

    return best_list
