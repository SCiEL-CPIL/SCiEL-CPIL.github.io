import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_val = float('inf')
    best_list = None
    
    # Check k=1
    x1 = 210 * 4
    if 210 < x1 <= 10210:
        val = x1 - 210
        if val < best_val:
            best_val = val
            best_list = [x1]
    
    # Check k=2
    m_max = int(math.sqrt(10210**2 / 210))
    
    for m in range(15, m_max + 1):
        if time.time() > deadline:
            break
        P = 210 * m * m
        limit = int(math.isqrt(P))
        if limit < 211:
            continue
        
        for d in range(limit, 210, -1):
            if P % d == 0:
                x2 = P // d
                if x2 > 210 and x2 != d:
                    val = x2 - 210
                    if val < best_val:
                        best_val = val
                        best_list = [d, x2]
                    break
    
    # Check k=3: ordered search by max value M
    # Search M from 211 upward; stop when M-210 >= best_val
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        if M - 210 >= best_val:
            break
        
        # Check triples (a, b, M) with 211 <= a < b < M
        found = False
        for a in range(211, M):
            if time.time() > deadline:
                break
            for b in range(a + 1, M):
                P = a * b * M
                if P % 210 != 0:
                    continue
                Q = P // 210
                s = math.isqrt(Q)
                if s * s == Q:
                    val = M - 210
                    if val < best_val:
                        best_val = val
                        best_list = [a, b, M]
                    found = True
                    break
            if found:
                break
    
    if best_list is None:
        return [840]
        
    return best_list
