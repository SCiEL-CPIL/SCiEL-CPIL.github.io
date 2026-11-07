import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_M = 333312
    best_ks = [336, 372, 434]
    
    MAX_M = 10**18
    
    for B in [3000000, 5000000, 8000000, 12000000, 16000000]:
        if time.time() > deadline - 25:
            break
        
        sigma = [0] * (B + 1)
        for i in range(1, B + 1):
            for j in range(i, B + 1, i):
                sigma[j] += i
        
        m_to_ks = defaultdict(list)
        for k in range(1, B + 1):
            s = sigma[k]
            M = k * s
            if M <= MAX_M and k * k <= M:
                m_to_ks[M].append(k)
        
        for M, ks in m_to_ks.items():
            if len(ks) > len(best_ks):
                best_ks = ks
                best_M = M
        
        if time.time() > deadline - 10:
            break
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
