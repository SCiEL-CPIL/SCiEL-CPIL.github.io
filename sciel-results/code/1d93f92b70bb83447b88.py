import time

def construct():
    start_time = time.time()
    
    # Search up to N
    N = 3_000_000
    
    # Sieve for sigma
    sigma = [1] * (N + 1)
    sig = sigma
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sig[j] += i
            
    # Find collisions
    m_to_ks = {}
    best_M = 0
    best_count = 0
    best_ks = []
    
    for k in range(1, N + 1):
        M_val = k * sig[k]
        
        if M_val in m_to_ks:
            lst = m_to_ks[M_val]
            lst.append(k)
            if len(lst) > best_count:
                best_count = len(lst)
                best_M = M_val
                best_ks = lst
        else:
            m_to_ks[M_val] = [k]
            
    return {'M': best_M, 'k': best_ks}
