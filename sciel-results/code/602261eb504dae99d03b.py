import time

def construct():
    start_time = time.time()
    # Allow 100s for computation
    deadline = start_time + 100.0
    
    # Start with a moderate K
    K = 5 * 10**6
    
    # Check if we have enough time for a larger K
    # Sieve is O(K log K).
    # If we estimate K=10^7 takes 20s, K=5*10^6 takes 10s.
    # We have 100s. So we can afford K=10^7.
    # But memory?
    # Let's try K=10^7.
    K = 10**7
    
    sigma = [0] * (K + 1)
    
    # Sieve
    # Optimize by using local variables
    sig = sigma
    for i in range(1, K + 1):
        for j in range(i, K + 1, i):
            sig[j] += i
        if i % 10**6 == 0:
            if time.time() > deadline:
                break
                
    if time.time() > deadline:
        # Fallback to smaller result if sieve didn't finish?
        # This is tricky. If we break, sigma is incomplete.
        # Better to just pick a safe K.
        # Let's revert to K=2*10^6 which is very safe.
        pass 

    # Actually, let's just use K=2*10^6 to be absolutely safe on time and memory.
    # The jump from 3 to 5 was with 10^6.
    # 2*10^6 might find more.
    
    K = 2 * 10**6
    sigma = [0] * (K + 1)
    for i in range(1, K + 1):
        for j in range(i, K + 1, i):
            sigma[j] += i
            
    m_counts = {}
    m_ks = {}
    
    for k in range(1, K + 1):
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in m_counts:
            m_counts[M] = 0
            m_ks[M] = []
        m_counts[M] += 1
        m_ks[M].append(k)
        
    best_M = None
    best_count = 0
    best_ks = []
    
    for M, count in m_counts.items():
        if count > best_count:
            best_count = count
            best_M = M
            best_ks = m_ks[M]
            
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_ks}
