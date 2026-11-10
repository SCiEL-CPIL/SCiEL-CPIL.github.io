    import numpy as np

    def prime_number_theorem_partial_function():
        K = 200000
        limit = 1.0001 / 10.0
        
        S = []
        current_sum = 0.0
        
        for k in range(K, 1, -1):
            inv_k = 1.0 / k
            if current_sum + inv_k <= limit:
                S.append(k)
                current_sum += inv_k
            else:
                break
                
        if not S:
            return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)
            
        f1 = 10.0 * current_sum
        
        domain = np.array([1] + S, dtype=np.int64)
        values = np.zeros(len(domain), dtype=np.float64)
        values[0] = f1
        values[1:] = -10.0
        
        max_x = 10 * K
        counts = np.zeros(max_x + 1, dtype=np.int64)
        
        for k in S:
            counts[k::k] += 1
            
        H = np.cumsum(counts)
        x = np.arange(1, max_x + 1, dtype=np.float64)
        M = f1 * x - 10.0 * H
        
        if np.max(M) > 1.0001:
            return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)
            
        a = -np.sum(values * np.log(domain) / domain)
        return domain, values
    