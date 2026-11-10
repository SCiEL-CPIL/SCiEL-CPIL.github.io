import numpy as np

def prime_number_theorem_partial_function():
    best_a = -1
    best_K = 0
    best_C = 0
    
    # Search K from 2 to 1000
    # We can go higher if we optimize the G(x) calculation
    for K in range(2, 1001):
        # Calculate L_K = sum_{k=2}^K log(k)/k
        ks = np.arange(2, K + 1)
        L_K = np.sum(np.log(ks) / ks)
        
        if L_K == 0:
            continue
            
        # Calculate G(x) = sum_{k=1}^K {x/k} for x in [1, 10K]
        # G(x) = H_K * x - D(x)
        # D(x) = sum_{k=1}^K floor(x/k)
        
        # Efficient calculation of D(x) for all x in [1, 10K]
        # D(x) = sum_{k=1}^K floor(x/k)
        # We can compute floor(x/k) for all k, x using vectorization
        # But memory is K * 10K.
        # For K=1000, 10^7. OK.
        
        x_vals = np.arange(1, 10*K + 1)
        
        # To save memory and time, we can compute D(x) incrementally?
        # Or just use the matrix if K is small enough.
        # Let's try matrix for K <= 500.
        # For K > 500, we might need a different method or skip.
        
        if K <= 500:
            # domain is 1..K
            # floors[i, j] = floor(x_vals[j] / (i+1))
            # k ranges 1..K
            # k = 1..K
            # x = 1..10K
            
            # We need sum_{k=1}^K floor(x/k)
            # Let's create a matrix of k values
            # k_col = (K, 1)
            # x_row = (1, 10K)
            
            # This might be slow in Python loop if not careful.
            # But 500 * 5000 = 2.5M elements.
            
            # Alternative:
            # D(x) = sum_{k=1}^K floor(x/k)
            # We can iterate k and add to D
            D = np.zeros(10*K + 1, dtype=np.float64)
            for k in range(1, K + 1):
                D[1:] += np.floor(x_vals / k)
            
            H_K = np.sum(1.0 / np.arange(1, K + 1))
            G = H_K * x_vals - D
            G_max = np.max(G)
            
            if G_max > 0:
                C_max = 1.0 / G_max
            else:
                C_max = 10.0
            
            if C_max > 10:
                C_max = 10.0
                
            a = C_max * L_K
            
            if a > best_a:
                best_a = a
                best_K = K
                best_C = C_max
                
    # Construct solution for best_K, best_C
    if best_K == 0:
        # Fallback
        return np.array([1, 2], dtype=np.int64), np.array([0, -1], dtype=np.float64)
        
    K = best_K
    C = best_C
    
    # f(k) = -C for k >= 2
    # f(1) = 0 initially
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.full(K, -C, dtype=np.float64)
    values[0] = 0.0 # f(1) initial
    
    # The evaluator will do:
    # 1. Clip values to [-10, 10]
    #    -C is in range if C <= 10.
    # 2. S = sum f(k)/k
    #    S = 0 + sum_{k=2}^K -C/k = -C (H_K - 1)
    # 3. f(1) = f(1) - S = 0 - (-C(H_K-1)) = C(H_K-1)
    # 4. Check M(x)
    #    M(x) = C(H_K-1)x - C sum_{k=2}^K floor(x/k)
    #    = C [ H_K x - x - (D(x) - x) ] = C [ H_K x - D(x) ] = C G(x)
    #    We ensured C G(x) <= 1.
    
    return domain, values
