    def check_run_segmented(p, primes_list):
        K = 200
        p_sq = p * p
        rem = [p_sq + i for i in range(1, K + 1)]
        count_ones = 0
        # Check initial ones (unlikely)
        
        for q in primes_list:
            if q > p:
                break
            
            # Check if all are 1
            if count_ones == K:
                break
            
            start_val = ((p_sq + 1 + q - 1) // q) * q
            start_idx = start_val - (p_sq + 1)
            
            if start_idx >= K:
                continue
                
            for j in range(start_idx, K, q):
                if rem[j] % q == 0:
                    while rem[j] % q == 0:
                        rem[j] //= q
                    if rem[j] == 1:
                        count_ones += 1
            
        # Determine max k
        k = 0
        for i in range(K):
            if rem[i] == 1:
                k = i + 1
            else:
                break
        
        return k
