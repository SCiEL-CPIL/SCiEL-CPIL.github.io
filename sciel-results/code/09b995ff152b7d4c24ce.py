    def construct():
        import numpy as np
        import time
        import sys
        
        start_time = time.time()
        deadline = start_time + 140.0
        
        # Generate powerful numbers up to N_limit
        N_limit = 10**7
        powerful = set()
        # v^3 <= N_limit => v <= 215
        max_v = int(N_limit ** (1/3)) + 1
        for v in range(1, max_v + 1):
            v3 = v**3
            max_u = int(np.sqrt(N_limit / v3))
            for u in range(1, max_u + 1):
                p = (u*u) * v3
                if p <= N_limit:
                    powerful.add(p)
        
        P = sorted(powerful)
        P_np = np.array(P, dtype=np.int64)
        
        # Compute all sums a+b with a <= b
        # Using chunks to avoid huge memory spikes if P was larger, but here P is small
        # P size ~ 3000. P x P is 9e6.
        # We can do it in one go or chunks.
        # Let's do chunks of 1000 rows.
        
        max_sum = 2 * N_limit
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        
        chunk_size = 1000
        for i in range(0, len(P_np), chunk_size):
            chunk = P_np[i:i+chunk_size]
            # Sums of chunk + P
            # We want a <= b.
            # This is tricky with simple broadcasting.
            # Simpler: compute all sums, then filter or adjust?
            # If we compute all sums, we count (a,b) and (b,a).
            # We can just count all ordered pairs and divide by 2, handling a=b.
            # But bincount counts integers.
            # Let's just compute all sums a+b for a in chunk, b in P.
            sums = chunk[:, None] + P_np[None, :]
            # Flatten and bincount
            # This counts ordered pairs.
            flat_sums = sums.ravel()
            counts += np.bincount(flat_sums, minlength=max_sum + 1)
            
            if time.time() > deadline - 10:
                break
        
        # counts[N] is the number of ordered pairs (a,b) such that a+b=N.
        # We want distinct pairs a <= b.
        # If a != b, they contribute 2 to counts[N] (a,b and b,a).
        # If a == b, they contribute 1 to counts[N].
        # So pairs(N) = (counts[N] - is_powerful(N/2)) / 2.
        
        best_N = 0
        best_pairs = 0
        
        # We need to check if N/2 is powerful.
        # powerful is a set.
        # Iterate through N? No, find max of pairs.
        # pairs(N) is roughly counts[N]/2.
        # So max counts[N] is a good proxy.
        # But we need exact pairs.
        
        # Find top candidates from counts
        # Get indices of max counts
        # Since counts is large, getting top K is O(M log K).
        
        # Let's just find the max count value
        max_count_val = np.max(counts)
        
        # Get all N with count close to max
        # Or just iterate? No, M is 2e7.
        
        # Let's get the top 100 sums by count
        # np.argsort is slow for 2e7.
        
        # Alternative:
        # We know the best N is likely to have high count.
        # We can just check the top few.
        
        # Let's use np.argpartition to find top 100 indices
        top_indices = np.argpartition(counts, -100)[-100:]
        
        for N in top_indices:
            if N > N_limit: continue # Actually sums can go up to 2*N_limit
            # But we generated P up to N_limit.
            # So max sum is 2*N_limit.
            # However, the task says N <= 10^18.
            # But we only have P up to 10^7.
            # So we can only certify N up to 2*10^7?
            # No, if a+b=N, and a,b <= 10^7, then N <= 2*10^7.
            # So we are limited to N <= 2*10^7.
            
            # Check if N/2 is powerful
            if N % 2 == 0:
                half = N // 2
                if half in powerful:
                    pairs = (counts[N] - 1) // 2
                else:
                    pairs = counts[N] // 2
            else:
                pairs = counts[N] // 2
            
            if pairs > best_pairs:
                best_pairs = pairs
                best_N = N
        
        # Construct pairs for best_N
        final_pairs = []
        for a in P:
            if a > best_N: break
            b = best_N - a
            if b < a: break # Since a is sorted, if b < a, then for larger a, b will be even smaller
            if b in powerful:
                final_pairs.append([a, b])
        
        return {'N': best_N, 'pairs': final_pairs}
    