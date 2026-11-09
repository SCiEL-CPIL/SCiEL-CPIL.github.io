    # EVOLVE-BLOCK-START
    def construct():
        from math import isqrt
        from collections import Counter

        # Limit for powerful numbers generation
        limit = 150_000_000
        
        # Generate powerful numbers
        pw = set()
        max_b = int(limit ** (1/3)) + 2
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = isqrt(limit // b3)
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    pw.add(val)
        
        pw_list = sorted(pw)
        n_pw = len(pw_list)
        
        sum_counts = Counter()
        
        # Local vars for speed
        pw_l = pw_list
        n = n_pw
        sc = sum_counts
        
        # Count pairs a <= b
        for i in range(n):
            a = pw_l[i]
            for j in range(i, n):
                b = pw_l[j]
                sc[a + b] += 1
                
        # Find best N
        best_N, best_count = max(sc.items(), key=lambda x: x[1])
        
        # Reconstruct pairs
        pw_set = set(pw_list)
        pairs = []
        for a in pw_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in pw_set and a <= b:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
    # EVOLVE-BLOCK-END
    