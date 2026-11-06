def construct():
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    def is_squarefree(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True
    
    def gen_powerful(B):
        # Generate all powerful numbers up to B using n = a^2 * b^3 with b squarefree
        powerful = set()
        max_b = int(B ** (1/3)) + 1
        for b in range(1, max_b + 1):
            if not is_squarefree(b):
                continue
            b3 = b ** 3
            max_a = int((B // b3) ** 0.5) + 1
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= B:
                    powerful.add(n)
        return sorted(powerful)
    
    B = 100000000  # 10^8
    P = gen_powerful(B)
    P_set = set(P)
    
    # Count sums
    from collections import Counter
    sum_count = Counter()
    n = len(P)
    for i in range(n):
        p = P[i]
        if time.time() > deadline - 10:
            break
        for j in range(i, n):
            q = P[j]
            s = p + q
            sum_count[s] += 1
            if time.time() > deadline - 10:
                break
    
    if not sum_count:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N, best_count = max(sum_count.items(), key=lambda x: x[1])
    
    # Reconstruct pairs
    pairs = []
    for p in P:
        q = best_N - p
        if p <= q and q in P_set:
            pairs.append([p, q])
    
    return {'N': best_N, 'pairs': pairs}
