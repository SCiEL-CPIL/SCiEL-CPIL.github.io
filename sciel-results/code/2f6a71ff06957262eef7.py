def construct():
    start_time = time.time()
    time_limit = 85.0
    
    best_set = []
    best_size = 0
    
    def can_add(S, sums_set, x):
        n = len(S)
        new_sums = set()
        expected = 1 + n + n * (n + 1) // 2
        
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                new_sums.add(x + si + S[j])
                
        if len(new_sums) != expected:
            return False, None
        if new_sums & sums_set:
            return False, None
        return True, new_sums

    while time.time() - start_time < time_limit:
        S = []
        sums_set = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            ok, new_sums = can_add(S, sums_set, x)
            if ok:
                S.append(x)
                sums_set.update(new_sums)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
