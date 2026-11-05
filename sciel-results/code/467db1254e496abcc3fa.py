    def construct():
        import time
        start_time = time.time()
        deadline = 50.0 # 10s margin
        
        N = 256
        nodes = list(range(2, N + 1))
        
        # Precompute conflicts
        # conflicts[i] is a set of indices j such that lcm(nodes[i], nodes[j]) <= 256
        # Using indices 0..254 corresponding to 2..256
        num_nodes = len(nodes)
        conflicts = [set() for _ in range(num_nodes)]
        
        for i in range(num_nodes):
            a = nodes[i]
            for j in range(i + 1, num_nodes):
                b = nodes[j]
                if lcm(a, b) <= 256:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        def get_score(S):
            return sum(1.0 / nodes[i] for i in S)
            
        def local_search(initial_S):
            S = set(initial_S)
            current_score = get_score(S)
            
            while time.time() - start_time < deadline:
                improved = False
                # Try adding each v not in S
                # To speed up, we can check candidates in order of high weight (small node value)
                # But we need to check all to find best improvement
                best_gain = 0
                best_v = -1
                best_evict = set()
                
                for v in range(num_nodes):
                    if v in S:
                        continue
                    # Find conflicts in S
                    evict = conflicts[v] & S
                    if not evict:
                        gain = 1.0 / nodes[v]
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                            best_evict = set()
                    else:
                        evict_cost = sum(1.0 / nodes[u] for u in evict)
                        gain = 1.0 / nodes[v] - evict_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                            best_evict = evict
                            
                if best_gain > 1e-9:
                    S.add(best_v)
                    S -= best_evict
                    current_score += best_gain
                    improved = True
                else:
                    break
                    
            return S, current_score

        # Strategy 1: Seed Greedy
        seed_S = []
        for x in nodes:
            idx = nodes.index(x) # This is slow, better to map value to idx
            # Actually, let's just use values for clarity and map to idx for conflict check
            # Or just use values in sets and precompute conflict values.
            pass
        
        # Let's stick to indices for efficiency.
        
        # Initial 1: Greedy Ascending (Seed)
        S1 = set()
        for i in range(num_nodes):
            if not (conflicts[i] & S1):
                S1.add(i)
        res1, score1 = local_search(S1)
        
        # Initial 2: Greedy Descending
        S2 = set()
        for i in range(num_nodes - 1, -1, -1):
            if not (conflicts[i] & S2):
                S2.add(i)
        res2, score2 = local_search(S2)
        
        # Initial 3: Greedy by Degree (Low degree first)
        degrees = [len(conflicts[i]) for i in range(num_nodes)]
        order = sorted(range(num_nodes), key=lambda i: degrees[i])
        S3 = set()
        for i in order:
            if not (conflicts[i] & S3):
                S3.add(i)
        res3, score3 = local_search(S3)
        
        # Pick best
        best_S, best_score = max([(res1, score1), (res2, score2), (res3, score3)], key=lambda x: x[1])
        
        # Convert back to list of values
        result = [nodes[i] for i in best_S]
        
        return result
    