def priority(vector, n):
        # Count 0s, 1s, 2s
        # Use sum(1 for x in vector if x==0) etc.
        # Or use collections.Counter? Counter might be slower for small n?
        # n is likely small (cap set problem is hard, n usually < 10 for greedy to finish in time?).
        # 3^10 = 59049. 3^11 = 177147.
        # Greedy check is O(|S|^2). If |S| is large, this is slow.
        # The bottleneck is the evaluator's greedy step, not the sorting, unless sorting is huge.
        # But we must be fast in priority() too.
        
        n0 = 0
        n1 = 0
        n2 = 0
        for v in vector:
            if v == 0: n0 += 1
            elif v == 1: n1 += 1
            else: n2 += 1
            
        # Target balance
        target = n / 3.0
        # Minimize squared deviation
        score = -((n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2)
        
        # Add a small deterministic tie-breaker based on vector content to avoid
        # large blocks of equal priority which might lead to bad lexicographic ordering.
        # Sum of coordinates?
        tie_breaker = sum(vector) * 0.001
        
        return score + tie_breaker
