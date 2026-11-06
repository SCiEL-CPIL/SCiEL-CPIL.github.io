"""Program using modular arithmetic sieve + greedy construction with fast mean-check."""

def construct():
    n = 512
    max_size = 48
    
    # Step 1: Modular arithmetic sieve to score candidates
    # We prioritize x with fewer exact 3-APs and lower modular vulnerability.
    candidates = list(range(1, n + 1))
    primes = [3, 5, 7]
    
    scores = {x: 0 for x in candidates}
    
    # Compute exact AP counts (k=2 violations)
    for x in candidates:
        count = 0
        low = max(1, 2*x - n)
        high = min(n, 2*x - 1)
        for y in range(low, high + 1):
            z = 2*x - y
            if z != y and z != x and 1 <= z <= n:
                count += 1
        scores[x] = count
        
    # Add modular penalty: count pairs (y,z) in [1, n] such that y+z ≡ 2x (mod p)
    # This is a heuristic for "density" of violations.
    for p in primes:
        for x in candidates:
            target_mod = (2 * x) % p
            count_mod = 0
            # Count y in [1, n] such that z = 2x - y exists in [1, n] and z != y, z != x
            # This is expensive to do exactly for mod p without iterating y.
            # But we can just iterate y. n=512 is small.
            for y in range(1, n + 1):
                if y == x:
                    continue
                z_val = 2 * x - y
                if 1 <= z_val <= n and z_val != y and z_val != x:
                    if (y + z_val) % p == target_mod:
                        count_mod += 1
            scores[x] += count_mod / 10.0 # Weight modular score lower than exact
            
    # Sort candidates by score (ascending).
    candidates.sort(key=lambda x: scores[x])
    
    def is_valid_mean_check(selected, candidate):
        """Check if adding candidate to selected keeps the mean-free property."""
        # 1. Check if candidate is the mean of any subset of selected.
        m = len(selected)
        if m >= 2:
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in selected:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(2, m + 1):
                target = k * candidate
                if target <= sum(selected) and (dp[k] >> target) & 1:
                    return False
        
        # 2. Check if any existing element a becomes the mean of a subset involving candidate.
        for a in selected:
            others = [x for x in selected if x != a]
            if not others:
                continue
                
            delta = a - candidate
            
            # Check if delta is a subset sum of {x - a for x in others}
            reachable = {0}
            for x in others:
                val = x - a
                new_reachable = set()
                for s in reachable:
                    new_reachable.add(s)
                    new_reachable.add(s + val)
                reachable = new_reachable
                
            if delta in reachable:
                # Since a != candidate, delta != 0.
                # Any sum to delta != 0 must come from a non-empty subset.
                return False
        
        return True

    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_mean_check(selected, x):
            selected.append(x)
    
    return selected
