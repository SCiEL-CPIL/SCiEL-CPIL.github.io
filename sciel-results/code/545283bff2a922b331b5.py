def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 82  # safety margin
    
    N = 512
    LIMIT = 512
    
    # Build conflict bitsets: adj[i] has bit j set if lcm(i,j) > LIMIT
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            if a * b > LIMIT * g:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    def greedy_mis(order):
        chosen_mask = 0
        for node in order:
            if not (adj[node] & chosen_mask):
                chosen_mask |= (1 << node)
        return chosen_mask
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def mask_to_set(mask):
        s = set()
        for i in range(1, N + 1):
            if mask & (1 << i):
                s.add(i)
        return s
    
    def polish(mask, max_iters=200):
        """Single-node swap polish: try removing one node and adding another."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try each removal
            nodes_in = [i for i in range(1, N + 1) if mask & (1 << i)]
            random.shuffle(nodes_in)
            for rem in nodes_in:
                new_mask = mask & ~(1 << rem)
                # Find best add
                best_add = -1
                for add in range(1, N + 1):
                    if not (mask & (1 << add)) and not (adj[add] & new_mask):
                        best_add = add
                        break
                if best_add != -1:
                    mask = new_mask | (1 << best_add)
                    improved = True
                    break
            if not improved:
                break
        return mask
    
    best_mask = 0
    best_size = 0
    nodes = list(range(1, N + 1))
    
    random.seed(12345)
    
    # Try several strategies for ordering
    # Strategy 1: natural order
    m = greedy_mis(nodes)
    s = count_bits(m)
    if s > best_size:
        best_size = s
        best_mask = m
    
    # Strategy 2: reverse order
    m = greedy_mis(nodes[::-1])
    s = count_bits(m)
    if s > best_size:
        best_size = s
        best_mask = m
    
    # Strategy 3: random orders
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        random.shuffle(nodes)
        m = greedy_mis(nodes)
        s = count_bits(m)
        if s > best_size:
            best_size = s
            best_mask = m
            # Polish immediately
            best_mask = polish(best_mask)
            best_size = count_bits(best_mask)
    
    # Final polish on best
    best_mask = polish(best_mask, max_iters=500)
    best_size = count_bits(best_mask)
    
    # Also try: start from a dense small-number set and expand
    # Numbers 1..k where all pairwise lcm <= 512
    # For k=22: lcm(22,21)=462, lcm(22,19)=418, etc. All <= 512? 
    # lcm(a,b) <= a*b <= 22*21=462 < 512. Yes!
    # Can we go higher? 23: lcm(23,22)=506 <= 512. 23*22=506. Yes.
    # 24: lcm(24,23)=552 > 512. No.
    # So {1..23} works? Check lcm(23, 22)=506, lcm(23,21)=483, lcm(23,20)=460...
    # Actually need to check all pairs. 23 is prime, so lcm(23,x)=23x if 23∤x.
    # Max x in {1..22} not divisible by 23 is 22, so max lcm=23*22=506<=512. Yes!
    # What about 24? 24=2^3*3. lcm(24,23)=552>512. So 24 conflicts with 23.
    # But maybe {1..22} ∪ {24}? lcm(24,22)=132, lcm(24,21)=168, lcm(24,19)=456...
    # 24*19=456<=512. 24*21=504<=512. 24*22=132. Seems ok.
    # But we already have 23 in the set. 23 conflicts with 24.
    # So either {1..23} or {1..22,24,...}
    
    # Let's also try a seeded approach: start with {1..23} and add more
    seed_mask = 0
    for i in range(1, 24):
        seed_mask |= (1 << i)
    # Verify it's valid
    # Then try to add more nodes
    for i in range(24, N + 1):
        if not (adj[i] & seed_mask):
            seed_mask |= (1 << i)
    s = count_bits(seed_mask)
    if s > best_size:
        best_size = s
        best_mask = seed_mask
    best_mask = polish(best_mask, max_iters=300)
    best_size = count_bits(best_mask)
    
    result = mask_to_set(best_mask)
    if not result:
        result = {1}
    return sorted(result)
