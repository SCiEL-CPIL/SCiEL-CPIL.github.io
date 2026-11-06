"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # For each candidate a, determine which larger values x > a are "forbidden"
    # to coexist with a. Specifically, if a and x are both in the set, we need to ensure
    # that for any b, c in the set with a < b <= c, a does not divide b+c.
    
    # A simpler characterization: 
    # If S is a subset of [1, n], the condition "for all a < b <= c in S, a ∤ (b+c)" 
    # is equivalent to: for every a in S, the set S \ {a} contains no pair (b, c) with 
    # a < b <= c such that a | (b+c).
    
    # Key insight from the seed: if S = [k, n], then for a in S, b,c in S with a<b<=c,
    # b+c >= 2(a+1) = 2a+2. The smallest multiple of a greater than 2a is 3a.
    # If 3a > 2n, then a cannot divide any b+c where b,c > a.
    # So if 3a > 2n for all a in S, the condition is automatically satisfied.
    # This requires 3k > 2n, i.e., k > 2n/3. For n=256, 2n/3 ≈ 170.67, so k >= 171.
    # This gives |S| = n - k + 1 = 256 - 171 + 1 = 86.
    
    # But we can do better by including some smaller numbers if we exclude conflicting ones.
    # Let's use a greedy approach: try to include numbers from small to large,
    # maintaining a conflict graph.
    
    # Actually, let's think about this differently. The constraint is:
    # For every triple (a, b, c) in S with a < b <= c, a ∤ (b+c).
    # 
    # This is a hereditary property? No, it's not simply hereditary in the usual sense.
    # But we can model it as: S is valid iff for every a in S, there is no pair 
    # (b, c) in S with a < b <= c and a | (b+c).
    
    # Let's use a maximum independent set approach on a conflict graph.
    # Define a graph G on vertices 1..n. Connect two vertices u, v (u < v) if they 
    # "conflict" in the sense that having both in S would create a violation.
    
    # But the constraint is ternary, not binary. So we need a different approach.
    
    # Alternative: Build S incrementally. Start with S = [171, 256] (the tail).
    # Then try to add smaller elements one by one, removing conflicting elements.
    
    # Let's try a greedy from the bottom up, but with careful conflict checking.
    
    # Actually, let's use the following insight: 
    # If we include a, then for any b, c > a in S, we must have a ∤ (b+c).
    # So when we add a to an existing set T (all elements > a), we must remove from T
    # any element that participates in a conflicting pair.
    
    # But this is complex. Let's try a different approach: 
    # Use the fact that the optimal set is likely to be a "tail" plus some carefully 
    # chosen smaller elements.
    
    # Let's try: S = [k, n] union some set of smaller elements, where we choose k 
    # and the smaller elements to maximize |S|.
    
    # For now, let's use a simple improvement: try all possible starting points k 
    # for the tail, and for each, greedily add smaller elements if they don't conflict.
    
    best_set = set()
    best_size = 0
    
    for k in range(1, n+1):
        # Start with the tail [k, n]
        S = set(range(k, n+1))
        
        # Now try to add elements from k-1 down to 1
        for a in range(k-1, 0, -1):
            # Check if we can add a to S
            can_add = True
            # For each pair (b, c) in S with b < c, check if a | (b+c)
            # But we only need to check pairs where both b and c are in S
            # and a < b <= c
            for b in sorted(S):
                if b <= a:
                    continue
                for c in sorted(S):
                    if c < b:
                        continue
                    if a > 0 and (b + c) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.add(a)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # The greedy might not be optimal, but let's see what we get.
    # The tail [171, 256] gives 86. Can we do better?
    
    # Let's also try: what if we use [172, 256] and add some smaller elements?
    # [172, 256] has size 85. If we can add 2+ smaller elements, we beat 86.
    
    # Actually, let's just return the best we found.
    return sorted(best_set)
# EVOLVE-BLOCK-END
