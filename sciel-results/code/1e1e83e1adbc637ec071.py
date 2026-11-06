import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55  # safety margin

    def elapsed():
        return time.time() - start_time

    # Initial valid set: upper tail [K, N]
    # Condition for automatic validity: for all a in S, 3*a > 2*N
    # (since max b+c = 2N when b=c=N, but actually max b+c for b<=c in [K,N] is 2N)
    # Actually we need: for a<b<=c in S, a does not divide b+c.
    # If S = [K, N], then for a in [K,N], b,c in [K,N] with b>a, c>=b:
    # b+c >= 2*(K+1) if b>=K+1, or b=K+1, c=K+1 min... 
    # Actually simpler: if 3*K > 2*N, then for any a >= K, b+c <= 2*N < 3*K <= 3*a,
    # so b+c < 3*a, meaning b+c can be at most 2*a (since b+c > a as b>a>=K and c>=b>a).
    # Wait, b+c > a since b > a and c >= b > a, so b+c >= 2*(a+1) > 2a.
    # And b+c <= 2*N. If 3*a > 2*N, then 2*N < 3*a, so b+c < 3*a.
    # Since b+c > 2*a (as b > a, c >= b > a implies b+c >= (a+1)+(a+1) = 2a+2 > 2a),
    # the only multiple of a in (2a, 3a) is... none! Because 2a < b+c < 3a, and the next
    # multiple after 2a is 3a which is not reached. So a does not divide b+c.
    # Actually wait: b+c could equal 2a? No, b > a means b >= a+1, c >= b >= a+1, so b+c >= 2a+2 > 2a.
    # So if b+c < 3a, then a doesn't divide b+c. 
    # So condition 3*min(S) > 2*max(S) suffices.
    
    K = (2 * N) // 3 + 1
    # K = 171 for N=256: 2*256//3 = 170, so K=171.
    # 3*171 = 513 > 512 = 2*256. Good.
    
    S = set(range(K, N + 1))
    
    # Now try to add elements from [1, K-1] greedily
    # For each candidate x, check if adding x to S maintains validity.
    # When adding x (x < all elements in S), we need to check:
    # 1. For a=x, b,c in S with b<=c: x must not divide b+c
    # 2. For a in S, b=x, c in S with x<c: a must not divide x+c
    #    But wait, a<b<=c means a<b. If a is in S and b=x, we need a<x. But x<K<=a for all a in S.
    #    So this case never occurs (a<x is false for all a in S since x<K<=a).
    # 3. For a=x, b in S, c=x: but b<=c means b<=x, but b in S means b>=K>x. Impossible.
    # 4. For a=x, b=x, c in S: b<=c means x<=c, true. But a<b means x<x which is false.
    #    Wait, a<b<=c. If a=x and b=x, then a<b is false. So this doesn't apply.
    # Actually let me re-read: "for every a<b<=c in the set, a must NOT divide b+c"
    # So we need a < b <= c, all in S.
    # If we add x to S, new triples involving x:
    # - a=x, b,c in S (original) with b<=c: need x does not divide b+c
    # - a in S (original), b=x, c in S (original) with x<c: but a<b means a<x. Since x<K and a>=K, impossible.
    # - a in S (original), b in S (original), c=x: b<=c means b<=x, but b>=K>x. Impossible.
    # - a=x, b=x, c in S: a<b means x<x, false.
    # - a=x, b in S, c=x: b<=c means b<=x, but b>=K>x. Impossible.
    # - a in S, b=x, c=x: a<b means a<x, impossible.
    # - a=x, b=x, c=x: a<b false.
    # So the ONLY new constraint is: for all b,c in S_original with b<=c, x does not divide b+c.
    
    # Precompute all sums b+c for b,c in S with b<=c
    S_list = sorted(S)
    sums = set()
    for i, b in enumerate(S_list):
        for c in S_list[i:]:
            sums.add(b + c)
    
    # Now greedily try to add candidates from [1, K-1]
    # Sort candidates by some heuristic - maybe larger x first (less likely to divide things)?
    # Or smaller x first? Let's try larger x first since they have fewer multiples in the sum range.
    candidates = list(range(K - 1, 0, -1))
    
    for x in candidates:
        if elapsed() > deadline:
            break
        # Check if x divides any sum in sums
        valid = True
        for s in sums:
            if s % x == 0:
                valid = False
                break
        if valid:
            S.add(x)
            # Update sums: new sums involving x
            # New sums: x + b for b in S_list (since x < b for all b in S_list)
            # Also x + x = 2x
            for b in S_list:
                sums.add(x + b)
            sums.add(2 * x)
    
    return sorted(S)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
