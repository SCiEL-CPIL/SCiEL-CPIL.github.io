import time

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 55s to be safe
    
    S = []
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
            
        valid = True
        
        # If x is 1 and we have at least 3 elements, 1 divides any sum of 3.
        # So x=1 is invalid if |S| >= 3.
        if x == 1 and len(S) >= 3:
            continue
            
        # 1. Check existing elements a in S
        # For each a, check if any b, c in S \ {a} (b != c)
        # such that a divides (x + b + c)
        
        # Optimization: If x is very small, x+b+c is close to b+c.
        # We just iterate.
        
        n = len(S)
        for idx_a, a in enumerate(S):
            # Check pairs b, c from S
            # b, c must be distinct and not a
            # We iterate all pairs (i, j) with i < j
            # If S[i] == a or S[j] == a, skip
            for i in range(n):
                b = S[i]
                if b == a:
                    continue
                for j in range(i + 1, n):
                    c = S[j]
                    if c == a:
                        continue
                    # b and c are distinct
                    # b != a and c != a
                    if (x + b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # 2. Check new element x
        # Need to ensure no b, c, d in S (distinct) such that x divides (b+c+d)
        
        if len(S) >= 3:
            # Max sum of 3 in S
            # S is [192, 191, ...] (descending)
            max_sum = S[0] + S[1] + S[2]
            S_set = set(S)
            
            # Iterate multiples of x
            # Start from x, but sum must be at least min_sum = S[-1] + S[-2] + S[-3]
            min_sum = S[-1] + S[-2] + S[-3]
            
            # First multiple >= min_sum
            start_mult = (min_sum + x - 1) // x
            if start_mult < 1:
                start_mult = 1
            start_M = start_mult * x
            
            for M in range(start_M, max_sum + 1, x):
                # Check if any 3 distinct in S sum to M
                found = False
                # Iterate pairs
                for i in range(len(S)):
                    b = S[i]
                    if b >= M:
                        # Since S is descending, b is large.
                        # If b >= M, then b+c+d > M (since c,d >= 1).
                        # So no solution with this b.
                        # And for larger i, b is smaller? No, S is descending.
                        # S[0] is max. S[i] decreases.
                        # So if b < M, it might work.
                        # If b >= M, it's too big.
                        # So we can break? No, b decreases.
                        # So if b is too big, we continue to smaller b?
                        # Wait. S = [192, 191, ...].
                        # b = 192. If M=100, b > M.
                        # Next b=191. Still > M.
                        # Eventually b < M.
                        # So we can't break immediately. We must check all.
                        pass
                    
                    for j in range(i + 1, len(S)):
                        c = S[j]
                        rem = M - b - c
                        # rem must be in S, distinct from b, c
                        # Also rem > 0
                        if rem <= 0:
                            # c is decreasing, so rem increases.
                            # If rem <= 0, it might become positive later?
                            # No, c decreases, so -c increases, so rem increases.
                            # So if rem <= 0, it will get bigger.
                            # But we need rem in S.
                            # Min in S is S[-1].
                            # If rem < S[-1], it's not in S.
                            # But rem increases. So it might enter S range.
                            continue
                            
                        if rem in S_set and rem != b and rem != c:
                            # Check distinctness of b, c, rem
                            # rem != b and rem != c handled.
                            # b != c handled by i != j.
                            found = True
                            break
                    if found:
                        break
                if found:
                    valid = False
                    break
                    
        if valid:
            S.append(x)
            
    return S
