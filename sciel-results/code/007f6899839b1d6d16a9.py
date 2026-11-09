import math

def construct():
    N = 180
    S = set(range(91, N + 1))
    
    for a in range(90, 0, -1):
        a2 = a * a
        conflict = False
        
        # d must be such that a+d <= N => d <= N-a
        # and a+a2/d <= N => a2/d <= N-a => d >= a2/(N-a)
        
        min_d = max(1, int(math.ceil(a2 / (N - a))) if N > a else 1)
        max_d = N - a
        
        for d in range(min_d, max_d + 1):
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                
                b1 = a + d1
                c1 = a + d2
                
                if b1 == c1:
                    continue
                
                # We know b1 <= N and c1 <= N from min_d/max_d constraints roughly
                # But d2 might be small, so c1 might be small?
                # c1 = a + a2/d. Since d >= min_d, a2/d <= N-a, so c1 <= N.
                # b1 = a + d. Since d <= max_d, b1 <= N.
                
                if b1 in S and c1 in S:
                    conflict = True
                    break
                
        if not conflict:
            S.add(a)
            
    return list(S)
