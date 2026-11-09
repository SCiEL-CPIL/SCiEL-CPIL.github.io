        for a in S:
            if a >= x: continue
            b = x - a
            if b <= a or b not in S: continue
            # a, b in S, a+b=x
            for c in S:
                if c == a or c == b: continue
                s1 = a + c
                s2 = b + c
                if s1 > N or s2 > N:
                    continue
                if (s1 in S or s1 == x) and (s2 in S or s2 == x):
                    return False
            # Check c=x
            if (a + x <= N and a + x in S) and (b + x <= N and b + x in S):
                 return False
