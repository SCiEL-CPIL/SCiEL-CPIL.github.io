import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 10
    p, q = 2, 5
    
    def crt(a, b):
        return a * q + b
    
    def decomp(i):
        return (i % p, i // p)
    
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
    
    def are_orthogonal(sq1, sq2):
        seen = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = sq1[i][j], sq2[i][j]
                if seen[a][b]:
                    return False
                seen[a][b] = True
        return True
    
    mols = []
    seen_keys = set()
    
    # Try all combinations of linear forms
    # s1 = (i1 + j1 + a*i2 + b*j2) % 2
    # s2 = (i2 + k*j2 + c*i1 + d*j1) % 5
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1, 2, 3, 4]:
                for d in [0, 1, 2, 3, 4]:
                    for k in [1, 2, 3, 4]:
                        if time.time() > deadline:
                            break
                        sq = [[0]*n for _ in range(n)]
                        for i in range(n):
                            i1, i2 = decomp(i)
                            for j in range(n):
                                j1, j2 = decomp(j)
                                s1 = (i1 + j1 + a*i2 + b*j2) % p
                                s2 = (i2 + k*j2 + c*i1 + d*j1) % q
                                sq[i][j] = crt(s1, s2)
                        
                        if not is_latin(sq):
                            continue
                        
                        key = tuple(tuple(row) for row in sq)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        
                        if all(are_orthogonal(sq, existing) for existing in mols):
                            mols.append(sq)
                            if len(mols) >= 9:
                                return mols
    
    return mols
