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
    
    # Strategy: Use the structural decomposition from the current program (CRT over Z2 x Z5)
    # but expand the search space significantly by allowing more general linear forms
    # and checking orthogonality more efficiently.
    # 
    # The key insight from inspirations is that we need to be more exhaustive in our
    # candidate generation while maintaining the mathematical structure.
    
    # First, try the basic family of linear forms
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1, 2, 3, 4]:
                for d in [0, 1, 2, 3, 4]:
                    for k in [1, 2, 3, 4]:
                        if time.time() > deadline:
                            return mols
                        
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
    
    # If we haven't found 9, try a broader family with additional parameters
    # Using the form: s1 = (i1 + j1 + a*i2 + b*j2 + e*i1*j1) % 2
    # s2 = (i2 + k*j2 + c*i1 + d*j1 + f*i2*j2 + g*i1*j2 + h*i2*j1) % 5
    
    if len(mols) < 9:
        for a in [0, 1]:
            for b in [0, 1]:
                for e in [0, 1]:
                    for c in [0, 1, 2, 3, 4]:
                        for d in [0, 1, 2, 3, 4]:
                            for k in [1, 2, 3, 4]:
                                for f in [0, 1, 2, 3, 4]:
                                    for g in [0, 1, 2, 3, 4]:
                                        for h in [0, 1, 2, 3, 4]:
                                            if time.time() > deadline:
                                                return mols
                                            
                                            sq = [[0]*n for _ in range(n)]
                                            for i in range(n):
                                                i1, i2 = decomp(i)
                                                for j in range(n):
                                                    j1, j2 = decomp(j)
                                                    s1 = (i1 + j1 + a*i2 + b*j2 + e*i1*j1) % p
                                                    s2 = (i2 + k*j2 + c*i1 + d*j1 + f*i2*j2 + g*i1*j2 + h*i2*j1) % q
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
    
    # Final fallback: try to add any remaining orthogonal squares from a precomputed set
    # This is a safety net in case the search doesn't find all 9
    
    # Known MOLS of order 10 (from combinatorial designs)
    # We'll try to add these if they're orthogonal to our current set
    known_squares = [
        [[(i + j) % 10 for j in range(10)] for i in range(10)],
    ]
    
    for sq in known_squares:
        if time.time() > deadline:
            break
        key = tuple(tuple(row) for row in sq)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if all(are_orthogonal(sq, existing) for existing in mols):
            mols.append(sq)
            if len(mols) >= 9:
                break
    
    # Ensure we return at least 1 square
    if not mols:
        # Return the basic cyclic Latin square as fallback
        mols = [[[(i + j) % 10 for j in range(10)] for i in range(10)]]
    
    # Convert to pure Python ints and lists
    result = [[[int(v) for v in row] for row in L] for L in mols]
    return result
