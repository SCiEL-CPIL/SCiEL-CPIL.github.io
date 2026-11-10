import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    n = 10
    p, q = 2, 5  # 10 = 2 * 5
    
    # CRT mapping: (a, b) -> a * 5 + b, where a in Z_2, b in Z_5
    def crt(a, b):
        return a * q + b
    
    # Helper: check if a square is Latin
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
    
    # Helper: check if two squares are orthogonal
    def are_orthogonal(sq1, sq2):
        seen = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = sq1[i][j], sq2[i][j]
                if seen[a][b]:
                    return False
                seen[a][b] = True
        return True
    
    # Construct MOLS using CRT-based construction
    # Represent index i as (i_1, i_2) where i_1 = i % 2, i_2 = i // 2
    # Wait, let's use: i_1 = i % p, i_2 = i // p
    # So for i in 0..9: i_1 in {0,1}, i_2 in {0,1,2,3,4}
    
    def decomp(i):
        return (i % p, i // p)
    
    # For each k in Z_q^* = {1, 2, 3, 4}, define L_k
    mols = []
    
    for k in [1, 2, 3, 4]:
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            i1, i2 = decomp(i)
            for j in range(n):
                j1, j2 = decomp(j)
                # Symbol: (i1 + j1 mod p, i2 + k*j2 mod q)
                s1 = (i1 + j1) % p
                s2 = (i2 + k * j2) % q
                sq[i][j] = crt(s1, s2)
        if is_latin(sq):
            mols.append(sq)
    
    # Verify they are mutually orthogonal
    valid_mols = [mols[0]]
    for sq in mols[1:]:
        if all(are_orthogonal(sq, existing) for existing in valid_mols):
            valid_mols.append(sq)
    mols = valid_mols
    
    # Try to add more MOLS using random search
    random.seed(42)
    
    # Generate candidate Latin squares
    # Use linear squares: L[i][j] = (a*i + b*j + c) % n where gcd(a,n)=1, gcd(b,n)=1
    # For n=10, a,b in {1,3,7,9}
    
    seen_keys = set()
    for sq in mols:
        seen_keys.add(tuple(tuple(row) for row in sq))
    
    # Try linear candidates
    for a in [1, 3, 7, 9]:
        for b in [1, 3, 7, 9]:
            for c in range(n):
                if time.time() > deadline:
                    break
                sq = [[(a*i + b*j + c) % n for j in range(n)] for i in range(n)]
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
    
    # If we haven't found enough, try random search
    if len(mols) < 9:
        for _ in range(50000):
            if time.time() > deadline:
                break
            
            # Generate a random candidate
            # Use random row permutations of a base square
            base = [[(i + j) % n for j in range(n)] for i in range(n)]
            
            # Apply random permutation to symbols
            perm = list(range(n))
            random.shuffle(perm)
            sq = [[perm[base[i][j]] for j in range(n)] for i in range(n)]
            
            if not is_latin(sq):
                continue
            
            key = tuple(tuple(row) for row in sq)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            if all(are_orthogonal(sq, existing) for existing in mols):
                mols.append(sq)
                if len(mols) >= 9:
                    break
    
    return mols
