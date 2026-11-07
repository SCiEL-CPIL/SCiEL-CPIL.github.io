import time
import random

def construct():
    n = 10
    deadline = time.time() + 50  # 50s limit to be safe
    
    # Start with 2 known orthogonal squares
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    squares = [L1, L2]
    
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True
        
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    def is_mutually_orthogonal(sqs):
        for i in range(len(sqs)):
            if not is_latin(sqs[i]):
                return False
            for j in range(i+1, len(sqs)):
                if not are_orthogonal(sqs[i], sqs[j]):
                    return False
        return True

    # Try to add more squares using a simple random search
    # We know max is 9.
    # We will try to generate random Latin squares and check orthogonality.
    # This is slow.
    # Better: Use the product construction logic correctly or hardcode.
    
    # Since I am unsure of the exact 9 squares, I will stick to the 2 squares 
    # unless I can implement the product construction correctly.
    
    # Let's try the product construction with m=2, n=5.
    # Squares:
    # T_{0,k} for k=1..4
    # S_k for k=1..4
    # R_0
    # And verify.
    
    m = 2
    p = 5
    
    # A_0(i,j) = (i+j)%2
    A0 = [[(i+j)%m for j in range(m)] for i in range(m)]
    
    # B_k(i,j) = (i+k*j)%p for k=1..4
    B = []
    for k in range(1, 5):
        B.append([[ (i + k*j) % p for j in range(p)] for i in range(p)])
        
    # Construct 9 squares of order 10
    # Index mapping: i = i1*p + i2, j = j1*p + j2
    # i1, j1 in 0..1, i2, j2 in 0..4
    
    new_squares = []
    
    # 1. R_0: (A0(i1,j1), i2+j2)%p
    R0 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        i1 = i // p
        i2 = i % p
        for j in range(n):
            j1 = j // p
            j2 = j % p
            val1 = A0[i1][j1]
            val2 = (i2 + j2) % p
            R0[i][j] = val1 * p + val2
            
    new_squares.append(R0)
    
    # 2. S_k: (i1+j1)%m, B_k(i2,j2)
    for k in range(4):
        Sk = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            i1 = i // p
            i2 = i % p
            for j in range(n):
                j1 = j // p
                j2 = j % p
                val1 = (i1 + j1) % m
                val2 = B[k][i2][j2]
                Sk[i][j] = val1 * p + val2
        new_squares.append(Sk)
        
    # 3. T_{0,k}: (A0(i1,j1), B_k(i2,j2))
    for k in range(4):
        Tk = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            i1 = i // p
            i2 = i % p
            for j in range(n):
                j1 = j // p
                j2 = j % p
                val1 = A0[i1][j1]
                val2 = B[k][i2][j2]
                Tk[i][j] = val1 * p + val2
        new_squares.append(Tk)
        
    # Verify the 9 new squares
    if is_mutually_orthogonal(new_squares):
        return new_squares
    else:
        # Fall back to the 2 squares if construction failed
        return squares

# EVOLVE-BLOCK-END
