import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 10
    
    # Step 1: Generate 5 MOLS of order 5
    # Identity and 4 linear squares over F5
    mols_5 = []
    # Identity
    mols_5.append([[i for i in range(5)] for _ in range(5)]) # Wait, Identity is L[i][j] = i? No, L[i][j]=j is identity for rows?
    # Standard Latin Square: L[i][j] = (i+j)%n
    # Identity for MOLS context usually means the square that is orthogonal to others.
    # L_0(i,j) = i? No, L(i,j) = j is a Latin square.
    # Let's use L_k(i,j) = (k*i + j) % 5 for k=0..4.
    # k=0: L(0,i,j) = j.
    # k=1: L(1,i,j) = i+j.
    # etc.
    for k in range(5):
        sq = [[(k * i + j) % 5 for j in range(5)] for i in range(5)]
        mols_5.append(sq)
        
    # Verify MOLS_5 are mutually orthogonal
    # (Assumed true for prime field)

    # Step 2: Generate 1 MOLS of order 2
    A = [[0, 1], [1, 0]]
    
    # Step 3: Construct 10x10 squares
    squares = []
    
    # Type 1: High part from A, Low part from M_5
    for k in range(5):
        sq = [[0]*10 for _ in range(10)]
        for i in range(10):
            i1, i2 = divmod(i, 5)
            for j in range(10):
                j1, j2 = divmod(j, 5)
                sq[i][j] = 5 * A[i1][j1] + mols_5[k][i2][j2]
        squares.append(sq)
        
    # Type 2: High part from M_5, Low part from A
    for k in range(5):
        sq = [[0]*10 for _ in range(10)]
        for i in range(10):
            i1, i2 = divmod(i, 5)
            for j in range(10):
                j1, j2 = divmod(j, 5)
                sq[i][j] = 5 * mols_5[k][i2][j2] + A[i1][j1]
        squares.append(sq)
        
    # Step 4: Check orthogonality and select max subset
    # We have 10 candidate squares.
    # We need to find the largest subset of mutually orthogonal squares.
    # Since 10 is small, we can try greedy or simple backtracking.
    
    def is_orthogonal(s1, s2):
        pairs = set()
        for i in range(10):
            for j in range(10):
                p = (s1[i][j], s2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
        
    # Greedy selection
    selected = []
    for sq in squares:
        if time.time() > deadline:
            break
        valid = True
        for sel in selected:
            if not is_orthogonal(sq, sel):
                valid = False
                break
        if valid:
            selected.append(sq)
            
    # If we have less than 5, something is wrong, but we expect at least 5.
    # Convert to lists of ints
    result = []
    for sq in selected:
        result.append([[int(sq[i][j]) for j in range(10)] for i in range(10)])
        
    return result

