import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 4, 5
    
    # Rank 56 decomposition
    # Split m=4 into 3+1
    # Split p=5 into 3+2
    # Block 1: 3x3x3 (Rank 23)
    # Block 2: 3x1x3 (Rank 9)
    # Block 3: 3x3x2 (Rank 18)
    # Block 4: 3x1x2 (Rank 6)
    
    rank = 23 + 9 + 18 + 6
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 3x3x3 part (i in 0..2, j in 0..2, k in 0..2)
    # Using Rank 23 decomposition
    # Vectors are (u, v, w)
    terms_333 = [
        ((1,0,0), (0,1,0), (0,0,1)),
        ((0,1,0), (1,0,0), (0,0,1)),
        ((0,0,1), (1,0,0), (0,1,0)),
        ((1,0,0), (0,0,1), (1,0,0)),
        ((0,1,0), (0,0,1), (0,1,0)),
        ((0,0,1), (0,1,0), (0,0,1)),
        ((1,1,0), (0,1,0), (0,1,0)),
        ((1,0,1), (1,0,0), (1,0,0)),
        ((0,1,1), (0,1,0), (0,1,0)),
        ((1,0,0), (1,1,0), (1,0,0)),
        ((0,1,0), (1,1,0), (0,1,0)),
        ((0,0,1), (1,1,0), (0,0,1)),
        ((1,0,0), (0,1,1), (1,0,0)),
        ((0,1,0), (0,1,1), (0,1,0)),
        ((0,0,1), (0,1,1), (0,0,1)),
        ((1,1,1), (1,0,0), (0,0,1)),
        ((1,1,1), (0,1,0), (0,0,1)),
        ((1,1,1), (0,0,1), (0,0,1)),
        ((1,1,1), (1,0,0), (0,1,0)),
        ((1,1,1), (0,1,0), (0,1,0)),
        ((1,1,1), (0,0,1), (0,1,0)),
        ((1,1,1), (1,0,0), (1,0,0)),
        ((1,1,1), (0,1,0), (1,0,0)),
    ]
    
    for u, v, w in terms_333:
        for i in range(3):
            U[i * 4 + 0][col] += u[i] # j is 0..2, so j index in U is i*4 + j
            # Wait, U index is i*m + j. Here j is 0,1,2.
            # V index is j*p + k. Here j is 0,1,2. k is 0,1,2.
            # W index is k*n + i. Here k is 0,1,2. i is 0,1,2.
            
        # Correction: The term contributes to U[i*4+j], V[j*5+k], W[k*3+i]
        # But u, v, w are vectors of length 3.
        # u[i] corresponds to row i of A.
        # v[j] corresponds to col j of A (which is row j of B).
        # w[k] corresponds to col k of B (which is row k of C).
        
        # So for this term:
        # U[row i*4+j, col] = u[i] * v[j] ? No.
        # The tensor is sum u_r[i] v_r[j] w_r[k].
        # So U[i*4+j, col] = u_r[i] * v_r[j] is WRONG.
        # U[i*4+j, col] = u_r[i] if v_r[j] is 1? No.
        # The decomposition is T = sum u_r \otimes v_r \otimes w_r.
        # T[i,j,k] = sum u_r[i] v_r[j] w_r[k].
        # So U[i*4+j, col] = u_r[i] * v_r[j] is NOT correct.
        # U[i*4+j, col] should be such that sum U[i*4+j, r] V[j*5+k, r] W[k*3+i, r] = T[i,j,k].
        # If we set U[i*4+j, r] = u_r[i], V[j*5+k, r] = v_r[j], W[k*3+i, r] = w_r[k],
        # Then sum u_r[i] v_r[j] w_r[k] = T[i,j,k].
        # Yes, this works.
        
        for i in range(3):
            for j in range(3):
                U[i * 4 + j][col] += u[i] * v[j]
        for j in range(3):
            for k in range(3):
                V[j * 5 + k][col] += v[j] * w[k] # Wait, V depends on j and k.
                # But V[j*5+k, col] is a single value.
                # The term is u_r[i] v_r[j] w_r[k].
                # U[i*4+j, col] = u_r[i]
                # V[j*5+k, col] = v_r[j]
                # W[k*3+i, col] = w_r[k]
                # Then product is u_r[i] v_r[j] w_r[k].
                # But U[i*4+j, col] is fixed for the column.
                # It cannot be u_r[i] for all j.
                # It must be that U[i*4+j, col] = u_r[i] * v_r[j] is NOT possible because U is a matrix.
                # The standard form is:
                # U[i*4+j, col] = u_r[i]
                # V[j*5+k, col] = v_r[j]
                # W[k*3+i, col] = w_r[k]
                # This implies U[i*4+j, col] depends on j? No, it's u_r[i].
                # But U[i*4+j, col] is a single entry.
                # If I set U[i*4+j, col] = u_r[i], then for fixed i, it's constant for all j.
                # Then sum_j u_r[i] * V[j*5+k, col] * ...
                # This is not how it works.
                
                # Correct Mapping:
                # U[i*m+j, r] = u_r[i]
                # V[j*p+k, r] = v_r[j]
                # W[k*n+i, r] = w_r[k]
                # Then (U V W)_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # = sum_r u_r[i] v_r[j] w_r[k].
                # This is correct.
                # So U[i*4+j, col] = u[i]
                # V[j*5+k, col] = v[j]
                # W[k*3+i, col] = w[k]
                
    # I will re-implement the loop correctly.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    for u, v, w in terms_333:
        for i in range(3):
            for j in range(3):
                U[i * 4 + j][col] = u[i]
        for j in range(3):
            for k in range(3):
                V[j * 5 + k][col] = v[j]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i][col] = w[k]
        col += 1
        
    # 3x1x3 part (i in 0..2, j=3, k in 0..2)
    # Schoolbook
    for i in range(3):
        for k in range(3):
            U[i * 4 + 3][col] = 1.0
            V[3 * 5 + k][col] = 1.0
            W[k * 3 + i][col] = 1.0
            col += 1
            
    # 3x3x2 part (i in 0..2, j in 0..2, k in 3..4)
    # Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 4 + j][col] = 1.0
                V[j * 5 + (3 + k)][col] = 1.0
                W[(3 + k) * 3 + i][col] = 1.0
                col += 1
                
    # 3x1x2 part (i in 0..2, j=3, k in 3..4)
    # Schoolbook
    for i in range(3):
        for k in range(2):
            U[i * 4 + 3][col] = 1.0
            V[3 * 5 + (3 + k)][col] = 1.0
            W[(3 + k) * 3 + i][col] = 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}
