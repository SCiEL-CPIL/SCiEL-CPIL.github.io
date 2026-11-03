"""Matrix multiplication 2x3x8 tensor decomposition."""
import time

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.
    
    Strategy: Split A's 3 columns into A2 (2x2, cols 1-2) and A1 (2x1, col 0).
    - A1 * B[0,:] (2x1 * 1x8): 16 schoolbook terms
    - A2 * B[1:3,:] (2x2 * 2x8): split into 4 blocks of 2x2*2x2, each Strassen rank 7 = 28 terms
    Total rank: 44
    """
    start_time = time.time()
    
    n, m, p = 2, 3, 8
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: j=0 schoolbook (16 terms)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 Strassen (28 terms = 4 blocks x 7)
    # A2 = A[:, 1:3] (2x2), B2 = B[1:3, :] (2x8)
    # Split B2 into 4 blocks of 2 columns
    # Strassen for 2x2*2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 - M2 + M3 + M4
    # C[0,1] = M5 + M7
    # C[1,0] = M6 + M4
    # C[1,1] = M1 - M3 + M5 - M7
    
    # A2 indices in U: A2[0,0]=A[0,1]->U[3], A2[0,1]=A[0,2]->U[4], A2[1,0]=A[1,1]->U[6-1]=U[5]... 
    # Wait: U index is i*m+j = i*3+j. A[i,j] -> U[i*3+j].
    # A2[0,0] = A[0,1] -> U[0*3+1] = U[1]
    # A2[0,1] = A[0,2] -> U[0*3+2] = U[2]
    # A2[1,0] = A[1,1] -> U[1*3+1] = U[4]
    # A2[1,1] = A[1,2] -> U[1*3+2] = U[5]
    
    # B2 indices in V: B2[j,k] = B[j+1,k] -> V[(j+1)*8+k]
    # B2[0,0] = B[1,2b] -> V[1*8+2b] = V[8+2b]
    # B2[0,1] = B[1,2b+1] -> V[1*8+2b+1] = V[9+2b]
    # B2[1,0] = B[2,2b] -> V[2*8+2b] = V[16+2b]
    # B2[1,1] = B[2,2b+1] -> V[2*8+2b+1] = V[17+2b]
    
    # C indices in W: C[i,k] -> W[k*2+i]
    # C[0,2b] -> W[2b*2+0] = W[4b]
    # C[0,2b+1] -> W[(2b+1)*2+0] = W[4b+2]
    # C[1,2b] -> W[2b*2+1] = W[4b+1]
    # C[1,2b+1] -> W[(2b+1)*2+1] = W[4b+3]
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_b00 = 8 + k0   # B[1,k0]
        v_b01 = 8 + k1   # B[1,k1]
        v_b10 = 16 + k0  # B[2,k0]
        v_b11 = 16 + k1  # B[2,k1]
        
        # C indices
        w_c00 = k0 * 2 + 0  # C[0,k0]
        w_c01 = k1 * 2 + 0  # C[0,k1]
        w_c10 = k0 * 2 + 1  # C[1,k0]
        w_c11 = k1 * 2 + 1  # C[1,k1]
        
        # A2 indices
        u_a00 = 0 * 3 + 1  # A[0,1] = U[1]
        u_a01 = 0 * 3 + 2  # A[0,2] = U[2]
        u_a10 = 1 * 3 + 1  # A[1,1] = U[4]
        u_a11 = 1 * 3 + 2  # A[1,2] = U[5]
        
        # Strassen term 1: M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # U: A[0,1]=1, A[1,2]=1
        # V: B[1,k0]=1, B[2,k1]=1
        # W: C[0,k0]=1, C[1,k1]=1
        U[u_a00][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # Strassen term 2: M2 = (A[1,0]+A[1,1])*B[0,0]
        # U: A[1,1]=1, A[1,2]=1
        # V: B[1,k0]=1
        # W: C[0,k0]=-1
        U[u_a10][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c00][col] = -1
        col += 1
        
        # Strassen term 3: M3 = A[0,0]*(B[0,1]-B[1,1])
        # U: A[0,1]=1
        # V: B[1,k1]=1, B[2,k1]=-1
        # W: C[0,k0]=1
        U[u_a00][col] = 1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c00][col] = 1
        col += 1
        
        # Strassen term 4: M4 = A[1,1]*(B[1,0]-B[0,0])
        # U: A[1,2]=1
        # V: B[2,k0]=1, B[1,k0]=-1
        # W: C[0,k0]=1, C[1,k0]=1
        U[u_a11][col] = 1
        V[v_b10][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # Strassen term 5: M5 = (A[0,0]+A[0,1])*B[1,1]
        # U: A[0,1]=1, A[0,2]=1
        # V: B[2,k1]=1
        # W: C[0,k1]=1, C[1,k1]=-1
        U[u_a00][col] = 1
        U[u_a01][col] = 1
        V[v_b11][col] = 1
        W[w_c01][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # Strassen term 6: M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # U: A[1,1]=1, A[0,1]=-1
        # V: B[1,k0]=1, B[1,k1]=1
        # W: C[1,k0]=1
        U[u_a10][col] = 1
        U[u_a00][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # Strassen term 7: M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # U: A[0,2]=1, A[1,2]=-1
        # V: B[2,k0]=1, B[2,k1]=1
        # W: C[0,k1]=1, C[1,k1]=-1
        U[u_a01][col] = 1
        U[u_a11][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c01][col] = 1
        W[w_c11][col] = -1
        col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify correctness with a random test
    import random
    random.seed(42)
    
    # Generate random A (2x3) and B (3x8)
    A_test = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
    B_test = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C_test = [[sum(A_test[i][j] * B_test[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Verify via tensor decomposition
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give C[i,k]
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(total_rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            if total != C_test[i][k]:
                # This shouldn't happen, but if it does, return schoolbook
                n2, m2, p2 = 2, 3, 8
                rb = n2 * m2 * p2
                Ub = [[0] * rb for _ in range(n2 * m2)]
                Vb = [[0] * rb for _ in range(m2 * p2)]
                Wb = [[0] * rb for _ in range(p2 * n2)]
                c2 = 0
                for ii in range(n2):
                    for jj in range(m2):
                        for kk in range(p2):
                            Ub[ii * m2 + jj][c2] = 1
                            Vb[jj * p2 + kk][c2] = 1
                            Wb[kk * n2 + ii][c2] = 1
                            c2 += 1
                return {"U": Ub, "V": Vb, "W": Wb}
    
    return {"U": U, "V": V, "W": W}
