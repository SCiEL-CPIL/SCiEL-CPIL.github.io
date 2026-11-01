import time

DIMENSIONS = [2, 2, 5]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = 2, 2, 5
    rank = 18
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # block_k_offset: the global column index 'k' offset for this 2x2 block
    def add_strassen_block(k_offset):
        nonlocal col
        # Indices for A: i, j -> i*m+j
        # Indices for B: j, k -> j*p + (k_offset + k_local)
        # Indices for C: k, i -> (k_offset + k_local)*n + i
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: A[0,0] + A[1,1] => U[0, col]+=1, U[3, col]+=1
        U[0*m+0][col] += 1
        U[1*m+1][col] += 1
        # V: B[0,k0] + B[1,k1] => V[0*p + k_offset + 0, col]+=1, V[1*p + k_offset + 1, col]+=1
        V[0*p + k_offset + 0][col] += 1
        V[1*p + k_offset + 1][col] += 1
        # W: C[0,0] += M1, C[1,1] += M1
        W[(k_offset+0)*n + 0][col] += 1
        W[(k_offset+1)*n + 1][col] += 1
        
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[1*m+0][col] += 1
        U[1*m+1][col] += 1
        V[0*p + k_offset + 0][col] += 1
        # W: C[1,0] += M2
        W[(k_offset+0)*n + 1][col] += 1
        
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k0])
        U[0*m+0][col] += 1
        V[0*p + k_offset + 1][col] += 1
        V[1*p + k_offset + 0][col] -= 1
        # W: C[0,1] += M3
        W[(k_offset+1)*n + 0][col] += 1
        
        col += 1
        
        # M4 = A[1,1] * (B[1,k1]-B[0,k0])
        U[1*m+1][col] += 1
        V[1*p + k_offset + 1][col] += 1
        V[0*p + k_offset + 0][col] -= 1
        # W: C[0,0] += M4, C[1,0] += M4
        W[(k_offset+0)*n + 0][col] += 1
        W[(k_offset+0)*n + 1][col] += 1
        
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0*m+0][col] += 1
        U[0*m+1][col] += 1
        V[1*p + k_offset + 1][col] += 1
        # W: C[0,0] -= M5, C[0,1] += M5
        W[(k_offset+0)*n + 0][col] -= 1
        W[(k_offset+1)*n + 0][col] += 1
        
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[1*m+0][col] += 1
        U[0*m+0][col] -= 1
        V[0*p + k_offset + 0][col] += 1
        V[0*p + k_offset + 1][col] += 1
        # W: C[1,1] += M6
        W[(k_offset+1)*n + 1][col] += 1
        
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[0*m+1][col] += 1
        U[1*m+1][col] -= 1
        V[1*p + k_offset + 0][col] += 1
        V[1*p + k_offset + 1][col] += 1
        # W: C[0,0] += M7
        W[(k_offset+0)*n + 0][col] += 1
        
        col += 1
        
    # Add Strassen for columns 0,1
    add_strassen_block(0)
    
    # Add Strassen for columns 2,3
    add_strassen_block(2)
    
    # Schoolbook for column 4
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # 4 terms:
    # 1. A[0,0] * B[0,4]
    # 2. A[0,1] * B[1,4]
    # 3. A[1,0] * B[0,4]
    # 4. A[1,1] * B[1,4]
    
    # Term 1
    U[0*m+0][col] += 1
    V[0*p + 4][col] += 1
    W[4*n + 0][col] += 1
    col += 1
    
    # Term 2
    U[0*m+1][col] += 1
    V[1*p + 4][col] += 1
    W[4*n + 0][col] += 1
    col += 1
    
    # Term 3
    U[1*m+0][col] += 1
    V[0*p + 4][col] += 1
    W[4*n + 1][col] += 1
    col += 1
    
    # Term 4
    U[1*m+1][col] += 1
    V[1*p + 4][col] += 1
    W[4*n + 1][col] += 1
    col += 1
    
    if time.time() > deadline:
         # Return best effort (though this construction is fast and deterministic)
         pass
         
    return {"U": U, "V": V, "W": W}
