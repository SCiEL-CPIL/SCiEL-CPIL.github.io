"""Matrix multiplication 2,4,8 - Block-Strassen decomposition with verification."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    U: shape (4, 7), indexed by i*2+j
    V: shape (4, 7), indexed by j*2+k  
    W: shape (4, 7), indexed by k*2+i
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # W[k*2+i]: k=0,i=0 -> 0; k=1,i=1 -> 3
    
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1  # k=0,i=1 -> 0*2+1=1
    
    # M3: A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=-1  # k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # M4: A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,0]=-1; W[1,3]=1  # k=0,i=0 -> 0; k=0,i=1 -> 1
    
    # M5: (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[2,4]=1  # k=1,i=0 -> 2
    
    # M6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1  # k=0,i=0 -> 0
    
    # M7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1  # k=1,i=1 -> 3
    
    return U, V, W

def _construct_block_strassen():
    """Construct the rank-56 Block-Strassen decomposition for 2x4 * 4x8."""
    n, m, p = DIMENSIONS
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Split A (2x4) into 2 blocks of (2x2) along columns: s=0,1
    # Split B (4x8) into 2x4 blocks of (2x2) along rows and columns: s=0,1; j=0..3
    for s in range(2):
        for j in range(4):
            # A_s indices in global A (2x4)
            # A_s is at rows 0-1, cols 2s, 2s+1
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j} indices in global B (4x8)
            # B_{s,j} is at rows 2s, 2s+1 and cols 2j, 2j+1
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C contribution indices in global C (2x8)
            # Local C (2x2) maps to global C rows 0,1 and cols 2j, 2j+1
            # Global C index: k*n + i
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen U (A-side) coefficients
            # m1 = (a00+a11)*(b00+b11)
            U[a00, col] += 1
            U[a11, col] += 1
            # m2 = (a10+a11)*b00
            U[a10, col+1] += 1
            U[a11, col+1] += 1
            # m3 = a00*(b01-b11)
            U[a00, col+2] += 1
            # m4 = a11*b10
            U[a11, col+3] += 1
            # m5 = (a00+a01)*b11
            U[a00, col+4] += 1
            U[a01, col+4] += 1
            # m6 = (a10-a00)*(b00+b01)
            U[a10, col+5] += 1
            U[a00, col+5] += -1
            # m7 = (a01-a11)*(b10+b11)
            U[a01, col+6] += 1
            U[a11, col+6] += -1
            
            # Strassen V (B-side) coefficients
            V[b00, col] += 1
            V[b11, col] += 1
            V[b00, col+1] += 1
            V[b01, col+2] += 1
            V[b11, col+2] += -1
            V[b10, col+3] += 1
            V[b00, col+3] += -1
            V[b11, col+4] += 1
            V[b00, col+5] += 1
            V[b01, col+5] += 1
            V[b10, col+6] += 1
            V[b11, col+6] += 1
            
            # Strassen W (C-side) coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c00, col] += 1
            W[c00, col+3] += 1
            W[c00, col+4] += -1
            W[c00, col+6] += 1
            
            # c01 = m3 + m5
            W[c01, col+2] += 1
            W[c01, col+4] += 1
            
            # c10 = m2 + m4
            W[c10, col+1] += 1
            W[c10, col+3] += 1
            
            # c11 = m1 - m2 + m3 + m6
            W[c11, col] += 1
            W[c11, col+1] += -1
            W[c11, col+2] += 1
            W[c11, col+5] += 1
            
            col += 7
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
    """Verify the decomposition using random integer matrices."""
    rank = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randint(-5, 5, size=(n, m)).astype(float)
        B = np.random.randint(-5, 5, size=(m, p)).astype(float)
        
        # Compute C = A @ B
        C_true = A @ B
        
        # Compute C via tensor decomposition
        # C[k, i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Note: The standard tensor decomposition for matrix multiplication T[i,j,k]
        # usually maps to C[i,k] = sum_j A[i,j] B[j,k].
        # The problem statement says:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Output factor follows W[k*n+i, r].
        # This implies C[i, k] = sum_{j, c} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        C_calc = np.zeros((n, p), dtype=float)
        
        # Efficient check:
        # For each c, let U_c be vector of length n*m, V_c of length m*p, W_c of length p*n.
        # Reshape U_c to (n, m), V_c to (m, p), W_c to (p, n).
        # The contribution is (U_c.reshape(n,m) * V_c.reshape(m,p)) ? No.
        # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # C[i,k] = sum_j T[i,j,k].
        
        # Let's do it directly to be safe, dimensions are small.
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - C_true[i, k]) > 1e-9:
                    return False
    return True

def _reduce_rank(U, V, W, n, m, p, time_limit=10.0):
    """Attempt to reduce rank by removing linearly dependent or redundant columns."""
    rank = U.shape[1]
    start_time = time.time()
    
    # Check for zero columns
    cols_to_remove = []
    for c in range(rank):
        if np.allclose(U[:, c], 0) or np.allclose(V[:, c], 0) or np.allclose(W[:, c], 0):
            cols_to_remove.append(c)
            
    if cols_to_remove:
        keep = [c for c in range(rank) if c not in cols_to_remove]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = U.shape[1]

    # Heuristic: Check if removing any single column preserves the identity.
    # This is O(rank * verification_cost). Verification is O(n*m*p*rank).
    # Total O(rank^2 * n*m*p). For rank=56, n=2, m=4, p=8 -> 56^2 * 64 ~ 200k ops. Very fast.
    
    current_rank = rank
    changed = True
    while changed:
        changed = False
        current_rank = U.shape[1]
        for c in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing column c
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, num_trials=1):
                U = U_test
                V = V_test
                W = W_test
                changed = True
                break # Restart loop with new rank
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 120.0 # Safety margin from 150s limit
    
    try:
        # 1. Construct Block-Strassen Rank 56
        U, V, W = _construct_block_strassen()
        
        # 2. Verify
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Block-Strassen verification failed")
        
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        
        # 3. Attempt Rank Reduction
        if time.time() - start_time < DEADLINE - 20.0: # Leave 20s buffer
            U_red, V_red, W_red = _reduce_rank(U, V, W, n, m, p, time_limit=10.0)
            if U_red.shape[1] < best_rank:
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
                
        # 4. Return Best Result
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
        
    except Exception as e:
        # Fallback: Schoolbook Rank 64
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
