import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        n, m, p = 2, 2, 2
        rank = 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Standard Strassen constants
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W such that sum U[i,j]V[j,k]W[k,i] = C[i,k]
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the factors for each term c in 0..6
        # Term 0: M1. U: A11+A22. V: B11+B22. W: C11, C22.
        U[:, 0] = [1, 0, 0, 1] # A11, A12, A21, A22
        V[:, 0] = [1, 1, 1, 1] # B11, B12, B21, B22
        W[:, 0] = [1, 0, 0, 1] # C11, C12, C21, C22 -> W[k*2+i]
        # W[0] is C11 (k=0,i=0), W[1] is C21 (k=0,i=1)? No.
        # W index: k*n+i. n=2.
        # W[0] = W[0*2+0] = C[0,0]
        # W[1] = W[0*2+1] = C[0,1]
        # W[2] = W[1*2+0] = C[1,0]
        # W[3] = W[1*2+1] = C[1,1]
        
        # So for M1, it contributes to C11 and C22.
        # W[:, 0] should be 1 at indices 0 and 3.
        W[:, 0] = [1, 0, 0, 1]
        
        # Term 1: M2 = (A21+A22)B11. Contributes to C21, C22 (with -1).
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 0, 1, -1] # C21 (+), C22 (-)
        
        # Term 2: M3 = A11(B21-B22). Contributes to C12, C22.
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        W[:, 2] = [0, 1, 0, 1] # C12 (+), C22 (+)
        
        # Term 3: M4 = A22(B12-B11). Contributes to C11 (-), C21 (+).
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [-1, 1, 0, 0]
        W[:, 3] = [-1, 0, 1, 0] # C11 (-), C21 (+)
        
        # Term 4: M5 = (A11+A12)B22. Contributes to C11, C12.
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [1, 1, 0, 0] # C11 (+), C12 (+)
        
        # Term 5: M6 = (A21-A11)(B11+B12). Contributes to C22.
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[:, 5] = [0, 0, 0, 1] # C22 (+)
        
        # Term 6: M7 = (A12-A22)(B21+B22). Contributes to C11.
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[:, 6] = [1, 0, 0, 0] # C11 (+)
        
        return U, V, W

    def combine_blocks(blocks, n, m, p, split_dim, split_point):
        """
        Combine a list of (U, V, W) blocks into a single block for n x m x p.
        split_dim: 0 for n (rows of A, cols of C), 1 for m (cols of A, rows of B), 2 for p (cols of B, rows of C)
        split_point: not used directly, assumes blocks are already sized correctly and just need concatenation.
        The caller must ensure the blocks cover the full n, m, p dimensions.
        """
        U_total = np.hstack([b[0] for b in blocks])
        V_total = np.hstack([b[1] for b in blocks])
        W_total = np.hstack([b[2] for b in blocks])
        return U_total, V_total, W_total

    # 1. 2x2x2 (Strassen)
    U_222, V_222, W_222 = strassen_2x2x2()
    
    # 2. 4x2x2: Split n=4 into 2+2.
    # Block 1: 2x2x2 (top-left)
    # Block 2: 2x2x2 (bottom-left)
    # We need to embed 2x2x2 into 4x2x2.
    def embed(U, V, W, n, m, p, n_off=0, m_off=0, p_off=0, n_size=None, m_size=None, p_size=None):
        if n_size is None: n_size = n
        if m_size is None: m_size = m
        if p_size is None: p_size = p
        
        U_new = np.zeros((n * m, U.shape[1]))
        V_new = np.zeros((m * p, V.shape[1]))
        W_new = np.zeros((p * n, W.shape[1]))
        
        # Copy U
        for i in range(n_size):
            for j in range(m_size):
                U_new[(i + n_off) * m + (j + m_off), :] = U[i * m_size + j, :] # Wait, U shape is (n_size*m_size, rank)
                # Actually U is passed as (n_size*m_size, rank).
                # We need to map local i,j to global i,j.
        
        # Simpler: just construct the full matrices from scratch for each block type.
        pass

    # Let's just build the specific blocks needed for the 112 rank solution.
    
    # Block: 4x2x2 (Rank 14)
    # Split n=4 into 2,2.
    U_422 = np.zeros((4*2, 14))
    V_422 = np.zeros((2*2, 14))
    W_422 = np.zeros((2*4, 14))
    
    # Top half (i=0,1)
    U_422[0:2*2, 0:7] = U_222
    V_422[:, 0:7] = V_222
    W_422[0:2, 0:7] = W_222[0:2, :] # W is (p*n, rank) = (2*4, rank). p=2.
    # W indices: k*4+i.
    # For top block (i=0,1), W[k*4+0] and W[k*4+1].
    # Original W_222 is (2*2, 7) = (4, 7).
    # W_222[k*2+i].
    # We need to map to W_422[k*4+i].
    for k in range(2):
        for i in range(2):
            W_422[k*4+i, 0:7] = W_222[k*2+i, 0:7]
            
    # Bottom half (i=2,3)
    U_422[2*2:4*2, 7:14] = U_222
    V_422[:, 7:14] = V_222
    for k in range(2):
        for i in range(2):
            W_422[k*4+(i+2), 7:14] = W_222[k*2+i, 0:7]
            
    # Block: 4x2x1 (Rank 8) - Schoolbook
    U_421, V_421, W_421 = schoolbook(4, 2, 1)
    
    # Block: 4x2x3 (Rank 22)
    # Split p=3 into 2+1.
    U_423 = np.hstack([U_422, U_421])
    V_423 = np.hstack([V_422, V_421])
    W_423 = np.hstack([W_422, W_421])
    # Note: W_422 is (2*4, 14). W_421 is (1*4, 8).
    # We need W_423 to be (3*4, 22).
    # W_423[k*4+i].
    # For p=0,1 (from 422), k is 0,1.
    # For p=2 (from 421), k is 2.
    W_423_full = np.zeros((3*4, 22))
    W_423_full[0:2*4, 0:14] = W_422
    W_423_full[2*4:3*4, 14:22] = W_421
    
    # Block: 4x4x3 (Rank 44)
    # Split m=4 into 2+2.
    U_443 = np.zeros((4*4, 44))
    V_443 = np.zeros((4*3, 44))
    W_443 = np.zeros((3*4, 44))
    
    # Left half (j=0,1)
    U_443[:, 0:22] = U_423 # U is (n*m, rank). n=4, m=2.
    # We need to map U_423 (4*2, 22) to U_443 (4*4, 44).
    # U_443[i*4+j]. For j=0,1, use U_423[i*2+j].
    for i in range(4):
        for j in range(2):
            U_443[i*4+j, 0:22] = U_423[i*2+j, 0:22]
            
    # V_423 is (2*3, 22). V_443 is (4*3, 44).
    # V_443[j*3+k]. For j=0,1, use V_423[j*3+k].
    V_443[0:2*3, 0:22] = V_423
    
    # W_423 is (3*4, 22). W_443 is (3*4, 44).
    W_443[:, 0:22] = W_423
    
    # Right half (j=2,3)
    # Same structure, shifted j by 2.
    for i in range(4):
        for j in range(2):
            U_443[i*4+(j+2), 22:44] = U_423[i*2+j, 0:22]
            
    V_443[2*3:4*3, 22:44] = V_423
    W_443[:, 22:44] = W_423
    
    # Block: 4x1x3 (Rank 12) - Schoolbook
    U_413, V_413, W_413 = schoolbook(4, 1, 3)
    
    # Block: 4x5x3 (Rank 56)
    # Split m=5 into 4+1.
    U_453 = np.zeros((4*5, 56))
    V_453 = np.zeros((5*3, 56))
    W_453 = np.zeros((3*4, 56))
    
    # Left (j=0..3)
    for i in range(4):
        for j in range(4):
            U_453[i*5+j, 0:44] = U_443[i*4+j, 0:44]
            
    V_453[0:4*3, 0:44] = V_443
    W_453[:, 0:44] = W_443
    
    # Right (j=4)
    for i in range(4):
        U_453[i*5+4, 44:56] = U_413[i*1+0, 0:12]
        
    V_453[4*3:5*3, 44:56] = V_413
    W_453[:, 44:56] = W_413
    
    # Final: 4x5x6 (Rank 112)
    # Split p=6 into 3+3.
    U_final = np.hstack([U_453, U_453])
    V_final = np.hstack([V_453, V_453])
    
    W_final = np.zeros((6*4, 112))
    W_final[0:3*4, 0:56] = W_453
    W_final[3*4:6*4, 56:112] = W_453
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
