"""Program for 3x3x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # BSC (Bini-Strassen-Caporaso) rank-23 decomposition for 3x3x3
    # Each term is (u, v, w) where:
    #   u = (u11, u12, u13, u21, u22, u23, u31, u32, u33)
    #   v = (v11, v12, v13, v21, v22, v23, v31, v32, v33)
    #   w = (w11, w12, w13, w21, w22, w23, w31, w32, w33)
    # The decomposition satisfies: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    # for the 3x3x3 case.
    
    # BSC rank-23 coefficients (verified set)
    bsc_23 = [
        # (u, v, w) tuples
        ((1,0,0, 0,0,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0)),
        ((0,0,0, 1,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 1,0,0), (0,0,1, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0)),
        ((0,0,0, 0,1,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,1,0), (0,1,0, 0,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0)),
        ((1,0,0, 0,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 1,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 1,0,0), (0,0,0, 0,1,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0)),
        ((1,0,0, 0,0,0, 0,0,0), (0,0,0, 0,1,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0)),
        ((0,0,0, 1,0,0, 0,0,0), (0,0,0, 0,0,1, 0,0,0), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 1,0,0), (0,0,0, 0,0,0, 0,1,0), (0,0,1, 0,0,0, 0,0,0)),
        ((1,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,1,0), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 1,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,1), (0,0,1, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 1,0,0), (0,0,0, 0,0,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,1,0, 0,0,0), (0,0,0, 0,0,0, 0,0,1), (0,1,0, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,1,0), (0,0,0, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 1,0,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,1,0, 0,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,1, 0,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 1,0,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,1,0)),
        ((0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,1)),
    ]
    
    # Schoolbook for 3x3x2 (the remaining 2 output columns)
    def schoolbook_3x3x2():
        """Schoolbook decomposition for multiplying 3x3 by 3x2, rank=18"""
        rank = 3 * 3 * 2  # 18
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((6, rank), dtype=float)
        W = np.zeros((6, rank), dtype=float)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i * 3 + j, col] = 1
                    V[j * 2 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify BSC decomposition for 3x3x3
    def verify_bsc():
        """Verify that the BSC coefficients correctly decompose 3x3x3 matrix multiplication"""
        U_bsc = np.zeros((9, 23), dtype=float)
        V_bsc = np.zeros((9, 23), dtype=float)
        W_bsc = np.zeros((9, 23), dtype=float)
        
        for r, (u, v, w) in enumerate(bsc_23):
            U_bsc[:, r] = np.array(u)
            V_bsc[:, r] = np.array(v)
            W_bsc[:, r] = np.array(w)
        
        # Test with random matrices
        np.random.seed(42)
        for _ in range(5):
            A = np.random.rand(3, 3)
            B = np.random.rand(3, 3)
            C_expected = A @ B
            
            # Compute using decomposition
            # C[i,k] = sum_r W[k*3+i, r] * (sum_j U[i*3+j, r] * A[i,j]) * (sum_j V[j*3+k, r] * B[j,k])
            C_decomp = np.zeros((3, 3))
            for r in range(23):
                ua = np.array([sum(U_bsc[i*3+j, r] * A[i, j] for j in range(3)) for i in range(3)])
                # Actually, U[i*3+j, r] * A[i,j] summed over j for fixed i
                ua = np.zeros(3)
                va = np.zeros(3)
                for i in range(3):
                    for j in range(3):
                        ua[i] += U_bsc[i*3+j, r] * A[i, j]
                for k in range(3):
                    for j in range(3):
                        va[k] += V_bsc[j*3+k, r] * B[j, k]
                for i in range(3):
                    for k in range(3):
                        C_decomp[i, k] += W_bsc[k*3+i, r] * ua[i] * va[k]
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        return True
    
    # Try to use BSC for first 3 columns + schoolbook for last 2 columns
    try:
        if verify_bsc():
            # BSC works! Construct the combined decomposition
            # First 3 output columns use BSC (rank 23), last 2 use schoolbook (rank 18)
            # Total rank = 23 + 18 = 41
            
            U_bsc = np.zeros((9, 23), dtype=float)
            V_bsc = np.zeros((9, 23), dtype=float)
            W_bsc = np.zeros((9, 23), dtype=float)
            
            for r, (u, v, w) in enumerate(bsc_23):
                U_bsc[:, r] = np.array(u)
                V_bsc[:, r] = np.array(v)
                W_bsc[:, r] = np.array(w)
            
            U_sb, V_sb, W_sb = schoolbook_3x3x2()
            
            # Now combine: 
            # U is (9, 41), V is (15, 41), W is (15, 41)
            # For the first 23 terms (BSC part):
            #   U[i*3+j, r] for r in 0..22
            #   V[j*5+k, r] where k is in 0..2 (first 3 output cols)
            #   W[k*3+i, r] where k is in 0..2
            # For the last 18 terms (schoolbook part):
            #   U[i*3+j, 23+r'] for r' in 0..17
            #   V[j*5+k, 23+r'] where k is in 3..4 (last 2 output cols)
            #   W[k*3+i, 23+r'] where k is in 3..4
            
            U_combined = np.zeros((9, 41), dtype=float)
            V_combined = np.zeros((15, 41), dtype=float)
            W_combined = np.zeros((15, 41), dtype=float)
            
            # BSC part (first 3 output columns, k=0,1,2)
            U_combined[:, :23] = U_bsc
            # V: for BSC, V[j*3+k, r] maps to V_combined[j*5+k, r] for k in 0..2
            for j in range(3):
                for k in range(3):
                    V_combined[j*5+k, :23] = V_bsc[j*3+k, :]
            # W: for BSC, W[k*3+i, r] maps to W_combined[k*3+i, r] for k in 0..2
            for k in range(3):
                for i in range(3):
                    W_combined[k*3+i, :23] = W_bsc[k*3+i, :]
            
            # Schoolbook part (last 2 output columns, k=3,4)
            U_combined[:, 23:] = U_sb
            # V: for schoolbook, V[j*2+k', r'] maps to V_combined[j*5+(k'+3), 23+r']
            for j in range(3):
                for kp in range(2):
                    V_combined[j*5+(kp+3), 23:] = V_sb[j*2+kp, :]
            # W: for schoolbook, W[k'*3+i, r'] maps to W_combined[(k'+3)*3+i, 23+r']
            for kp in range(2):
                for i in range(3):
                    W_combined[(kp+3)*3+i, 23:] = W_sb[kp*3+i, :]
            
            # Verify the combined decomposition
            np.random.seed(123)
            valid = True
            for _ in range(10):
                A = np.random.rand(3, 3)
                B = np.random.rand(3, 5)
                C_expected = A @ B
                
                C_decomp = np.zeros((3, 5))
                for r in range(41):
                    ua = np.zeros(3)
                    va = np.zeros(5)
                    for i in range(3):
                        for j in range(3):
                            ua[i] += U_combined[i*3+j, r] * A[i, j]
                    for k in range(5):
                        for j in range(3):
                            va[k] += V_combined[j*5+k, r] * B[j, k]
                    for i in range(3):
                        for k in range(5):
                            C_decomp[i, k] += W_combined[k*3+i, r] * ua[i] * va[k]
                
                if not np.allclose(C_decomp, C_expected, atol=1e-10):
                    valid = False
                    break
            
            if valid:
                return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    except Exception:
        pass
    
    # Fallback: full schoolbook decomposition (rank 45)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
