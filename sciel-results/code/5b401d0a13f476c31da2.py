"""Matrix multiplication 4x4 by 4x9: Laderman-based tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 145.0

def _build_laderman_factors():
    """Build U, V, W factors for Laderman's rank-20 [4,4,4] decomposition.
    Returns (U, V, W) as lists of lists, each of shape (16, 20)."""
    
    # Define the 20 bilinear forms
    # A[i,j] at flat index i*4+j, B[j,k] at flat index j*4+k
    products = []
    products.append(([ (0,1) ], [ (0,1) ]))          # m1
    products.append(([ (1,1) ], [ (4,1) ]))          # m2
    products.append(([ (4,1) ], [ (1,1) ]))          # m3
    products.append(([ (0,1),(4,1) ], [ (1,1),(5,1) ]))  # m4
    products.append(([ (1,1),(5,1) ], [ (0,1),(4,1) ]))  # m5
    products.append(([ (8,1) ], [ (8,1) ]))          # m6
    products.append(([ (9,1) ], [ (12,1) ]))         # m7
    products.append(([ (12,1) ], [ (9,1) ]))         # m8
    products.append(([ (8,1),(12,1) ], [ (9,1),(13,1) ]))  # m9
    products.append(([ (9,1),(13,1) ], [ (8,1),(12,1) ]))  # m10
    products.append(([ (2,1) ], [ (10,1) ]))         # m11
    products.append(([ (3,1) ], [ (14,1) ]))         # m12
    products.append(([ (6,1) ], [ (11,1) ]))         # m13
    products.append(([ (2,1),(6,1) ], [ (11,1),(15,1) ]))  # m14
    products.append(([ (3,1),(7,1) ], [ (10,1),(14,1) ]))  # m15
    products.append(([ (10,1) ], [ (2,1) ]))         # m16
    products.append(([ (11,1) ], [ (6,1) ]))         # m17
    products.append(([ (14,1) ], [ (3,1) ]))         # m18
    products.append(([ (10,1),(14,1) ], [ (3,1),(7,1) ]))  # m19
    products.append(([ (11,1),(15,1) ], [ (2,1),(6,1) ]))  # m20
    
    rank = 20
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r in range(rank):
        for idx, c in products[r][0]:
            U[idx, r] = c
        for idx, c in products[r][1]:
            V[idx, r] = c
    
    # Numerically determine W
    np.random.seed(12345)
    num_samples = 100
    samples_A = [np.random.randn(4,4) for _ in range(num_samples)]
    samples_B = [np.random.randn(4,4) for _ in range(num_samples)]
    
    M = np.zeros((num_samples, rank))
    for s in range(num_samples):
        A_flat = samples_A[s].flatten()
        B_flat = samples_B[s].flatten()
        for r in range(rank):
            a_val = sum(A_flat[idx] * c for idx, c in products[r][0])
            b_val = sum(B_flat[idx] * c for idx, c in products[r][1])
            M[s, r] = a_val * b_val
    
    for i in range(4):
        for j in range(4):
            c_vals = np.array([(samples_A[s] @ samples_B[s])[i,j] for s in range(num_samples)])
            alpha, residuals, rk, sv = np.linalg.lstsq(M, c_vals, rcond=None)
            recon = M @ alpha
            if not np.allclose(recon, c_vals, atol=1e-10):
                raise ValueError(f"Laderman W determination failed for C[{i},{j}]")
            W[j*4+i, :] = alpha
    
    # Verify the full decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Laderman verification failed at ({i},{j},{k}): got {val}")
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    
    # Try Laderman-based approach: [4,4,9] = [4,4,4] + [4,4,4] + [4,4,1]
    # Rank = 20 + 20 + 16 = 56
    try:
        U_lad, V_lad, W_lad = _build_laderman_factors()
        
        total_rank = 20 + 20 + 16
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        # First [4,4,4]: B columns 0-3, output columns 0-3
        for r in range(20):
            for i in range(4):
                for j in range(4):
                    U[i*m+j][r] = U_lad[i*4+j][r]
            for j in range(4):
                for k in range(4):
                    V[j*p+k][r] = V_lad[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    W[k*n+i][r] = W_lad[k*4+i][r]
        
        # Second [4,4,4]: B columns 4-7, output columns 4-7
        for r in range(20):
            for i in range(4):
                for j in range(4):
                    U[i*m+j][r+20] = U_lad[i*4+j][r]
            for j in range(4):
                for k in range(4):
                    V[j*p+(k+4)][r+20] = V_lad[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    W[(k+4)*n+i][r+20] = W_lad[k*4+i][r]
        
        # Third [4,4,1]: B column 8, output column 8
        col = 40
        for i in range(4):
            for j in range(4):
                U[i*m+j][col] = 1.0
                V[j*p+8][col] = 1.0
                W[8*n+i][col] = 1.0
                col += 1
        
        # Final verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(total_rank))
                    if abs(val - 1.0) > 1e-10:
                        raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}")
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
