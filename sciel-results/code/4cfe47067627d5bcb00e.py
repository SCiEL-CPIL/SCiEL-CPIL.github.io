"""Matrix multiplication 2x4x6 tensor decomposition with structured rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    schoolbook_rank = nm * mp * pn // (n * m)  # This is wrong, let me fix
    schoolbook_rank = n * m * p  # 48

    def verify_numpy(U, V, W, rank):
        """Verify exact decomposition using numpy for speed."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_c = Ua[:, r].reshape(n, m)
            v_c = Va[:, r].reshape(m, p)
            w_c = Wa[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def clean_halfint(arr):
        """Clean values to nearest half-integer."""
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val) < 1e-9:
                    clean_row.append(0)
                else:
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) < 1e-6:
                        clean_row.append(rounded)
                    else:
                        clean_row.append(rounded)
            result.append(clean_row)
        return result

    # Try structured decomposition using block structure
    # Split p=6 into three blocks of 2
    # For each block, we have 2x4 * 4x2 = 2x2
    # Rank of 2x4*4x2: can we do better than 8?
    
    # Let's try a direct approach: use the fact that the tensor is all-ones
    # T[i,j,k] = 1 for all i,j,k
    
    # Strategy 1: Try rank 24 (theoretical lower bound)
    # The lower bound is max(nm, mp, pn) = max(8, 24, 12) = 24
    
    # Let's try to construct a rank-24 decomposition
    # If we can find one, that would be optimal
    
    # Approach: Use the structure that T is all-ones
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # One idea: if we set U[i*m+j, r] = a_{i,j,r}, V[j*p+k, r] = b_{j,k,r}, W[k*n+i, r] = c_{k,i,r}
    # We need sum_r a_{i,j,r} * b_{j,k,r} * c_{k,i,r} = 1 for all i,j,k
    
    # Let's try a specific structured approach
    # Since n=2, we have i in {0,1}
    # Let's try to use the symmetry
    
    # Try random search with structured initialization
    np.random.seed(12345)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 24 to 47
    for rank in range(24, min(schoolbook_rank, 48)):
        if time.time() > deadline - 10:
            break
        
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured pattern
            U = np.zeros((nm, rank), dtype=np.float64)
            V = np.zeros((mp, rank), dtype=np.float64)
            W = np.zeros((pn, rank), dtype=np.float64)
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                # Sparse random
                for r in range(rank):
                    # Randomly choose a few non-zero entries
                    nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
                    nz_v = np.random.choice(mp, size=min(5, mp), replace=False)
                    nz_w = np.random.choice(pn, size=min(3, pn), replace=False)
                    U[nz_u, r] = np.random.choice([-1, 0.5, 1, -0.5], size=len(nz_u))
                    V[nz_v, r] = np.random.choice([-1, 0.5, 1, -0.5], size=len(nz_v))
                    W[nz_w, r] = np.random.choice([-1, 0.5, 1, -0.5], size=len(nz_w))
            elif attempt % 3 == 1:
                # Dense with small values
                U[:] = np.random.choice([-0.5, 0.5, 1, -1], size=(nm, rank)) * 0.5
                V[:] = np.random.choice([-0.5, 0.5, 1, -1], size=(mp, rank)) * 0.5
                W[:] = np.random.choice([-0.5, 0.5, 1, -1], size=(pn, rank)) * 0.5
            else:
                # Block-structured
                # Divide rank into blocks
                block_size = rank // 3
                for block in range(3):
                    start_r = block * block_size
                    end_r = (block + 1) * block_size if block < 2 else rank
                    for r in range(start_r, end_r):
                        # Each block handles part of the computation
                        U[:, r] = np.random.choice([0, 0.5, 1], size=nm)
                        V[:, r] = np.random.choice([0, 0.5, 1], size=mp)
                        W[:, r] = np.random.choice([0, 0.5, 1], size=pn)
            
            # Verify
            if verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank):
                U_clean = clean_halfint(U.tolist())
                V_clean = clean_halfint(V.tolist())
                W_clean = clean_halfint(W.tolist())
                best_result = (U_clean, V_clean, W_clean, rank)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
