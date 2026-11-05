import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    def time_left():
        return deadline - time.time()

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    # Rank 28 decomposition (Strassen + Schoolbook)
    def build_rank_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank
        return U, V, W

    # Try to find a rank-27 decomposition by removing one term from rank-28
    # and checking if the residual can be absorbed
    def try_rank_27(U28, V28, W28, T):
        if time_left() < 5:
            return None
        
        rank28 = 28
        # For each term r, check if removing it leaves a tensor that can be 
        # represented with the remaining 27 terms plus one new term
        # Actually, simpler: check if any single term is linearly dependent
        # on the others in the tensor sense
        
        # Alternative: try removing each term and see if the residual tensor
        # has rank 1 (can be added as one more term)
        for r_remove in range(rank28):
            if time_left() < 3:
                break
            
            # Remove term r_remove
            mask = [x for x in range(rank28) if x != r_remove]
            U_r = U28[:, mask]
            V_r = V28[:, mask]
            W_r = W28[:, mask]
            
            # Compute residual
            T_rec = np.einsum('ir,vr,wr->ivw', U_r, V_r, W_r, optimize=True)
            residual = T - T_rec
            
            # Check if residual is rank-1 tensor
            # Reshape and check if it's a sum of rank-1 tensors
            # A tensor is rank-1 iff it can be written as a ⊗ b ⊗ c
            # Check by flattening and looking at the structure
            res_flat = residual.reshape(-1)
            if np.allclose(res_flat, 0, atol=1e-10):
                # Already rank 27
                return U_r, V_r, W_r
            
            # Try to see if residual is rank 1
            # For a rank-1 tensor A[i,j,k] = a[i]*b[j]*c[k]
            # We can check by looking at 2D slices
            # Take two different slices and see if they're proportional
            found_rank1 = False
            for i1 in range(nm):
                for i2 in range(nm):
                    if i1 == i2:
                        continue
                    slice1 = residual[i1, :, :]
                    slice2 = residual[i2, :, :]
                    if np.any(np.abs(slice1) > 1e-10) and np.any(np.abs(slice2) > 1e-10):
                        # Check if slice2 = alpha * slice1
                        # Find alpha from a non-zero entry
                        nz1 = np.nonzero(np.abs(slice1) > 1e-10)
                        nz2 = np.nonzero(np.abs(slice2) > 1e-10)
                        if len(nz1[0]) > 0 and len(nz2[0]) > 0:
                            idx1 = (nz1[0][0], nz1[1][0])
                            idx2 = (nz2[0][0], nz2[1][0])
                            if slice1[idx1] != 0:
                                alpha = slice2[idx2] / slice1[idx1] if slice1[idx2] != 0 else 0
                                # This approach is getting complicated
                                pass
                    break
                break
            break
        
        return None

    # Try a subspace-partitioning based construction
    def build_subspace_decomposition():
        """
        Partition the j-index into {0,1} and {2}.
        For j in {0,1}: use Strassen for k in {0,1}, and handle k in {2,3,4} separately
        For j=2: handle all k values
        """
        if time_left() < 10:
            return None
        
        U = np.zeros((nm, 30))  # max possible
        V = np.zeros((mp, 30))
        W = np.zeros((pn, 30))
        col = 0

        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal col
            U[u_idx, col] += u_val
            V[v_idx, col] += v_val
            W[w_idx, col] += w_val
            col += 1

        # Sub-block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, rank 7 (Strassen)
        def strassen_2x2x2(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # 7 Strassen terms
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        strassen_2x2x2(0, 1, 0, 1, 0, 1)  # 7 terms, col=7

        # Sub-block 2: i in {0,1}, j in {0,1}, k in {2,3,4} -> 2x2x3
        # This is multiplying a 2x2 matrix by a 2x3 matrix
        # Rank of 2x2x3 tensor: let's use schoolbook for now (12 terms) but try to optimize
        # Schoolbook: 2*2*3 = 12 terms
        for i in range(2):
            for j in range(2):
                for k in range(2, 5):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        # col = 7 + 12 = 19

        # Sub-block 3: i in {0,1}, j=2, k in {0,4} (all k)
        # This is 2x1 by 1x5, which is 10 schoolbook terms
        for i in range(2):
            for k in range(5):
                U[i*m+2, col] = 1
                V[2*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
        # col = 19 + 10 = 29

        # Total: 29 terms, worse than 28!
        # Let's try a different partition
        
        return None

    # Build the rank-28 decomposition
    U28, V28, W28 = build_rank_28()
    if check_decomposition(U28, V28, W28, T):
        # Try to reduce to rank 27
        result_27 = try_rank_27(U28, V28, W28, T)
        if result_27 is not None:
            U27, V27, W27 = result_27
            if check_decomposition(U27, V27, W27, T):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
        
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
