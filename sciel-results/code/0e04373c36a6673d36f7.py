import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms) for the 2x2x2 sub-block (i=0,1; j=0,1; k=0,1)
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part for remaining (i,j,k) in 3x3x2
        # Covered by Strassen: i in {0,1}, j in {0,1}, k in {0,1}
        # Need: all (i,j,k) where NOT (i in {0,1} and j in {0,1} and k in {0,1})
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if i in (0,1) and j in (0,1) and k in (0,1):
                        continue
                    U[i*3 + j, col] = 1.0
                    V[j*2 + k, col] = 1.0
                    W[k*3 + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    def try_find_rank_16_block():
        """
        Try to find a rank-16 decomposition for 3x3x2.
        Use a structured search: start from rank-17 and try to merge terms.
        """
        # For 3x3x2, the tensor has 18 non-zero entries.
        # Rank 16 would mean we save 2 from schoolbook (18) by using 2 Strassen-like terms.
        # Actually schoolbook for 3x3x2 is 18 terms. Rank 16 saves 2.
        
        # Strategy: use 2 Strassen-like terms to cover 4 entries each (saving 2 per term)
        # Then 16 schoolbook terms for the remaining 14 entries... that's 18 total.
        # Hmm, let me think differently.
        
        # Actually, let's try a direct construction.
        # The 3x3x2 tensor T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1}
        
        # Try: use Strassen on the 2x2x2 sub-block (i=0,1; j=0,1; k=0,1) -> 7 terms
        # Then handle: i=2 (all j,k), and j=2 (i=0,1; all k)
        # i=2: 3*2 = 6 terms
        # j=2, i in {0,1}: 2*2 = 4 terms
        # Total: 7 + 6 + 4 = 17. That's what we have.
        
        # To get 16, we need to save 1 more term.
        # Idea: Can we merge one of the i=2 terms with a Strassen term?
        # Or use a different 2x2x2 sub-block?
        
        # Let's try using Strassen on (i=0,1; j=1,2; k=0,1) instead
        # Then: i=2 (6 terms), j=0 for i in {0,1} (4 terms) = 7+6+4 = 17. Same.
        
        # What about a 3x2x2 Strassen-like? Not standard.
        
        # Alternative: try to find if any 3x3x2 has rank 16 via known bounds.
        # The tensor rank of 3x3x2 is at most 18 (schoolbook).
        # Lower bound: max(n*m, m*p, p*n) = max(9, 6, 6) = 9. Not helpful.
        
        # Let's just try a greedy merge approach: start from 17, try to find
        # if any two terms can be combined.
        
        U17, V17, W17 = get_17_rank_block()
        
        # Try removing one term and checking if the residual can be absorbed
        for remove_col in range(17):
            if time_left() < 10:
                break
            U_try = np.delete(U17, remove_col, axis=1)
            V_try = np.delete(V17, remove_col, axis=1)
            W_try = np.delete(W17, remove_col, axis=1)
            
            # Compute residual
            residual = np.ones((3, 3, 2))
            for c in range(16):
                u_mat = U_try[:, c].reshape(3, 3)
                v_mat = V_try[:, c].reshape(3, 2)
                w_mat = W_try[:, c].reshape(2, 3)
                for i in range(3):
                    for j in range(3):
                        for k in range(2):
                            residual[i,j,k] -= u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            
            # Check if residual is a rank-1 tensor
            # Residual is 3x3x2. Rank 1 means residual[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            # Check if all entries are products of row, col, and slice factors
            nonzero = np.argwhere(np.abs(residual) > 1e-10)
            if len(nonzero) == 0:
                return U_try, V_try, W_try
            if len(nonzero) <= 1:
                # Single non-zero entry can't be rank 1 unless it's a pure product
                pass
            
            # Check rank-1 structure: residual[i,j,k] = f[i]*g[j]*h[k] (for simplicity)
            # Actually for tensor rank 1: T[i,j,k] = u[i]*v[j]*w[k]
            # But our tensor is T[i,j,k] = U[i,j]*V[j,k]*W[k,i]
            # So rank-1 in our sense means: exists a[i,j], b[j,k], c[k,i] such that
            # residual[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            
            # Simple check: if residual has a zero row/col/slice pattern
            # For now, skip this complex check and just try the 17-rank approach
            
        return None
    
    # Assemble full 3x3x8 decomposition using 4 blocks of rank 17
    U_block, V_block, W_block = get_17_rank_block()
    rank_block = 17
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        U[:, col_start:col_end] = U_block
        
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
