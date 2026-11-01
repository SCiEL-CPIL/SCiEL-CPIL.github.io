"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    Uses Winograd Fourier Transform Algorithm (WFTA) via Kronecker substitution.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # = 24
    
    # Try WFTA-based approach first
    wfta_result = _try_wfta(n, m, p)
    if wfta_result is not None:
        return wfta_result
    
    # Fallback to Strassen-based rank 21
    strassen_result = _try_strassen(n, m, p)
    if strassen_result is not None:
        return strassen_result
    
    # Final fallback: schoolbook
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

def _try_wfta(n, m, p):
    """
    Apply Winograd Fourier Transform Algorithm (WFTA) to the tensor.
    
    For multiplying 2x2 by 2x6, we can view this as:
    - Split A into 2x2 blocks (already one block since m=2)
    - Split B into 2x3 blocks along columns
    - Use polynomial multiplication via Kronecker substitution
    
    The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can use the fact that for each i, this is a dot product of length m=2.
    Using WFTA with base-2 representation, we can reduce the rank.
    
    For 2x2x6: We can decompose B's columns into groups and use FFT-like transforms.
    """
    # For 2x2x6, try rank 16 via WFTA
    # The idea: use 2-point FFT for the inner product dimension
    
    # Actually, let's try a more systematic approach:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is already a simple dot product. The tensor has rank at most 2*2*6=24.
    
    # Let's try to find a better decomposition using the structure.
    # Use the fact that we can apply Strassen to 2x2 multiplications.
    
    # For 2x2x6, we can view it as 6 independent 2x1 multiplications,
    # or as 3 independent 2x2 multiplications.
    
    # 3x(2x2) via Strassen: each 2x2 takes rank 7, so 3*7=21
    # But maybe we can do better by sharing structure.
    
    # Let's try rank 14: 2x(2x3) where 2x3 can be done with rank 7 (Strassen on 2x2 + 1 direct)
    # Actually 2x3: C[i,k] = sum_j A[i,j] B[j,k] for j=0,1, k=0,1,2
    # This is a 2x2x3 tensor. Rank of 2x2x3?
    
    # Let me just try to construct explicit decompositions and verify.
    
    # Try rank 12: this is the lower bound max(nm, mp, pn) = max(4, 12, 12) = 12
    # Can we achieve rank 12?
    
    # For rank 12, we need U(4x12), V(12x12), W(12x12)
    # This is a very tight constraint.
    
    # Let's try to construct it directly.
    
    # Actually, let's try a different approach: use the known optimal rank for small tensors.
    # The rank of M_{2,2,2} is 7 (Strassen).
    # The rank of M_{2,2,3} is at most 7*2 = 14? Or can we do better?
    
    # Let me try to just implement a search for low-rank decompositions with integer/half-integer entries.
    
    nm_val, mp_val, pn_val = n * m, m * p, p * n
    
    # Build tensor as list of nonzero entries
    nonzeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.append((a, b, c))
    
    # Try to find rank-r decomposition with r from lower bound up
    start_rank = max(nm_val, mp_val, pn_val)  # 12
    
    start_time = time.time()
    deadline = start_time + 110
    
    for rank in range(start_rank, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _search_rank(n, m, p, nm_val, mp_val, pn_val, rank, nonzeros)
        if result is not None:
            return result
    
    return None

def _search_rank(n, m, p, nm, mp, pn, rank, nonzeros):
    """Search for rank-r decomposition."""
    # Use the structure of the tensor to guide the search.
    # The tensor T[a,b,c] = 1 iff there exist i,j,k with:
    #   a = i*m+j, b = j*p+k, c = k*n+i
    # i.e., a mod m = b div p = (c mod n) div ... wait let me re-derive.
    
    # a = i*m + j, so i = a div m, j = a mod m
    # b = j*p + k, so j = b div p, k = b mod p  
    # c = k*n + i, so k = c div n, i = c mod n
    
    # Consistency: j = a mod m = b div p, i = a div m = c mod n, k = b mod p = c div n
    
    # For each rank, try random initialization + ALS
    np.random.seed(rank * 1000)
    
    T_flat = np.zeros((nm, mp * pn))
    T_flat_V = np.zeros((mp, nm * pn))
    T_flat_W = np.zeros((pn, nm * mp))
    
    for (a, b, c) in nonzeros:
        T_flat[a, b * pn + c] = 1.0
        T_flat_V[b, a * pn + c] = 1.0
        T_flat_W[c, a * mp + b] = 1.0
    
    n_trials = 10
    for trial in range(n_trials):
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        converged = False
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U
            M_U = np.zeros((mp * pn, rank))
            for b in range(mp):
                for c in range(pn):
                    M_U[b * pn + c, :] = V[b, :] * W[c, :]
            
            try:
                A_U = M_U.T @ M_U + 1e-8 * np.eye(rank)
                B_U = M_U.T @ T_flat.T
                U_new = np.linalg.solve(A_U, B_U).T
            except np.linalg.LinAlgError:
                U_new = U
            
            # Update V
            M_V = np.zeros((nm * pn, rank))
            for a in range(nm):
                for c in range(pn):
                    M_V[a * pn + c, :] = U_new[a, :] * W[c, :]
            
            try:
                A_V = M_V.T @ M_V + 1e-8 * np.eye(rank)
                B_V = M_V.T @ T_flat_V.T
                V_new = np.linalg.solve(A_V, B_V).T
            except np.linalg.LinAlgError:
                V_new = V
            
            # Update W
            M_W = np.zeros((nm * mp, rank))
            for a in range(nm):
                for b in range(mp):
                    M_W[a * mp + b, :] = U_new[a, :] * V_new[b, :]
            
            try:
                A_W = M_W.T @ M_W + 1e-8 * np.eye(rank)
                B_W = M_W.T @ T_flat_W.T
                W_new = np.linalg.solve(A_W, B_W).T
            except np.linalg.LinAlgError:
                W_new = W
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            error = 0.0
            for (a, b, c) in nonzeros:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                error += (val - 1.0) ** 2
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if (a, b, c) not in set(nonzeros):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += val ** 2
            
            if error < 1e-10:
                converged = True
                break
        
        if converged:
            # Verify
            max_err = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        expected = 1.0 if (a, b, c) in set(nonzeros) else 0.0
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-8:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None

def _try_strassen(n, m, p):
    """Try Strassen-based decomposition for rank 21."""
    # Decompose 2x2x6 as 3 independent 2x2x2 multiplications
    # Each 2x2x2 can be done with Strassen (rank 7)
    # Total rank = 3 * 7 = 21
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2x6, we have 3 blocks of 2x2x2 (columns 0-1, 2-3, 4-5)
    strassen_coeffs = [
        # (U_coeff, V_coeff, W_coeff) for each of 7 multiplications
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, -1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, -1, 0, 1]),
    ]
    
    for block in range(3):
        # block covers columns 2*block to 2*block+1 of B, and corresponding columns of C
        col_offset = 2 * block
        
        for s in range(7):
            col_idx = block * 7 + s
            u_coeff, v_coeff, w_coeff = strassen_coeffs[s]
            
            # u_coeff: [A00, A01, A10, A11] -> rows in U
            # For block, A rows are i*m+j = i*2+j, so rows 0,1,2,3
            U[0, col_idx] += u_coeff[0]  # A[0,0] -> row 0
            U[1, col_idx] += u_coeff[1]  # A[0,1] -> row 1
            U[2, col_idx] += u_coeff[2]  # A[1,0] -> row 2
            U[3, col_idx] += u_coeff[3]  # A[1,1] -> row 3
            
            # v_coeff: [B00, B01, B10, B11] -> rows in V
            # B[j,k] -> row j*p+k = j*6+k
            # For block, B columns are col_offset, col_offset+1
            # B[0, col_offset] -> row 0*6+col_offset = col_offset
            # B[0, col_offset+1] -> row col_offset+1
            # B[1, col_offset] -> row 1*6+col_offset = 6+col_offset
            # B[1, col_offset+1] -> row 6+col_offset+1
            V[col_offset, col_idx] += v_coeff[0]       # B[0, col_offset]
            V[col_offset + 1, col_idx] += v_coeff[1]   # B[0, col_offset+1]
            V[6 + col_offset, col_idx] += v_coeff[2]   # B[1, col_offset]
            V[6 + col_offset + 1, col_idx] += v_coeff[3]  # B[1, col_offset+1]
            
            # w_coeff: [C00, C01, C10, C11] -> rows in W
            # C[i,k] -> row k*n+i = k*2+i
            # For block, C columns are col_offset, col_offset+1
            # C[0, col_offset] -> row col_offset*2+0 = 2*col_offset
            # C[0, col_offset+1] -> row (col_offset+1)*2+0 = 2*col_offset+2
            # C[1, col_offset] -> row col_offset*2+1 = 2*col_offset+1
            # C[1, col_offset+1] -> row (col_offset+1)*2+1 = 2*col_offset+3
            W[2 * col_offset, col_idx] += w_coeff[0]       # C[0, col_offset]
            W[2 * col_offset + 2, col_idx] += w_coeff[1]   # C[0, col_offset+1]
            W[2 * col_offset + 1, col_idx] += w_coeff[2]   # C[1, col_offset]
            W[2 * col_offset + 3, col_idx] += w_coeff[3]   # C[1, col_offset+1]
    
    # Verify
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    max_err = 0.0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = abs(val - T_full[a, b, c])
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-9:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return None
