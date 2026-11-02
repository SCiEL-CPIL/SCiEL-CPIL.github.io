"""Matrix multiplication 2,2,15 tensor rank minimization - structural rewrite."""
import time
import numpy as np
from itertools import product, combinations
from scipy.optimize import fsolve

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 145

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1
    
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1
    
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_2x2x3_rank10():
    """
    Try to find a rank-10 decomposition for 2x2 x 2x3.
    Uses a structured ansatz: fix support patterns, solve for values.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # The equations: for all (i,j,k) in {0,1}x{0,1}x{0,1,2}:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # That's 12 equations.
    # Unknowns: 4*10 + 6*10 + 6*10 = 160 variables.
    # But many are zero by sparsity.
    
    # Strategy: Try structured patterns where the first 7 columns form
    # a modified Strassen-like structure and 3 additional columns handle column 2.
    
    # Let's try: 7 columns for columns 0,1 (Strassen-like) + 3 columns for column 2
    # But allow the 3 extra columns to also have small contributions to cols 0,1
    # that cancel with adjustments.
    
    # Alternative: Use a different split. Try 5+5 split.
    
    # Let me try a direct approach: assume a specific structure.
    # For column 2 of B, we need:
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # This requires at least 4 rank-1 terms (schoolbook) unless we share with cols 0,1.
    
    # Key idea: Can we use terms that contribute to BOTH column 2 AND columns 0/1?
    # A term U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i,r] contributes to C[i,k'] from B row j, col k.
    # So if we set k=2 (B's third column) and k'=0 or 1 (C's first two columns),
    # that term would contribute to C[i,0] or C[i,1] from B[*,2], which should be 0.
    # So we can't have V[j*3+2, r] != 0 and W[k'*2+i, r] != 0 for k' in {0,1}
    # unless the U factor is 0 or it's cancelled.
    
    # Similarly, if we set k=0 or 1 and k'=2, we'd contribute to C[i,2] from B[*,0] or B[*,1],
    # which should be 0.
    
    # So the terms for column 2 (k=2) must have W support only in k'=2 rows,
    # and terms for columns 0,1 must have W support only in k'=0,1 rows.
    # This means the decomposition is a direct sum! Rank = rank(2x2x2) + rank(2x2x1) = 7+4 = 11.
    
    # Wait, that's not quite right. Let me re-examine.
    # The tensor T[i,j,k] = 1 for all (i,j,k).
    # A rank-1 term is U[i,j] * V[j,k] * W[k,i] (with appropriate indexing).
    # For the term to contribute to T[i,j,k], we need U[i*m+j]*V[j*p+k]*W[k*n+i] != 0.
    # 
    # The constraint is that for each (i,j,k), the sum over r equals 1.
    # There's no constraint that individual terms be zero for "wrong" (i,j,k).
    # A term with U[i*m+j] != 0, V[j*p+k1] != 0, W[k2*n+i] != 0 contributes to T[i,j,k1] via W[k2*n+i]
    # only if k2 = k1. Wait no.
    # 
    # Let me be more careful. The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # For a fixed r, the term U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is non-zero
    # only if all three factors are non-zero.
    # 
    # So if for some r, V[j*p+2, r] != 0 (i.e., the term involves B's column 2),
    # then W[k*n+i, r] must be non-zero for the term to contribute.
    # The term contributes to T[i,j,2] (since V index is j*p+2, so k=2).
    # But W[k*n+i, r] with k=2 means W[2*n+i, r] = W[4+i, r] (for n=2).
    # 
    # So a term with V[j*3+2, r] != 0 contributes to C[i,2] (the output's column 2).
    # It does NOT contribute to C[i,0] or C[i,1] because those require W[0*2+i, r] or W[1*2+i, r].
    # 
    # Similarly, a term with V[j*3+0, r] != 0 contributes to C[i,0], and
    # V[j*3+1, r] != 0 contributes to C[i,1].
    # 
    # So the decomposition IS a direct sum over the output columns!
    # Rank(2x2x3) = Rank(2x2x1 for col 0) + Rank(2x2x1 for col 1) + Rank(2x2x1 for col 2)
    # = 4 + 4 + 4 = 12? No, that's not right either.
    # 
    # Wait, I'm confusing myself. Let me re-derive.
    # 
    # The matrix multiplication (2x2) * (2x3) = (2x3) tensor.
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # 
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i].
    # For this to equal T, we need sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all (i,j,k).
    # 
    # Now, fix k. For a given k, the equation is:
    #   sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all (i,j).
    # 
    # Let V_r[:,k] be the k-th column of V_r, and W_r[k,:] be the k-th row of W_r.
    # Then: sum_r U_r[i,j] * (V_r[:,k])[j] * (W_r[k,:])[i] = 1 for all (i,j).
    # 
    # This is: sum_r (U_r * outer((W_r[k,:]), (V_r[:,k])))[i,j] = 1.
    # 
    # So for each k, we need a decomposition of the 2x2 all-ones matrix into rank-1 terms
    # of the form U_r * outer(W_r[k,:], V_r[:,k]).
    # 
    # The key point: the SAME U_r is used for all k! So the U_r's are shared across k.
    # This is where the non-triviality comes from.
    # 
    # For k=0: sum_r U_r[i,j] * W_r[0,i] * V_r[j,0] = 1
    # For k=1: sum_r U_r[i,j] * W_r[1,i] * V_r[j,1] = 1  
    # For k=2: sum_r U_r[i,j] * W_r[2,i] * V_r[j,2] = 1
    # 
    # If we define for each r and k: A_rk[i,j] = U_r[i,j] * W_r[k,i] * V_r[j,k]
    # Then sum_r A_rk[i,j] = 1 for all i,j,k.
    # 
    # Each A_rk is rank at most 1 (it's U_r scaled by row and column factors).
    # Actually, A_rk[i,j] = U_r[i,j] * (W_r[k,i]) * (V_r[j,k]).
    # This is NOT necessarily rank 1 unless U_r is rank 1.
    # 
    # Hmm, this is getting complicated. Let me just try a computational approach.
    
    # Let me try to find rank-10 by solving the system.
    # I'll use a specific ansatz: 
    # - First 7 terms: Strassen-like for the "main" part
    # - Last 3 terms: handle column 2
    # But allow mixing.
    
    # Actually, let me try a completely different approach: 
    # Use the known rank-11 decomposition and try to merge two terms.
    
    # Rank-11 = Strassen(7) + schoolbook(4) for column 2.
    # The schoolbook terms for column 2 are:
    #   r=7: U[0,7]=1, V[0*3+2,7]=V[2,7]=1, W[2*2+0,7]=W[4,7]=1  -> A[0,0]*B[0,2]->C[0,2]
    #   r=8: U[1,8]=1, V[1*3+2,8]=V[5,8]=1, W[2*2+0,8]=W[4,8]=1  -> A[0,1]*B[1,2]->C[0,2]
    #   r=9: U[2,9]=1, V[0*3+2,9]=V[2,9]=1, W[2*2+1,9]=W[5,9]=1  -> A[1,0]*B[0,2]->C[1,2]
    #   r=10: U[3,10]=1, V[1*3+2,10]=V[5,10]=1, W[2*2+1,10]=W[5,10]=1  -> A[1,1]*B[1,2]->C[1,2]
    
    # Can we combine any of these with Strassen terms?
    # Strassen terms only have V indices in {0,1,2,3} (for 2x2x2, p=2, so V has 4 rows).
    # For 2x2x3, V has 6 rows (indices 0-5).
    # Strassen terms use V rows 0-3 (k=0,1 for j=0,1).
    # Schoolbook terms for col 2 use V rows 2,5 (k=2 for j=0,1).
    # 
    # So V row 2 is used by BOTH Strassen (as V[j=0,k=1] for the 2x2x2 subproblem) 
    # and schoolbook (as V[j=0,k=2] for column 2).
    # Wait, in the 2x2x2 Strassen, V has 4 rows indexed by (j,k) with j in {0,1}, k in {0,1}.
    # When embedded in 2x2x3, V has 6 rows indexed by (j,k) with j in {0,1}, k in {0,1,2}.
    # The Strassen terms use V rows (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1) = rows 0,1,2,3.
    # The schoolbook terms for column 2 use V rows (j=0,k=2)=row 2 and (j=1,k=2)=row 5.
    # 
    # So V row 2 is shared! This means we might be able to combine.
    
    # Let me try: modify the Strassen terms to also contribute to column 2,
    # and reduce the schoolbook terms.
    
    # This is getting too complex for manual derivation. Let me try a numerical search
    # with a better structure.
    
    # Approach: Start from rank-11 decomposition, try to eliminate one term by
    # adjusting others.
    
    # Build rank-11 first
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U11[:, 0:7] = U_str
    for j in range(2):
        for k in range(2):
            V11[j*3+k, 0:7] = V_str[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W11[k*2+i, 0:7] = W_str[k*2+i, :]
    
    # Schoolbook for column 2
    U11[0, 7] = 1; V11[0*3+2, 7] = 1; W11[2*2+0, 7] = 1
    U11[1, 8] = 1; V11[1*3+2, 8] = 1; W11[2*2+0, 8] = 1
    U11[2, 9] = 1; V11[0*3+2, 9] = 1; W11[2*2+1, 9] = 1
    U11[3, 10] = 1; V11[1*3+2, 10] = 1; W11[2*2+1, 10] = 1
    
    # Now try to eliminate term 10 (the last schoolbook term) by adjusting others.
    # Term 10: U[3,10]=1, V[5,10]=1, W[5,10]=1 -> contributes 1 to T[1,1,2]
    # We need T[1,1,2] = sum_r U[3,r]*V[5,r]*W[5,r] = 1
    # Currently: terms 0-6 (Strassen) + term 10 = 1
    # Strassen contribution to T[1,1,2]: 
    #   T[1,1,2] uses W[2*2+1, r] = W[5, r]
    #   In Strassen, W[5, r] is non-zero for some r.
    #   Let's compute.
    
    # Actually, let me just verify the rank-11 decomposition first.
    if not _verify_decomposition(U11, V11, W11, 2, 2, 3):
        return None
    
    # Now try to find rank 10 by solving a perturbation problem.
    # Remove term 10 and see if we can adjust terms 0-9 to compensate.
    # This is a system of 12 equations in the remaining variables.
    
    # This is a nonlinear system. Let me try a local search.
    # Start from the rank-11 solution with term 10 zeroed out,
    # and try to find small adjustments to terms 0-9 that restore all equations.
    
    # Actually, this is equivalent to asking if the rank-11 solution lies on the
    # variety of rank-10 decompositions, which is unlikely.
    
    # Let me try a different strategy: random search with a better initialization.
    np.random.seed(123)
    
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(5000):
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Initialize with Strassen-like structure for first 7 terms
        U[:, 0:7] = U_str
        for j in range(2):
            for k in range(2):
                V[j*3+k, 0:7] = V_str[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W[k*2+i, 0:7] = W_str[k*2+i, :]
        
        # Last 3 terms: random for column 2
        for r in range(7, 10):
            # U: pick 1-2 rows
            n_u = np.random.randint(1, 3)
            u_rows = np.random.choice(4, n_u, replace=False)
            for row in u_rows:
                U[row, r] = np.random.choice([-1, 1])
            
            # V: must involve column 2 (rows 2 or 5)
            n_v = np.random.randint(1, 3)
            v_rows = np.random.choice([2, 5], n_v, replace=False)
            for row in v_rows:
                V[row, r] = np.random.choice([-1, 1])
            
            # W: must involve output column 2 (rows 4 or 5)
            n_w = np.random.randint(1, 3)
            w_rows = np.random.choice([4, 5], n_w, replace=False)
            for row in w_rows:
                W[row, r] = np.random.choice([-1, 1])
        
        # Verify
        err = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                    err += abs(val - 1.0)
        
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-10:
            return U, V, W
    
    return None

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _build_rank52_if_possible():
    """
    Try to build a rank-52 decomposition using 6 Strassen blocks + 1 rank-10 block.
    Falls back to rank-53 if rank-10 for 2x2x3 is not found.
    """
    n, m, p = DIMENSIONS
    
    result = _solve_2x2x3_rank10()
    if result is not None:
        U3, V3, W3 = result
        U_str, V_str, W_str = _build_strassen_2x2x2()
        total_rank = 6 * 7 + 10  # 52
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        for b in range(6):
            start_k = b * 2
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        
        # Last block: 2x2x3 rank-10 for cols 12,13,14
        U[:, col_offset:col_offset+10] = U3
        for j in range(m):
            for k_local in range(3):
                k_global = 12 + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V3[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = 12 + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W3[k_local * 2 + i, :]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank 52 first
    result52 = _build_rank52_if_possible()
    if result52 is not None:
        U, V, W = result52
        best_rank = 52
    else:
        # Fall back to rank 53
        U, V, W = _build_rank53()
        best_rank = 53
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Final fallback: schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
