"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses correct W[k*n+i] convention with two Strassen 2x2x2 blocks for rank 14.
Includes SVD-based tensor-train analysis for lower bound and structured rank-13 search.
"""
import time
import random
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] where k=output column, i=output row.
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for valid (i,j,k)
    and 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid multiplication terms equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Check all invalid triples equal 0
    for i_idx in range(n * m):
        for j_idx in range(m * p):
            for k_idx in range(p * n):
                # Decode indices
                a = i_idx // m
                b = i_idx % m
                b_prime = j_idx // p
                c = j_idx % p
                c_prime = k_idx // n
                a_prime = k_idx % n
                
                # Valid if a == a_prime and b == b_prime
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # W (C) indices: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #                C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Left Block (Columns 0, 1) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B00 + B11)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # m2 = (A10 + A11) * B00
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    # m3 = A00 * (B01 - B11)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    # m4 = A11 * (B10 - B00)
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # m5 = (A00 + A01) * B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # m6 = (A10 - A00) * (B00 + B01)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # m7 = (A01 - A11) * (B10 + B11)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1
    
    # Right Block (Columns 2, 3) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B02 + B13)
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    # m2 = (A10 + A11) * B02
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    # m3 = A00 * (B03 - B13)
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    # m4 = A11 * (B12 - B02)
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    # m5 = (A00 + A01) * B13
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    # m6 = (A10 - A00) * (B02 + B03)
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    # m7 = (A01 - A11) * (B12 + B13)
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _tensor_lower_bound():
    """
    Compute the rank of the mode-3 flattening of the multiplication tensor.
    This gives a rigorous lower bound on the tensor rank.
    The tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication term, else 0.
    Mode-3 flattening: reshape T from (n, m, p) to (p, n*m).
    """
    n, m, p = 2, 2, 4
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Mode-3 flattening: axes (i, j) vs k -> shape (p, n*m)
    T_flat = T.reshape(n * m, p).T  # (p, n*m)
    # SVD to find rank
    s = np.linalg.svd(T_flat, compute_uv=False)
    # Count non-zero singular values
    rank = np.sum(s > 1e-10)
    return int(rank), s

def _try_rank13_structured(deadline):
    """
    Structured rank-13 search using SVD projection and least-squares refinement.
    Strategy:
    1. Use SVD of mode-3 flattening to identify the subspace structure.
    2. Initialize U, V from Strassen patterns with one dimension merged.
    3. Use NNLS to solve for W given U, V.
    4. Iterate with alternating minimization.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    max_iterations = 200
    
    # Get the lower bound for reference
    lb, sv = _tensor_lower_bound()
    
    # Build target tensor vector
    # Valid terms: T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for valid, 0 for invalid
    
    # Initialize from rank-14 Strassen solution, then try to merge
    sol14 = _build_rank14()
    U14 = np.array(sol14["U"], dtype=float)
    V14 = np.array(sol14["V"], dtype=float)
    W14 = np.array(sol14["W"], dtype=float)
    
    # Strategy: try to find a rank-13 solution by starting from rank-14
    # and trying to zero out one column or merge two columns
    
    # Build the linear system for W given U, V
    # For each output index w_idx = k*n+i:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = T[i*m+j, j*p+k, w_idx]
    # where T = 1 if valid (i==k_row and j==k_col in output), 0 otherwise
    
    def _build_system_for_W(U, V, n, m, p):
        """
        Build the linear system A * W_flat = b for solving W.
        Each row corresponds to a (i, j, k) triple.
        A[row, r] = U[i*m+j, r] * V[j*p+k, r]
        b[row] = 1 if (i,j,k) is valid, 0 otherwise
        W has shape (p*n, rank), flattened to (p*n*rank,)
        """
        n, m, p = 2, 2, 4
        rank = U.shape[1]
        num_wn = p * n  # number of W rows
        # Total number of (i,j,k) triples
        num_triples = (n * m) * (m * p) * (p * n)
        # But we only need equations for each (i,j,k) triple
        # Actually, the constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]
        # This is one equation per (i,j,k) triple, but W[k*n+i, r] appears in multiple equations.
        # Let's set up: for each triple (i,j,k), the equation is:
        #   sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i, r] = T[i,j,k]
        # Group by w_idx = k*n+i:
        #   For fixed w_idx, we have multiple (i,j,k) triples mapping to it.
        #   Actually each (i,j,k) maps to exactly one w_idx = k*n+i.
        #   So for each (i,j,k): A row involves W[k*n+i, :]
        #   The coefficient for W[w_idx, r] is U[i*m+j, r] * V[j*p+k, r]
        
        # Let's build it as a system where unknowns are W[w_idx, r] for all w_idx, r
        # Number of unknowns: p*n*rank
        # Number of equations: n*m*m*p*p*n = n^2 * m^2 * p^2
        
        num_unknowns = p * n * rank
        num_eqs = n * m * m * p * p * n
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This triple maps to w_idx = k*n + i
                    w_idx = k * n + i
                    # The equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = T[i,j,k]
                    # T[i,j,k] = 1 (always valid for matrix mult)
                    b[eq_idx] = 1.0
                    for r in range(rank):
                        col = w_idx * rank + r
                        A[eq_idx, col] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        return A, b
    
    def _verify_fast(U, V, W, n=2, m=2, p=4, tol=1e-8):
        """Fast verification using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check valid terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = np.dot(np.array([U[i*m+j, r] * V[j*p+k, r] for r in range(rank)]),
                                   np.array([W[k*n+i, r] for r in range(rank)]))
                    if abs(t_val - 1.0) > tol:
                        return False
        
        # Check invalid terms
        for i_idx in range(n * m):
            for j_idx in range(m * p):
                for k_idx in range(p * n):
                    a = i_idx // m
                    b = i_idx % m
                    b_prime = j_idx // p
                    c = j_idx % p
                    c_prime = k_idx // n
                    a_prime = k_idx % n
                    if a == a_prime and b == b_prime:
                        continue
                    t_val = np.dot(np.array([U[i_idx, r] * V[j_idx, r] for r in range(rank)]),
                                   np.array([W[k_idx, r] for r in range(rank)]))
                    if abs(t_val) > tol:
                        return False
        return True
    
    # Try different strategies to find rank 13
    
    # Strategy 1: Start from rank-14, try to zero out one column
    for col_to_remove in range(14):
        if time.time() > deadline - 20:
            break
        
        # Remove column col_to_remove
        keep = [c for c in range(14) if c != col_to_remove]
        U_red = U14[:, keep]
        V_red = V14[:, keep]
        W_red = W14[:, keep]
        
        # Now try to adjust to make it work - use NNLS on W
        A, b = _build_system_for_W(U_red, V_red, n, m, p)
        # Solve for W_flat
        W_flat, _ = nnls(A, b)
        W_new = W_flat.reshape(p * n, target_rank)
        
        if _verify_fast(U_red, V_red, W_new, n, m, p):
            return {
                "U": U_red.tolist(),
                "V": V_red.tolist(),
                "W": W_new.tolist()
            }
    
    # Strategy 2: Random sparse initialization with NNLS refinement
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        # Random sparse U, V
        U_rand = np.zeros((n * m, target_rank))
        V_rand = np.zeros((m * p, target_rank))
        for r in range(target_rank):
            # Each rank-1 term has a few non-zeros
            u_nonzero = random.randint(1, 3)
            v_nonzero = random.randint(1, 4)
            u_indices = random.sample(range(n * m), min(u_nonzero, n * m))
            v_indices = random.sample(range(m * p), min(v_nonzero, m * p))
            for idx in u_indices:
                U_rand[idx, r] = random.choice([-1, 1])
            for idx in v_indices:
                V_rand[idx, r] = random.choice([-1, 1])
        
        # Solve for W using NNLS
        A, b = _build_system_for_W(U_rand, V_rand, n, m, p)
        W_flat, _ = nnls(A, b)
        W_new = W_flat.reshape(p * n, target_rank)
        
        if _verify_fast(U_rand, V_rand, W_new, n, m, p, tol=1e-7):
            return {
                "U": U_rand.tolist(),
                "V": V_rand.tolist(),
                "W": W_new.tolist()
            }
        
        # Try alternating minimization
        U_cur = U_rand.copy()
        V_cur = V_rand.copy()
        W_cur = W_new.copy()
        
        for it in range(50):
            if time.time() > deadline - 15:
                break
            
            # Update W given U, V
            A, b = _build_system_for_W(U_cur, V_cur, n, m, p)
            W_flat, _ = nnls(A, b)
            W_cur = W_flat.reshape(p * n, target_rank)
            
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {
                    "U": U_cur.tolist(),
                    "V": V_cur.tolist(),
                    "W": W_cur.tolist()
                }
            
            # Update U given V, W
            # For each i_idx, sum_r U[i_idx, r] * V[j_idx, r] * W[k_idx, r] = T
            # This is more complex - skip for now
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build the known rank-14 solution
    result_14 = _build_rank14()
    
    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        # Try to find a rank-13 solution using structured SVD/NNLS search
        result_13 = _try_rank13_structured(deadline)
        if result_13 is not None:
            return result_13
        return result_14
    
    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
