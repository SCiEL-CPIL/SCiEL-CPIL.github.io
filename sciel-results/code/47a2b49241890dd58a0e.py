"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 63
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition_np(U, V, W, n_, m_, p_, tol=1e-9):
        """
        Fast verification using NumPy.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Checks sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        """
        if len(U) == 0:
            return False
        R = len(U[0])
        if len(V[0]) != R or len(W[0]) != R:
            return False
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_3d = U_np.reshape(n_, m_, R)
        V_3d = V_np.reshape(m_, p_, R)
        W_3d = W_np.reshape(p_, n_, R)
        
        # We need to check for all i, j, k:
        # sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r] == 1
        
        # This is a tensor contraction.
        # We can iterate over i, j, k as there are only n*m*p = 63 checks.
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    # Extract vectors
                    u_vec = U_3d[i, j, :]
                    v_vec = V_3d[j, k, :]
                    w_vec = W_3d[k, i, :]
                    
                    # Compute sum of element-wise products
                    val = np.sum(u_vec * v_vec * w_vec)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_result(U, V, W):
        """Convert to list-of-lists format with clean values."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        def clean_val(x):
            # Snap to half-integers
            r = round(x * 2) / 2
            if abs(x - r) < 1e-8:
                return r
            r = round(x)
            if abs(x - r) < 1e-8:
                return float(r)
            return x
        
        U_list = [[clean_val(x) for x in row] for row in U_list]
        V_list = [[clean_val(x) for x in row] for row in V_list]
        W_list = [[clean_val(x) for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}

    def search_subproblem(n_, m_, p_, target_rank, max_trials=1000):
        """
        Search for a decomposition of rank < target_rank for n_ x m_ x p_.
        Returns (U, V, W) if found, else None.
        """
        # Schoolbook for this subproblem
        sb_rank = n_ * m_ * p_
        U_sb = np.zeros((n_*m_, sb_rank))
        V_sb = np.zeros((m_*p_, sb_rank))
        W_sb = np.zeros((p_*n_, sb_rank))
        
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U_sb[i*m_+j, col] = 1
                    V_sb[j*p_+k, col] = 1
                    W_sb[k*n_+i, col] = 1
                    col += 1
        
        best_sub = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
        best_sub_rank = sb_rank
        
        # Try to find lower rank
        # Use random search with small integer coefficients
        coeff_set = [-1, 0, 1]
        
        for rank in range(min(target_rank - 1, sb_rank - 1), n_*m_ - 1, -1):
            if time.time() > deadline - 10:
                break
            
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                
                # Random sparse initialization
                U = np.zeros((n_*m_, rank))
                V = np.zeros((m_*p_, rank))
                W = np.zeros((p_*n_, rank))
                
                for r in range(rank):
                    # Ensure each term contributes to some (i,j,k)
                    # Randomly pick i, j, k
                    i = random.randint(0, n_-1)
                    j = random.randint(0, m_-1)
                    k = random.randint(0, p_-1)
                    
                    # Assign values
                    u_val = random.choice(coeff_set)
                    v_val = random.choice(coeff_set)
                    w_val = random.choice(coeff_set)
                    
                    if u_val == 0 or v_val == 0 or w_val == 0:
                        continue
                    
                    U[i*m_+j, r] = u_val
                    V[j*p_+k, r] = v_val
                    W[k*n_+i, r] = w_val
                
                if verify_decomposition_np(U.tolist(), V.tolist(), W.tolist(), n_, m_, p_):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return best_sub

    # Strategy: Split 7 into 3 + 2 + 2
    # Search for 3x3x3 and 3x3x2 decompositions
    
    # 1. Search for 3x3x3
    U_3, V_3, W_3 = search_subproblem(3, 3, 3, 27, max_trials=500)
    rank_3 = len(U_3[0])
    
    # 2. Search for 3x3x2
    U_2, V_2, W_2 = search_subproblem(3, 3, 2, 18, max_trials=1000)
    rank_2 = len(U_2[0])
    
    # Combine if we found improvements
    # Note: The indices in U, V, W must be consistent with the global indices.
    # For 3x3x3 part (cols 0-2 of B):
    # U rows: i*3+j (0-8)
    # V rows: j*7+k (j=0..2, k=0..2) -> indices 0..5, 7..12, 14..19
    # W rows: k*3+i (k=0..2, i=0..2) -> indices 0..8
    
    # For 3x3x2 part (cols 3-4 of B):
    # U rows: i*3+j (0-8)
    # V rows: j*7+k (j=0..2, k=3,4) -> indices 3..5, 10..12, 17..19
    # W rows: k*3+i (k=3,4, i=0..2) -> indices 9..14
    
    # We need to map the local indices to global indices.
    
    def combine_decompositions(parts):
        """
        parts: list of (U, V, W, offset_k, offset_k_end)
        """
        total_rank = sum(len(p[0][0]) for p in parts)
        U_final = [[0] * total_rank for _ in range(9)]
        V_final = [[0] * total_rank for _ in range(21)]
        W_final = [[0] * total_rank for _ in range(21)]
        
        col_offset = 0
        for U, V, W, k_start, k_end in parts:
            r = len(U[0])
            for i in range(9):
                U_final[i][col_offset:col_offset+r] = U[i]
            
            # Map V rows
            # Local V row index: j * 2 + k_local (for 3x3x2)
            # Global V row index: j * 7 + k_global
            for j in range(3):
                for k_local in range(k_end - k_start):
                    k_global = k_start + k_local
                    v_local_row = j * (k_end - k_start) + k_local
                    v_global_row = j * 7 + k_global
                    V_final[v_global_row][col_offset:col_offset+r] = V[v_local_row]
            
            # Map W rows
            # Local W row index: k_local * 3 + i
            # Global W row index: k_global * 3 + i
            for k_local in range(k_end - k_start):
                k_global = k_start + k_local
                for i in range(3):
                    w_local_row = k_local * 3 + i
                    w_global_row = k_global * 3 + i
                    W_final[w_global_row][col_offset:col_offset+r] = W[w_local_row]
            
            col_offset += r
        
        return U_final, V_final, W_final

    # Part 1: 3x3x3 (k=0,1,2)
    part1 = (U_3, V_3, W_3, 0, 3)
    # Part 2: 3x3x2 (k=3,4)
    part2 = (U_2, V_2, W_2, 3, 5)
    # Part 3: 3x3x2 (k=5,6)
    part3 = (U_2, V_2, W_2, 5, 7)
    
    U_comb, V_comb, W_comb = combine_decompositions([part1, part2, part3])
    
    if verify_decomposition_np(U_comb, V_comb, W_comb, 3, 3, 7):
        result = make_result(U_comb, V_comb, W_comb)
        best_result = result
        best_rank = len(U_comb[0])
    
    return best_result

# EVOLVE-BLOCK-END
