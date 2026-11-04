"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Strategy: Structured local search with rank reduction.
- Start from a structured decomposition (Strassen blocks + schoolbook)
- Use local search to try to reduce rank by zeroing out terms
- Accept mutations only if tensor residual stays at zero
"""

import time
import random
import copy

def _tensor_residual(U, V, W, n, m, p):
    """Compute max |sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1| over all (i,j,k)."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    return _tensor_residual(U, V, W, n, m, p) < tol

def _strassen_2x2():
    """Returns 7-term Strassen decomposition for 2x2 * 2x2."""
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, -1]
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, -1]
    return [u1,u2,u3,u4,u5,u6,u7], [v1,v2,v3,v4,v5,v6,v7], [w1,w2,w3,w4,w5,w6,w7]

def _build_initial_4x4_4x5():
    """Build initial 4x4*4x5 decomposition using Strassen blocks + schoolbook. Rank 65."""
    n, m, p = 4, 4, 5
    u_2x2, v_2x2, w_2x2 = _strassen_2x2()
    
    block_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,-1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),
    ]
    
    def block_to_indices(bi):
        if bi == 0: return (0,1,0,1)
        if bi == 1: return (0,1,2,3)
        if bi == 2: return (2,3,0,1)
        if bi == 3: return (2,3,2,3)
    
    rank_44 = 49
    U_44 = [[0]*rank_44 for _ in range(16)]
    V_44 = [[0]*rank_44 for _ in range(16)]
    W_44 = [[0]*rank_44 for _ in range(16)]
    
    for t in range(7):
        a_blocks, b_blocks, c_blocks = block_terms[t]
        for s in range(7):
            c = t*7 + s
            u_vec = u_2x2[s]
            v_vec = v_2x2[s]
            w_vec = w_2x2[s]
            for b_idx, sign in a_blocks:
                r0,r1,c0,c1 = block_to_indices(b_idx)
                U_44[r0*4+c0][c] += sign*u_vec[0]
                U_44[r0*4+c1][c] += sign*u_vec[1]
                U_44[r1*4+c0][c] += sign*u_vec[2]
                U_44[r1*4+c1][c] += sign*u_vec[3]
            for b_idx, sign in b_blocks:
                r0,r1,c0,c1 = block_to_indices(b_idx)
                V_44[r0*4+c0][c] += sign*v_vec[0]
                V_44[r0*4+c1][c] += sign*v_vec[1]
                V_44[r1*4+c0][c] += sign*v_vec[2]
                V_44[r1*4+c1][c] += sign*v_vec[3]
            for b_idx, sign in c_blocks:
                r0,r1,c0,c1 = block_to_indices(b_idx)
                W_44[c0*4+r0][c] += sign*w_vec[0]
                W_44[c1*4+r0][c] += sign*w_vec[1]
                W_44[c0*4+r1][c] += sign*w_vec[2]
                W_44[c1*4+r1][c] += sign*w_vec[3]
    
    # Schoolbook for the 5th column
    rank_41 = 16
    U_41 = [[0]*rank_41 for _ in range(16)]
    V_41 = [[0]*rank_41 for _ in range(4)]
    W_41 = [[0]*rank_41 for _ in range(4)]
    c = 0
    for i in range(4):
        for j in range(4):
            U_41[i*4+j][c] = 1
            V_41[j][c] = 1
            W_41[i][c] = 1
            c += 1
    
    # Combine into 4x4*4x5
    rank = rank_44 + rank_41
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank_44):
            U[i][c] = U_44[i][c]
        for c in range(rank_41):
            U[i][rank_44+c] = U_41[i][c]
    
    for j in range(m):
        for k in range(4):
            for c in range(rank_44):
                V[j*p+k][c] = V_44[j*4+k][c]
        for c in range(rank_41):
            V[j*p+4][rank_44+c] = V_41[j][c]
    
    for k in range(4):
        for i in range(n):
            for c in range(rank_44):
                W[k*n+i][c] = W_44[k*4+i][c]
    for i in range(n):
        for c in range(rank_41):
            W[4*n+i][rank_44+c] = W_41[i][c]
    
    return U, V, W

def _try_zero_column(U, V, W, n, m, p, col):
    """Try to zero out column 'col' of U. Returns (success, new_U, new_V, new_W)."""
    # If we zero U[:,col], we need the remaining terms to still satisfy the tensor.
    # This only works if U[:,col] is "redundant" - i.e., its contribution can be absorbed.
    # Simple check: just zero it and verify.
    U_new = [row[:] for row in U]
    for i in range(len(U_new)):
        U_new[i][col] = 0
    if _is_valid(U_new, V, W, n, m, p):
        return True, U_new, V, W
    return False, None, None, None

def _local_search(U, V, W, n, m, p, time_budget=100, seed=42):
    """
    Local search: try to reduce rank by zeroing columns.
    Also try small perturbations to enable zeroing.
    """
    random.seed(seed)
    rank = len(U[0])
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    deadline = time.time() + time_budget
    best_err = 0.0
    
    # Phase 1: Try zeroing each column
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        current_rank = len(U[0])
        if current_rank <= best_rank:
            best_rank = current_rank
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        # Try to zero a random column
        if current_rank <= 1:
            break
        col = random.randint(0, current_rank - 1)
        success, U_new, V_new, W_new = _try_zero_column(U, V, W, n, m, p, col)
        if success:
            U, V, W = U_new, V_new, W_new
            rank = current_rank - 1
            # Remove the column from all matrices
            U = [row[:col] + row[col+1:] for row in U]
            V = [row[:col] + row[col+1:] for row in V]
            W = [row[:col] + row[col+1:] for row in W]
            continue
        
        # Phase 2: Try small perturbations to make a column zeroable
        if time.time() > deadline - 20:
            break
        
        # Pick a column to try to zero, and perturb another column to help
        target_col = random.randint(0, len(U[0])-1)
        help_col = random.randint(0, len(U[0])-1)
        if help_col == target_col:
            help_col = (help_col + 1) % len(U[0])
        
        # Small perturbation on help_col
        eps = random.choice([0.1, 0.01, 0.001])
        U_pert = [row[:] for row in U]
        V_pert = [row[:] for row in V]
        W_pert = [row[:] for row in W]
        
        for i in range(len(U_pert)):
            if random.random() < 0.3:
                U_pert[i][help_col] += eps * random.uniform(-1, 1)
        for j in range(len(V_pert)):
            if random.random() < 0.3:
                V_pert[j][help_col] += eps * random.uniform(-1, 1)
        for k in range(len(W_pert)):
            if random.random() < 0.3:
                W_pert[k][help_col] += eps * random.uniform(-1, 1)
        
        # Check if perturbation keeps residual small
        err = _tensor_residual(U_pert, V_pert, W_pert, n, m, p)
        if err < 1e-6:
            # Try zeroing target_col
            success, U_z, V_z, W_z = _try_zero_column(U_pert, V_pert, W_pert, n, m, p, target_col)
            if success:
                U = [row[:target_col] + row[target_col+1:] for row in U_z]
                V = [row[:target_col] + row[target_col+1:] for row in V_z]
                W = [row[:target_col] + row[target_col+1:] for row in W_z]
    
    return best_U, best_V, best_W, best_rank

def _schoolbook_4x4_4x5():
    """Schoolbook decomposition, rank 80."""
    n, m, p = 4, 4, 5
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses local search starting from a structured Strassen+schoolbook decomposition.
    """
    n, m, p = 4, 4, 5
    start_time = time.time()
    
    # Build initial structured decomposition
    U, V, W = _build_initial_4x4_4x5()
    initial_rank = len(U[0])
    
    # Verify initial
    if not _is_valid(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_4x4_4x5()
        return {"U": U, "V": V, "W": W}
    
    # Run local search with time budget
    time_budget = min(100, 140 - (time.time() - start_time))
    if time_budget > 10:
        U_ls, V_ls, W_ls, best_rank = _local_search(U, V, W, n, m, p, time_budget=time_budget)
        
        if best_rank < initial_rank and _is_valid(U_ls, V_ls, W_ls, n, m, p):
            U, V, W = U_ls, V_ls, W_ls
    
    # Final verification
    if not _is_valid(U, V, W, n, m, p):
        U, V, W = _schoolbook_4x4_4x5()
    
    return {"U": U, "V": V, "W": W}
