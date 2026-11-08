import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] == delta_{i,i'} * delta_{k,k'}
    for the multiplication tensor. The j index is summed implicitly in the contraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        total = 0
                        for r in range(rank):
                            total += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        expected = 1 if (i == ip and k == kp) else 0
                        if abs(total - expected) > 1e-9:
                            return False
    return True


def _try_rank_decomposition(n, m, p, target_rank, timeout=100):
    """Try to find a rank-target_rank decomposition using iterative optimization."""
    start = time.time()
    deadline = start + timeout
    
    # The tensor T[i,j,k, ip, kp] = delta_{i,ip} * delta_{k,kp}
    # We want: sum_r U_r[i,j] * V_r[j,k] * W_r[kp,ip] = T[i,j,k,ip,kp]
    
    # Reshape tensor to a matrix for SVD-based approach
    # Flatten: input indices (i,j,k) -> 2*4*8=64, output indices (ip,kp) -> 2*8=16
    # But the tensor is 5-mode, so we need a different approach.
    
    # Alternative: use the fact that the multiplication tensor has special structure.
    # For 2x4 x 4x8, let's try a block structure.
    
    # Try rank 48, 44, 40, 36, 32, 28, 24, 20, 16
    # The trivial rank is 64 (one per nonzero entry), Strassen gives 56.
    
    # Let's use a more systematic approach: set up as a linear system
    # For each r, U_r, V_r, W_r are vectors. The constraint is:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[kp*n+ip] = delta_{i,ip} * delta_{k,kp}
    
    # This is a multilinear constraint. Let's try a greedy/iterative approach.
    
    # First, let's see if we can do better with a smarter block decomposition.
    # 2x4 x 4x8: think of it as (2x2 x 2x8) * 2 blocks in j, or (2x4 x 4x2) * 4 blocks in k.
    
    # Try: split j into 2 groups of 2, k into 4 groups of 2
    # Each subproblem: 2x2 x 2x2 -> rank 7 (Strassen)
    # Total: 2*4 = 8 blocks * 7 = 56 (same as current)
    
    # Alternative: use the fact that n=2 is small.
    # The output is 2x8 = 16 entries.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can try to exploit the structure more.
    
    # Let's try a direct search for rank < 56 using random initialization + optimization
    best_result = None
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        np.random.seed(attempt * 1000 + 42)
        U_mat = np.random.randn(n * m, target_rank)
        V_mat = np.random.randn(m * p, target_rank)
        W_mat = np.random.randn(p * n, target_rank)
        
        # Normalize to prevent overflow
        U_mat /= np.linalg.norm(U_mat)
        V_mat /= np.linalg.norm(V_mat)
        W_mat /= np.linalg.norm(W_mat)
        
        # Iterative refinement: fix two, solve for the third
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
                
            # Solve for U given V, W
            # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r] = delta_{i,ip}*delta_{k,kp}
            # This is: for fixed (i,j), the vector U[i*m+j, :] must satisfy:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[kp*n+ip, r]) = delta_{i,ip}*delta_{k,kp} for all k,ip,kp
            # Let's define: for fixed (i,j), M_r[k, ip, kp] = V[j*p+k, r] * W[kp*n+ip, r]
            # We need: sum_r U[i*m+j, r] * M_r = T_ij where T_ij[k, ip, kp] = delta_{i,ip}*delta_{k,kp}
            
            # This is a linear system in U[i*m+j, :]
            # Let's set up: for each (k, ip, kp), we have one equation
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r] = delta_{i,ip}*delta_{k,kp}
            
            # There are p*n*n = 8*2*2 = 32 equations for each (i,j), and target_rank unknowns
            # If target_rank <= 32, we can solve directly
            
            if target_rank <= p * n * n:
                # Build the linear system for each (i,j)
                for i in range(n):
                    for j in range(m):
                        # A[r, eq_idx] = V[j*p+k, r] * W[kp*n+ip, r]
                        # b[eq_idx] = delta_{i,ip}*delta_{k,kp}
                        A_sys = np.zeros((p * n * n, target_rank))
                        b_vec = np.zeros(p * n * n)
                        eq_idx = 0
                        for k in range(p):
                            for ip in range(n):
                                for kp in range(p):
                                    for r in range(target_rank):
                                        A_sys[eq_idx, r] = V_mat[j * p + k, r] * W_mat[kp * n + ip, r]
                                    b_vec[eq_idx] = 1.0 if (i == ip and k == kp) else 0.0
                                    eq_idx += 1
                        # Solve: A_sys @ U[i*m+j, :] = b_vec
                        try:
                            U_mat[i * m + j, :] = np.linalg.lstsq(A_sys, b_vec, rcond=None)[0]
                        except:
                            pass
            
            # Similarly solve for V and W
            # For V: fix U, W
            # For each (j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r] = delta_{i,ip}*delta_{k,kp}
            # For fixed (j,k): sum_r V[j*p+k, r] * (U[i*m+j, r] * W[kp*n+ip, r]) = delta_{i,ip}*delta_{k,kp}
            # Equations: n*n*p = 2*2*8 = 32, unknowns: target_rank
            
            if target_rank <= n * n * p:
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n * n * p, target_rank))
                        b_vec = np.zeros(n * n * p)
                        eq_idx = 0
                        for i in range(n):
                            for ip in range(n):
                                for kp in range(p):
                                    for r in range(target_rank):
                                        A_sys[eq_idx, r] = U_mat[i * m + j, r] * W_mat[kp * n + ip, r]
                                    b_vec[eq_idx] = 1.0 if (i == ip and k == kp) else 0.0
                                    eq_idx += 1
                        try:
                            V_mat[j * p + k, :] = np.linalg.lstsq(A_sys, b_vec, rcond=None)[0]
                        except:
                            pass
            
            # For W: fix U, V
            # For each (kp, ip): sum_r U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r] = delta_{i,ip}*delta_{k,kp}
            # For fixed (kp, ip): sum_r W[kp*n+ip, r] * (U[i*m+j, r] * V[j*p+k, r]) = delta_{i,ip}*delta_{k,kp}
            # Equations: n*m*p = 2*4*8 = 64, unknowns: target_rank
            
            if target_rank <= n * m * p:
                for kp in range(p):
                    for ip in range(n):
                        A_sys = np.zeros((n * m * p, target_rank))
                        b_vec = np.zeros(n * m * p)
                        eq_idx = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    for r in range(target_rank):
                                        A_sys[eq_idx, r] = U_mat[i * m + j, r] * V_mat[j * p + k, r]
                                    b_vec[eq_idx] = 1.0 if (i == ip and k == kp) else 0.0
                                    eq_idx += 1
                        try:
                            W_mat[kp * n + ip, :] = np.linalg.lstsq(A_sys, b_vec, rcond=None)[0]
                        except:
                            pass
        
        # Check if we found a valid decomposition
        if time.time() > deadline - 2:
            break
            
        # Convert to list format for verification
        U_list = [[U_mat[i, r] for r in range(target_rank)] for i in range(n * m)]
        V_list = [[V_mat[i, r] for r in range(target_rank)] for i in range(m * p)]
        W_list = [[W_mat[i, r] for r in range(target_rank)] for i in range(p * n)]
        
        # Verify
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            return best_result
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with the known rank-56 solution
    best = _build_rank56()
    best_rank = 56
    
    # Try to find lower rank decompositions
    # Theoretically, the rank could be as low as max(n*m, m*p, p*n) = 32, but likely higher
    # Let's try ranks: 52, 48, 44, 40, 36, 32
    
    for target_rank in [52, 48, 44, 40, 36, 32]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, timeout=min(20, deadline - time.time() - 5))
        if result is not None:
            best = result
            best_rank = target_rank
            break  # Found a valid lower-rank decomposition
    
    return best

