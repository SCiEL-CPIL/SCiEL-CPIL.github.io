import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48

    # Target: 2x4x6
    # We will try to find a rank 24 solution by solving two 2x2x6 sub-problems of rank 12.
    # If that fails, we return schoolbook.

    # Sub-problem: 2x2x6
    n_sub = 2
    m_sub = 2
    p_sub = 6
    target_rank_sub = 12
    
    # Build T_sub
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

    def compute_tensor(U_mat, V_mat, W_mat):
        # U: (4, R), V: (12, R), W: (12, R)
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)

    def objective_and_grads(U_mat, V_mat, W_mat, T_target):
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T_target
        obj = np.sum(diff**2)
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return obj, 2 * dU, 2 * dV, 2 * dW

    def normalize_U(U_mat):
        # Normalize columns of U to unit norm
        norms = np.linalg.norm(U_mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        U_mat = U_mat / norms
        return U_mat

    def try_rank_sub_optimized(rank, max_restarts=50):
        """
        Try to find a rank decomposition for the 2x2x6 sub-problem.
        Uses symmetry breaking: U columns are normalized to unit norm.
        """
        best_obj = np.inf
        best_U = None
        best_V = None
        best_W = None

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            np.random.seed(restart * 1000 + 42)
            
            # Initialize
            U_mat = np.random.randn(n_sub * m_sub, rank)
            V_mat = np.random.randn(m_sub * p_sub, rank)
            W_mat = np.random.randn(p_sub * n_sub, rank)
            
            # Scale initialization to avoid huge/small gradients
            U_mat = normalize_U(U_mat)
            V_mat *= 0.5
            W_mat *= 0.5

            lr = 0.01
            obj, _, _, _ = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
            best_obj_local = obj
            stall_count = 0
            max_iter = 3000

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
                
                # Adaptive step size based on gradient magnitude
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm < 1e-12:
                    break
                    
                # Step size decay or adaptive
                step_scale = lr / (1 + grad_norm)
                
                U_new = U_mat - step_scale * dU
                V_new = V_mat - step_scale * dV
                W_new = W_mat - step_scale * dW
                
                # Enforce symmetry breaking: Normalize U
                U_new = normalize_U(U_new)
                
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_new, T_sub)
                
                if obj_new < best_obj_local - 1e-12:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj_local = obj_new
                    stall_count = 0
                    # Increase lr slightly if improving
                    if lr < 0.1:
                        lr *= 1.01
                else:
                    stall_count += 1
                    lr *= 0.9
                    if lr < 1e-6 or stall_count > 100:
                        break
                
                if best_obj_local < 1e-8:
                    break

            if best_obj_local < 1e-4:
                # Polish phase
                lr = 1e-4
                for _ in range(1000):
                    if time.time() > deadline:
                        break
                    obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
                    grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                    step_scale = lr / (1 + grad_norm)
                    U_mat = normalize_U(U_mat - step_scale * dU)
                    V_mat -= step_scale * dV
                    W_mat -= step_scale * dW
                    lr *= 0.99
                    if obj < 1e-10:
                        break
                
                final_obj = objective_and_grads(U_mat, V_mat, W_mat, T_sub)[0]
                if final_obj < 1e-4:
                    if final_obj < best_obj:
                        best_obj = final_obj
                        best_U = U_mat.copy()
                        best_V = V_mat.copy()
                        best_W = W_mat.copy()
                        # If we found a very good solution, we might break early
                        # but let's try a few more restarts to ensure robustness
                        # or just break if it's extremely good
                        if final_obj < 1e-6:
                            break

        if best_obj < 1e-4:
            return best_U, best_V, best_W, True
        return None, None, None, False

    # Try to find rank 12 solution for sub-problem
    U_sub, V_sub, W_sub, found = try_rank_sub_optimized(target_rank_sub, max_restarts=60)

    if found:
        # Construct full solution
        full_rank = 2 * target_rank_sub
        U_out = np.zeros((n * m, full_rank))
        V_out = np.zeros((m * p, full_rank))
        W_out = np.zeros((p * n, full_rank))

        # Mapping as in original code
        # Block 1: j=0,1
        # U_full rows: 0,1 (i=0, j=0,1) and 4,5 (i=1, j=0,1)
        # U_sub rows: 0,1 (i=0, j=0,1) and 2,3 (i=1, j=0,1)
        
        for r in range(target_rank_sub):
            # Block 1
            U_out[0, r] = U_sub[0, r]
            U_out[1, r] = U_sub[1, r]
            U_out[4, r] = U_sub[2, r]
            U_out[5, r] = U_sub[3, r]
            
            # Block 2: j=2,3
            # U_full rows: 2,3 (i=0, j=2,3) and 6,7 (i=1, j=2,3)
            # U_sub rows: 0,1 (i=0, j=0,1) and 2,3 (i=1, j=0,1)
            # Note: The tensor structure is identical for block 2, just shifted indices.
            # A[i, 2+j'] and B[2+j', k].
            # This is isomorphic to A[i, j'] and B[j', k].
            # So we reuse the same U_sub, V_sub, W_sub.
            
            U_out[2, r + target_rank_sub] = U_sub[0, r]
            U_out[3, r + target_rank_sub] = U_sub[1, r]
            U_out[6, r + target_rank_sub] = U_sub[2, r]
            U_out[7, r + target_rank_sub] = U_sub[3, r]
            
            # V mapping
            # V_full rows 0..11 correspond to j=0,1.
            # V_sub rows 0..11 correspond to j=0,1.
            V_out[0:12, r] = V_sub[:, r]
            
            # V_full rows 12..23 correspond to j=2,3.
            V_out[12:24, r + target_rank_sub] = V_sub[:, r]
            
            # W mapping
            # W_full rows correspond to k, i.
            # W_sub rows correspond to k, i.
            # Same indices.
            W_out[:, r] = W_sub[:, r]
            W_out[:, r + target_rank_sub] = W_sub[:, r]

        # Convert to list of lists
        U_list = [[0.0] * full_rank for _ in range(n * m)]
        V_list = [[0.0] * full_rank for _ in range(m * p)]
        W_list = [[0.0] * full_rank for _ in range(p * n)]
        
        for a in range(n * m):
            for r in range(full_rank):
                U_list[a][r] = float(U_out[a, r])
        for b in range(m * p):
            for r in range(full_rank):
                V_list[b][r] = float(V_out[b, r])
        for c in range(p * n):
            for r in range(full_rank):
                W_list[c][r] = float(W_out[c, r])

        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Schoolbook
    U_out = np.zeros((n * m, schoolbook_rank))
    V_out = np.zeros((m * p, schoolbook_rank))
    W_out = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_out[i * m + j, col] = 1.0
                V_out[j * p + k, col] = 1.0
                W_out[k * n + i, col] = 1.0
                col += 1

    U_list = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_list = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_list = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(schoolbook_rank):
            U_list[a][r] = float(U_out[a, r])
    for b in range(m * p):
        for r in range(schoolbook_rank):
            V_list[b][r] = float(V_out[b, r])
    for c in range(p * n):
        for r in range(schoolbook_rank):
            W_list[c][r] = float(W_out[c, r])

    return {"U": U_list, "V": V_list, "W": W_list}
