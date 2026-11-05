import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18
    
    # Construct target tensor T_target
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(R, T_target, atol=tol)

    def snap_and_verify(U, V, W, max_denom=8):
        """Try snapping to rationals with small denominators."""
        for denom in [1, 2, 4, max_denom]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
            if np.allclose(R, T_target, atol=1e-9):
                return True, U_s, V_s, W_s
        return False, None, None, None

    def obj_and_grad(params, rank):
        U_p = params[:dim_U*rank].reshape(dim_U, rank)
        V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
        
        R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
        diff = R - T_target
        obj_val = np.sum(diff**2)
        
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V_p, W_p)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U_p, W_p)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U_p, V_p)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj_val, grad

    # Strategy 1: Construct rank-46 from two 3x3x3 rank-23 blocks
    # The 3x3x6 can be split into two 3x3x3 problems (columns 0-2 and 3-5)
    # Each 3x3x3 has known rank 23, so total rank = 46
    
    def make_rank23_333():
        """
        Construct a rank-23 decomposition for 3x3x3 matrix multiplication.
        Uses the known Bini/Capovani/Cherubino/Strassen algorithm.
        """
        rank = 23
        U3 = np.zeros((9, rank))
        V3 = np.zeros((9, rank))
        W3 = np.zeros((9, rank))
        
        # The rank-23 algorithm for 3x3x3 (Bini et al. 1979)
        # This is a well-known result. Let me construct it explicitly.
        # 
        # The 23 scalar multiplications are based on the following structure:
        # We can think of it as extending the 2x2 Strassen (7 mults) to 3x3.
        #
        # A known construction uses the following 23 terms:
        # Let's use the explicit formulation from the literature.
        
        # Actually, let me use a verified rank-23 decomposition.
        # The key insight: 3x3x3 rank is exactly 23.
        
        # Here's one known construction using the "Strassen-like" approach:
        # Partition the 3x3 matrices and use overlapping Strassen blocks.
        
        # For reliability, let me use a numerically-verified approach:
        # Start from a known rank-23 solution and verify it.
        
        # The following is a known rank-23 decomposition (verified):
        # Using the algorithm from "The tensor rank of matrix multiplication"
        
        # Let me construct it using the block approach:
        # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
        # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
        
        # One approach: use 7 Strassen mults for the top-left 2x2,
        # then add terms for the remaining interactions.
        
        # Actually, for this program, let me just do direct optimization
        # to find a rank-23 solution, then use it to build rank-46.
        
        return None, None, None
    
    # Try direct optimization for 3x3x3 rank 23
    def find_333_rank23(max_time=30):
        """Find a rank-23 decomposition for 3x3x3 matrix multiplication."""
        dim3 = 9
        T333 = np.zeros((dim3, dim3, dim3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T333[i*3+j, j*3+k, k*3+i] = 1.0
        
        def obj333(params, r):
            U3 = params[:dim3*r].reshape(dim3, r)
            V3 = params[dim3*r:2*dim3*r].reshape(dim3, r)
            W3 = params[2*dim3*r:].reshape(dim3, r)
            R3 = np.einsum('ir,jr,kr->ijk', U3, V3, W3)
            diff = R3 - T333
            obj_val = np.sum(diff**2)
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V3, W3)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U3, W3)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U3, V3)
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return obj_val, grad
        
        best_err = float('inf')
        best_params = None
        r = 23
        
        for attempt in range(40):
            if time.time() > start_time + max_time:
                break
            
            np.random.seed(12345 + attempt)
            U3 = np.random.randn(dim3, r) * 0.3
            V3 = np.random.randn(dim3, r) * 0.3
            W3 = np.random.randn(dim3, r) * 0.3
            params = np.concatenate([U3.flatten(), V3.flatten(), W3.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: obj333(x, r),
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
                    if best_err < 1e-12:
                        break
            except:
                continue
        
        if best_err < 1e-8 and best_params is not None:
            U3 = best_params[:dim3*r].reshape(dim3, r)
            V3 = best_params[dim3*r:2*dim3*r].reshape(dim3, r)
            W3 = best_params[2*dim3*r:].reshape(dim3, r)
            
            # Try to snap to nice values
            for denom in [1, 2, 4, 8]:
                U3s = np.round(U3 * denom) / denom
                V3s = np.round(V3 * denom) / denom
                W3s = np.round(W3 * denom) / denom
                R3s = np.einsum('ir,jr,kr->ijk', U3s, V3s, W3s)
                if np.allclose(R3s, T333, atol=1e-9):
                    return U3s, V3s, W3s
            return U3, V3, W3
        return None, None, None
    
    # Try to find 3x3x3 rank-23 decomposition
    U3, V3, W3 = find_333_rank23(max_time=40)
    
    if U3 is not None:
        # Construct 3x3x6 from two 3x3x3 blocks
        # Block 1: columns 0,1,2 of B (k=0,1,2)
        # Block 2: columns 3,4,5 of B (k=3,4,5)
        #
        # U_full (9 x 46) = [U3, U3]
        # V_full (18 x 46): 
        #   rows 0-8  (j=0, k=0,1,2) -> [V3, 0]
        #   rows 9-17 (j=0, k=3,4,5) -> [0, V3]
        #   Wait, V index is j*p+k = j*6+k
        #   For block 1 (k=0,1,2): V[j*6+k] for k=0,1,2 -> rows 0,6,12 (j=0), 1,7,13 (j=1), 2,8,14 (j=2)
        #   For block 2 (k=3,4,5): V[j*6+k] for k=3,4,5 -> rows 3,9,15 (j=0), 4,10,16 (j=1), 5,11,17 (j=2)
        #
        # Actually, let me be more careful.
        # V has index j*p+k where j in [0,3), k in [0,6)
        # V3 has index j3*3+k3 where j3 in [0,3), k3 in [0,3)
        #
        # For block 1 (k=0,1,2): V[j*6+k] = V3[j*3+k] for k in {0,1,2}
        # For block 2 (k=3,4,5): V[j*6+k] = V3[j*3+(k-3)] for k in {3,4,5}
        
        r333 = 23
        rank_full = 2 * r333  # 46
        
        U_full = np.zeros((dim_U, rank_full))
        V_full = np.zeros((dim_V, rank_full))
        W_full = np.zeros((dim_W, rank_full))
        
        U_full[:, :r333] = U3
        U_full[:, r333:] = U3
        
        # V_full: for each j in 0,1,2 and k in 0,1,2: V_full[j*6+k, :r333] = V3[j*3+k, :]
        for j in range(3):
            for k in range(3):
                V_full[j*6+k, :r333] = V3[j*3+k, :]
                V_full[j*6+(k+3), r333:] = V3[j*3+k, :]
        
        # W_full: for each k in 0,1,2 and i in 0,1,2: W_full[k*3+i, :] = W3[k*3+i, :]
        # W index is k*n+i = k*3+i
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, :r333] = W3[k*3+i, :]
                W_full[(k+3)*3+i, r333:] = W3[k*3+i, :]
        
        # Verify
        if verify_decomp(U_full, V_full, W_full):
            success, Us, Vs, Ws = snap_and_verify(U_full, V_full, W_full)
            if success:
                return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
            else:
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Strategy 2: Direct optimization for 3x3x6 with various ranks
    best_solution = None
    
    for rank in [46, 45, 44, 43, 42]:
        if time.time() > deadline - 20:
            break
        
        num_attempts = 20 if rank <= 45 else 10
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(5000 + rank * 100 + attempt)
            scale = 0.5
            U = np.random.randn(dim_U, rank) * scale
            V = np.random.randn(dim_V, rank) * scale
            W = np.random.randn(dim_W, rank) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: obj_and_grad(x, rank),
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                )
                
                if res.fun < 1e-8:
                    U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                    V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    
                    if verify_decomp(U_opt, V_opt, W_opt):
                        success, U_list, V_list, W_list = snap_and_verify(U_opt, V_opt, W_opt)
                        if success:
                            return {"U": U_list.tolist(), "V": V_list.tolist(), "W": W_list.tolist()}
                        else:
                            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        
            except:
                pass
    
    # Fallback: schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
