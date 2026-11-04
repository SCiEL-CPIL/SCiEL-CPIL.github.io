import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 115.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_2x4x4_strassen():
    """Build 2x4x4 via Strassen on the 2x2 block structure. Rank 28."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    
    # U has shape (2*4, 28) = (8, 28)
    # Indexing: U[i, r] where i is in [0,8) representing row i of the 2x4 matrix
    # r is in [0,28) representing the 28 rank terms
    # The 2x4x4 multiplication: C[i,k] = sum_j A[i,j]*B[j,k] where A is 2x4, B is 4x4
    # We can view this as: split A into two 1x4 rows, B into 2x2 blocks of 2x2
    # Actually, let's use the tensor decomposition approach directly.
    
    # For 2x4x4: n=2, m=4, p=4
    # T[i,j,k] = 1 for all i,j,k
    # We can decompose this using the Kronecker structure:
    # View 2x4 as (2x1) x (1x4), and 4x4 as (2x2) x (2x2)
    # Use Strassen on the 2x2x2 part and schoolbook on the 1x4 part
    
    # Alternative: use the fact that 2x4x4 can be seen as 4 independent 1x4x4 multiplications
    # but that gives rank 16. Let's use a better decomposition.
    
    # Actually, for 2x4x4, we can use the following:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Split j into two groups: j=0,1 and j=2,3
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Each sum is a 2x2x2 multiplication (A is 2x2, B is 2x2)
    # So rank = 2 * 7 = 14
    
    U = np.zeros((8, 14), dtype=np.int16)
    V = np.zeros((16, 14), dtype=np.int16)
    W = np.zeros((8, 14), dtype=np.int16)
    
    # For the first group (j=0,1):
    # A_block = A[:, 0:2] (2x2), B_block = B[0:2, :] (2x4)
    # Wait, B is 4x4, so B[0:2, :] is 2x4
    # This is a 2x2x4 multiplication, not 2x2x2
    
    # Let me reconsider. For 2x4x4:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # We can view this as two 2x2x4 multiplications:
    # C1[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # C2[i,k] = sum_{j=2}^{3} A[i,j] * B[j,k]
    # C = C1 + C2
    
    # Each 2x2x4 can be decomposed using Strassen-like methods
    # For 2x2x4: n=2, m=2, p=4
    # This can be seen as two 2x2x2 multiplications
    # C1[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # Split k into two groups: k=0,1 and k=2,3
    # C1[i,0:2] = A[:,0:2] @ B[0:2,0:2] (2x2x2)
    # C1[i,2:4] = A[:,0:2] @ B[0:2,2:4] (2x2x2)
    # So rank for 2x2x4 = 2 * 7 = 14
    
    # Total rank for 2x4x4 = 2 * 14 = 28
    
    # Let's build this properly
    # For group 1 (j=0,1):
    #   For k=0,1: use Strassen on A[:,0:2] and B[0:2,0:2]
    #   For k=2,3: use Strassen on A[:,0:2] and B[0:2,2:4]
    # For group 2 (j=2,3):
    #   For k=0,1: use Strassen on A[:,2:4] and B[2:4,0:2]
    #   For k=2,3: use Strassen on A[:,2:4] and B[2:4,2:4]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    for group in range(2):  # j group: 0 or 1
        for kgroup in range(2):  # k group: 0 or 1
            base_rank = (group * 2 + kgroup) * 7
            for r in range(7):
                col = base_rank + r
                # U: A[i, j] where i in [0,2), j in [group*2, group*2+2)
                for i in range(2):
                    U[i, col] = U2[i, r]
                # V: B[j, k] where j in [group*2, group*2+2), k in [kgroup*2, kgroup*2+2)
                for j in range(2):
                    for k in range(2):
                        V[(group * 2 + j) * 4 + (kgroup * 2 + k), col] = V2[j * 2 + k, r]
                # W: C[i, k] where i in [0,2), k in [kgroup*2, kgroup*2+2)
                for i in range(2):
                    for k in range(2):
                        W[(kgroup * 2 + k) * 2 + i, col] = W2[k * 2 + i, r]
    
    return U, V, W

def _build_4x4x8_slice_and_dice():
    """Build 4x4x8 via slice-and-dice: split into two 2x4x4 blocks. Rank 56."""
    U2, V2, W2 = _build_2x4x4_strassen()
    total_rank = 56
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # For block 0: rows 0,1 of A
    # For block 1: rows 2,3 of A
    for block in range(2):
        col_start = block * 28
        col_end = (block + 1) * 28
        # U: A[i, j] where i in [block*2, block*2+2), j in [0,4)
        for i in range(2):
            for j in range(4):
                U[(block * 2 + i) * 4 + j, col_start:col_end] = U2[i * 4 + j, :]
        # V: B[j, k] where j in [0,4), k in [0,8)
        # The 2x4x4 block uses B[0:4, 0:4] for block 0 and B[0:4, 4:8] for block 1
        # Wait, that's not right. Let me reconsider.
        
        # Actually, for 4x4x8: C[i,k] = sum_j A[i,j]*B[j,k]
        # If we split i into two groups of 2, we get two 2x4x8 multiplications
        # Not 2x4x4. Let me reconsider the approach.
        
        # Alternative: split k into two groups of 4
        # C[:, 0:4] = A @ B[:, 0:4] (4x4x4)
        # C[:, 4:8] = A @ B[:, 4:8] (4x4x4)
        # Each 4x4x4 has rank 49 using Strassen-Kronecker
        # Total rank = 98, which is what we already have.
        
        # The slice-and-dice approach from the directive suggests:
        # "treating the 4x4x8 multiplication as a sum of four independent 2x4x4 multiplications
        # along the input row dimension"
        # This doesn't quite make sense. Let me try a different interpretation.
        
        # Perhaps: split A into 2x2 blocks and use Strassen on each
        # A is 4x4, so split into four 2x2 blocks: A00, A01, A10, A11
        # B is 4x8, so split into four 2x4 blocks: B00, B01, B10, B11
        # C is 4x8, so split into four 2x4 blocks: C00, C01, C10, C11
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # Each 2x2x4 multiplication can be done with rank 14 (two 2x2x2 Strassen)
        # Total: 8 * 14 = 112, which is worse.
        
        # Let's just use the 4x4x4 Strassen-Kronecker for each half of k
        pass
    
    # Let's use the simpler approach: two 4x4x4 blocks
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    
    U2_3d = U2.reshape(2, 2, r2)
    V2_3d = V2.reshape(2, 2, r2)
    W2_3d = W2.reshape(2, 2, r2)
    
    U4_5d = np.einsum('abk,cdl->acbdkl', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    V4 = np.einsum('abk,cdl->acbdkl', V2_3d, V2_3d).reshape(16, 49).astype(np.int16)
    W4 = np.einsum('abk,cdl->acbdkl', W2_3d, W2_3d).reshape(16, 49).astype(np.int16)
    
    return U4, V4, W4

def _build_4x4x8():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _residual_frobenius(params, n, m, p, target_rank):
    """Compute Frobenius norm of residual tensor for given factor matrices."""
    U = params[:n*m*target_rank].reshape(n*m, target_rank)
    V = params[n*m*target_rank:(n*m + m*p)*target_rank].reshape(m*p, target_rank)
    W = params[(n*m + m*p)*target_rank:].reshape(p*n, target_rank)
    
    T_comp = np.zeros((n, m, p), dtype=np.float64)
    for r in range(target_rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    residual = np.sum((T_comp - T_target)**2)
    return residual

def _residual_and_grad(params, n, m, p, target_rank):
    """Compute residual and gradient for L-BFGS-B."""
    U = params[:n*m*target_rank].reshape(n*m, target_rank)
    V = params[n*m*target_rank:(n*m + m*p)*target_rank].reshape(m*p, target_rank)
    W = params[(n*m + m*p)*target_rank:].reshape(p*n, target_rank)
    
    T_comp = np.zeros((n, m, p), dtype=np.float64)
    for r in range(target_rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    residual = np.sum((T_comp - T_target)**2)
    
    grad = np.zeros_like(params)
    diff = T_comp - T_target
    
    for r in range(target_rank):
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        Ur = U[:, r].reshape(n, m)
        
        for i in range(n):
            for j in range(m):
                g = 0.0
                for k in range(p):
                    g += diff[i,j,k] * Vr[j,k] * Wr[k,i]
                grad[i*m+j, r] = 2.0 * g
        
        for j in range(m):
            for k in range(p):
                g = 0.0
                for i in range(n):
                    g += diff[i,j,k] * Ur[i,j] * Wr[k,i]
                grad[n*m*target_rank + j*p+k, r] = 2.0 * g
        
        for k in range(p):
            for i in range(n):
                g = 0.0
                for j in range(m):
                    g += diff[i,j,k] * Ur[i,j] * Vr[j,k]
                grad[(n*m+m*p)*target_rank + k*n+i, r] = 2.0 * g
    
    return residual, grad

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50, time_budget=80.0):
    """Try to reduce rank by zeroing out columns with small norm, then re-optimizing."""
    start_time = time.time()
    current_U, current_V, current_W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        if current_rank <= best_rank:
            break
            
        col_norms = np.zeros(current_rank)
        for r in range(current_rank):
            col_norms[r] = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
        
        min_col = np.argmin(col_norms)
        
        test_U = current_U.copy()
        test_V = current_V.copy()
        test_W = current_W.copy()
        test_U[:, min_col] = 0
        test_V[:, min_col] = 0
        test_W[:, min_col] = 0
        
        T_comp = np.zeros((n, m, p), dtype=np.float64)
        for r in range(current_rank):
            if r == min_col:
                continue
            Ur = test_U[:, r].reshape(n, m)
            Vr = test_V[:, r].reshape(m, p)
            Wr = test_W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        
        T_target = np.ones((n, m, p), dtype=np.float64)
        residual = np.sum((T_comp - T_target)**2)
        
        if residual < 1e-6:
            mask = np.arange(current_rank) != min_col
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        else:
            params = np.concatenate([current_U.ravel(), current_V.ravel(), current_W.ravel()])
            
            def objective(params):
                U_p = params[:n*m*current_rank].reshape(n*m, current_rank)
                V_p = params[n*m*current_rank:(n*m+m*p)*current_rank].reshape(m*p, current_rank)
                W_p = params[(n*m+m*p)*current_rank:].reshape(p*n, current_rank)
                T_c = np.zeros((n, m, p), dtype=np.float64)
                for r in range(current_rank):
                    Ur = U_p[:, r].reshape(n, m)
                    Vr = V_p[:, r].reshape(m, p)
                    Wr = W_p[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_c[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                T_t = np.ones((n, m, p), dtype=np.float64)
                return np.sum((T_c - T_t)**2)
            
            try:
                result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 10, 'gtol': 1e-4})
                if result.fun < 1e-6:
                    opt_params = result.x
                    opt_U = opt_params[:n*m*current_rank].reshape(n*m, current_rank)
                    opt_V = opt_params[n*m*current_rank:(n*m+m*p)*current_rank].reshape(m*p, current_rank)
                    opt_W = opt_params[(n*m+m*p)*current_rank:].reshape(p*n, current_rank)
                    test_U2 = opt_U.copy()
                    test_V2 = opt_V.copy()
                    test_W2 = opt_W.copy()
                    test_U2[:, min_col] = 0
                    test_V2[:, min_col] = 0
                    test_W2[:, min_col] = 0
                    T_c2 = np.zeros((n, m, p), dtype=np.float64)
                    for r in range(current_rank):
                        if r == min_col:
                            continue
                        Ur = test_U2[:, r].reshape(n, m)
                        Vr = test_V2[:, r].reshape(m, p)
                        Wr = test_W2[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_c2[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                    T_t2 = np.ones((n, m, p), dtype=np.float64)
                    res2 = np.sum((T_c2 - T_t2)**2)
                    if res2 < 1e-6:
                        mask2 = np.arange(current_rank) != min_col
                        current_U = opt_U[:, mask2]
                        current_V = opt_V[:, mask2]
                        current_W = opt_W[:, mask2]
                        current_rank -= 1
                        best_rank = current_rank
                        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            except Exception:
                pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    n, m, p = 4, 4, 8
    U, V, W = _build_4x4x8()
    
    ok, err = _verify(U, V, W)
    if not ok:
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    time_budget = _time_left() - 10.0
    if time_budget > 5.0:
        U_opt, V_opt, W_opt, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, time_budget=time_budget)
        if reduced_rank < U.shape[1]:
            U_int = np.round(U_opt).astype(np.int16)
            V_int = np.round(V_opt).astype(np.int16)
            W_int = np.round(W_opt).astype(np.int16)
            ok2, err2 = _verify(U_int, V_int, W_int)
            if ok2:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
