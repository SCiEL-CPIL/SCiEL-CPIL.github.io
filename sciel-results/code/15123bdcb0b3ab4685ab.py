import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # We are constructing a rank 32 decomposition.
    # This is composed of:
    # 4 blocks of M_222 (rank 7) for columns 0-1, 2-3, 4-5, 6-7
    # 1 block of M_221 (rank 4) for column 8
    # Total rank = 4*7 + 4 = 32.
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)

    # Convert to numpy arrays for efficient operations
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Target tensor: T[i, j, k] = 1 if i == j else 0 (for matrix multiplication A(i,k) * B(k,j) = C(i,j))
    # But wait, the tensor for matrix multiplication is T[i, j, k] where:
    # A is n x m, B is m x p, C is n x p
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # So the tensor T has indices (i, j, k) and T[i, j, k] = 1 if the k-th term contributes to C[i, j]
    # Actually, for matrix multiplication, the tensor is defined as:
    # T[i, j, k] = 1 if (i, k) is an index of A and (k, j) is an index of B
    # Wait, let me reconsider. The matrix multiplication tensor T is such that:
    # T_{i, j, k} = 1 if i, j, k are valid indices and the multiplication involves A_{i,k} and B_{k,j}
    # Actually, the standard definition is:
    # For A (n x m), B (m x p), C (n x p):
    # C_{i,j} = sum_{k=0}^{m-1} A_{i,k} * B_{k,j}
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,p), k in [0,m)
    # T_{i,j,k} = 1 if the k-th multiplication contributes to C_{i,j}
    # Since every k contributes to every (i,j) pair via A_{i,k} * B_{k,j}, 
    # T_{i,j,k} = 1 for all valid i, j, k.
    
    # Wait, that's not right either. Let me think again.
    # The matrix multiplication tensor is a 3-tensor T with entries T_{i,j,k}
    # such that C_{i,j} = sum_k T_{i,j,k} * A_{i,k} * B_{k,j}
    # For standard matrix multiplication, T_{i,j,k} = 1 for all i, j, k.
    # So the tensor is all ones!
    
    # Let's verify with a small example.
    # A is 2x2, B is 2x2, C is 2x2
    # C_{0,0} = A_{0,0}*B_{0,0} + A_{0,1}*B_{1,0}
    # C_{0,1} = A_{0,0}*B_{0,1} + A_{0,1}*B_{1,1}
    # C_{1,0} = A_{1,0}*B_{0,0} + A_{1,1}*B_{1,0}
    # C_{1,1} = A_{1,0}*B_{0,1} + A_{1,1}*B_{1,1}
    
    # So T_{0,0,0} = 1, T_{0,0,1} = 1, T_{0,1,0} = 1, T_{0,1,1} = 1, etc.
    # Yes, T is all ones.
    
    # Now, a rank-r decomposition means:
    # T_{i,j,k} = sum_{r=0}^{rank-1} U_{i,r} * V_{k,r} * W_{j,r}
    # Wait, the indices need to match. Let me check the convention.
    # The problem says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And W[k*n+i, r] is the convention.
    
    # So U has indices (i*m + k, r) for A_{i,k}
    # V has indices (k*p + j, r) for B_{k,j}
    # W has indices (j*n + i, r) for C_{i,j}
    
    # The decomposition is:
    # C_{i,j} = sum_k A_{i,k} * B_{k,j} = sum_r (sum_k U_{i*m+k, r} * A_{i,k}) * (sum_k V_{k*p+j, r} * B_{k,j}) * W_{j*n+i, r}
    
    # Wait, that's not quite right. Let me reconsider.
    # The tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r}
    
    # But T_{i,j,k} = 1 for all i, j, k.
    # So we need:
    # 1 = sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r}
    
    # Let's verify our current decomposition.
    # For the Strassen block with k0, k1:
    # U_{0*m+0, r} = U_{0, r}, U_{1*m+0, r} = U_{m, r}, U_{2*m+0, r} = U_{2m, r}, U_{3*m+0, r} = U_{3m, r}
    # Wait, n=2, m=2, so n*m = 4.
    # U has 4 rows: indices 0, 1, 2, 3 corresponding to (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    # V has m*p = 2*9 = 18 rows: indices 0..8 for k=0, 9..17 for k=1
    # W has p*n = 9*2 = 18 rows: indices j*n+i for j=0..8, i=0,1
    
    # Let's check the first Strassen term (M1) for k0=0, k1=1:
    # U[0][col] = 1, U[3][col] = 1
    # This means U_{0, r} = 1 (i=0, k=0) and U_{3, r} = 1 (i=1, k=1)
    # V[0*p+0][col] = 1, V[1*p+1][col] = 1
    # This means V_{0, r} = 1 (k=0, j=0) and V_{9+1, r} = V_{10, r} = 1 (k=1, j=1)
    # W[0*n+0][col] = 1, W[1*n+1][col] = 1
    # This means W_{0, r} = 1 (j=0, i=0) and W_{1*2+1, r} = W_{3, r} = 1 (j=1, i=1)
    
    # So the contribution to T_{i,j,k} from this term is:
    # U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r}
    
    # For (i=0, j=0, k=0): U_{0, r} * V_{0, r} * W_{0, r} = 1 * 1 * 1 = 1
    # For (i=1, j=1, k=1): U_{3, r} * V_{10, r} * W_{3, r} = 1 * 1 * 1 = 1
    # For all other (i,j,k): at least one factor is 0.
    
    # So this term contributes 1 to T_{0,0,0} and T_{1,1,1}.
    
    # Let's check the second term (M2):
    # U[2][col] = 1, U[3][col] = 1
    # U_{2, r} = 1 (i=1, k=0), U_{3, r} = 1 (i=1, k=1)
    # V[0*p+0][col] = 1
    # V_{0, r} = 1 (k=0, j=0)
    # W[0*n+1][col] = 1, W[1*n+1][col] = -1
    # W_{1, r} = 1 (j=0, i=1), W_{3, r} = -1 (j=1, i=1)
    
    # For (i=1, j=0, k=0): U_{2, r} * V_{0, r} * W_{1, r} = 1 * 1 * 1 = 1
    # For (i=1, j=1, k=1): U_{3, r} * V_{10, r} * W_{3, r} = 1 * 0 * (-1) = 0 (V_{10, r} = 0 for this term)
    # Wait, V[1*p+1] is not set for this term, so V_{10, r} = 0.
    # So only (i=1, j=0, k=0) gets 1.
    
    # This is getting complex. Let me just compute the residual using numpy.
    
    def compute_residual(U_arr, V_arr, W_arr):
        # U_arr: (n*m, rank), V_arr: (m*p, rank), W_arr: (p*n, rank)
        # T_{i,j,k} = sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r}
        # We want T_{i,j,k} = 1 for all i, j, k
        
        n, m, p = DIMENSIONS
        residual = np.zeros((n, p, m))
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # U index: i*m + k
                    # V index: k*p + j
                    # W index: j*n + i
                    u_val = U_arr[i*m + k]
                    v_val = V_arr[k*p + j]
                    w_val = W_arr[j*n + i]
                    # Sum over r
                    val = np.dot(u_val, v_val * w_val)
                    residual[i, j, k] = val - 1.0
        
        return residual
    
    # Compute initial residual
    residual = compute_residual(U_arr, V_arr, W_arr)
    max_res = np.max(np.abs(residual))
    
    # Try to reduce the residual by optimizing the decomposition
    # We'll use a simple gradient-based approach
    
    # The objective is to minimize the maximum absolute residual
    # f(U, V, W) = max_{i,j,k} |sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r} - 1|
    
    # This is a non-smooth objective. We can use a smooth surrogate:
    # g(U, V, W) = log(sum_{i,j,k} exp(beta * |residual_{i,j,k}|))
    
    # Or we can use a different approach: minimize the sum of squares of residuals
    # h(U, V, W) = sum_{i,j,k} (sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r} - 1)^2
    
    # Let's try minimizing the sum of squares first, as it's smoother.
    
    def compute_sum_sq_residual(U_arr, V_arr, W_arr):
        residual = compute_residual(U_arr, V_arr, W_arr)
        return np.sum(residual ** 2)
    
    # Use scipy.optimize.minimize to optimize U, V, W
    from scipy.optimize import minimize
    
    def flatten_params(U_arr, V_arr, W_arr):
        return np.concatenate([U_arr.flatten(), V_arr.flatten(), W_arr.flatten()])
    
    def unflatten_params(params):
        size_u = U_arr.shape[0] * U_arr.shape[1]
        size_v = V_arr.shape[0] * V_arr.shape[1]
        size_w = W_arr.shape[0] * W_arr.shape[1]
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U_new = U_flat.reshape(U_arr.shape)
        V_new = V_flat.reshape(V_arr.shape)
        W_new = W_flat.reshape(W_arr.shape)
        
        return U_new, V_new, W_new
    
    def objective(params):
        U_new, V_new, W_new = unflatten_params(params)
        return compute_sum_sq_residual(U_new, V_new, W_new)
    
    # Initial parameters
    init_params = flatten_params(U_arr, V_arr, W_arr)
    
    # Try to optimize
    result = None
    try:
        result = minimize(objective, init_params, method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-10})
    except Exception as e:
        pass
    
    if result is not None and result.fun < compute_sum_sq_residual(U_arr, V_arr, W_arr):
        U_opt, V_opt, W_opt = unflatten_params(result.x)
        U_arr = U_opt
        V_arr = V_opt
        W_arr = W_opt
    
    # Check if we can round to integers/half-integers
    # The problem requires integer or half-integer entries
    
    def can_round(arr):
        # Check if arr can be rounded to integer or half-integer
        # i.e., 2*arr is close to an integer
        return np.all(np.abs(2*arr - np.round(2*arr)) < 1e-6)
    
    if can_round(U_arr) and can_round(V_arr) and can_round(W_arr):
        # Round to half-integers
        U_rounded = np.round(2*U_arr) / 2
        V_rounded = np.round(2*V_arr) / 2
        W_rounded = np.round(2*W_arr) / 2
        
        # Check if the rounded version still has small residual
        residual_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(residual_rounded)) < 1e-6:
            U_arr = U_rounded
            V_arr = V_rounded
            W_arr = W_rounded
    
    # Convert back to lists
    U_list = U_arr.tolist()
    V_list = V_arr.tolist()
    W_list = W_arr.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
