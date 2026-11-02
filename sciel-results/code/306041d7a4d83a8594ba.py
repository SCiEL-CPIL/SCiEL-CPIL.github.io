import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # First, ensure we have the rank 28 fallback
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    # Try to find rank 27
    if time.time() < deadline - 5:
        result = _search_rank_27_linear(n, m, p, 27, deadline - 5)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])

    # Fallback to rank 28
    return _to_lists(U28, V28, W28)

def _build_rank28_arrays(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: j=0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Block 2a: j=1,2, k=0,1, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    # Block 2b: j=1,2, k=2, schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
            
    # Block 2c: j=1,2, k=3,4, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 3, col)
    
    return U, V, W

def _add_strassen(U, V, W, n, m, p, base_j, base_k, col):
    # Strassen for 2x2x2 block
    # i in {0,1}, j in {base_j, base_j+1}, k in {base_k, base_k+1}
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # 1
    U[0 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 2
    U[1 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    W[base_k * n + 1, col] = 1
    W[(base_k + 1) * n + 1, col] = -1
    col += 1
    
    # 3
    U[0 * m + base_j, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 4
    U[1 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + base_k, col] = 1
    V[base_j * p + base_k, col] = -1
    W[base_k * n + 0, col] = 1
    W[base_k * n + 1, col] = 1
    col += 1
    
    # 5
    U[0 * m + base_j, col] = 1
    U[0 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    col += 1
    
    # 6
    U[1 * m + base_j, col] = 1
    U[0 * m + base_j, col] = -1
    V[base_j * p + base_k, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 7
    U[0 * m + (base_j + 1), col] = 1
    U[1 * m + (base_j + 1), col] = -1
    V[(base_j + 1) * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    col += 1
    
    return col

def _search_rank_27_linear(n, m, p, target_rank, deadline):
    """
    Search for rank 27 solution by fixing U and V to random sparse half-integers
    and solving the linear system for W.
    """
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    num_constraints = n * m * p  # 30
    
    # W vector length is target_rank * dim_w
    # Equation index: i * m * p + j * p + k
    # W index: t * dim_w + k * n + i
    
    np.random.seed(42)
    
    # Try different sparsity levels
    for sparsity in [0.2, 0.3, 0.4, 0.5]:
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Generate random U and V
            # Entries from {-1, -0.5, 0, 0.5, 1}
            # Bias towards 0 for sparsity
            
            def gen_sparse_matrix(rows, cols, sparsity):
                mat = np.zeros((rows, cols))
                for r in range(rows):
                    for c in range(cols):
                        if np.random.rand() < sparsity:
                            val = np.random.choice([-1, -0.5, 0.5, 1])
                            mat[r, c] = val
                return mat

            U = gen_sparse_matrix(dim_u, target_rank, sparsity)
            V = gen_sparse_matrix(dim_v, target_rank, sparsity)
            
            # Build M matrix (30 x 270)
            M = np.zeros((num_constraints, target_rank * dim_w))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for t in range(target_rank):
                            u_val = U[i * m + j, t]
                            v_val = V[j * p + k, t]
                            if u_val != 0 and v_val != 0:
                                w_idx = t * dim_w + k * n + i
                                M[eq_idx, w_idx] = u_val * v_val
            
            # Solve M w = 1
            try:
                w_solution, residuals, rank, sv = np.linalg.lstsq(M, np.ones(num_constraints), rcond=None)
            except Exception:
                continue
                
            # Check if w_solution is close to half-integers
            w_solution = w_solution.reshape(target_rank, dim_w)
            
            # Snap to half-integers
            w_snapped = np.round(w_solution * 2) / 2
            
            # Check error
            error = np.max(np.abs(w_solution - w_snapped))
            if error < 1e-5:
                # Verify exact solution with snapped values
                U_check = U
                V_check = V
                W_check = w_snapped.T # W is (dim_w, rank)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(target_rank):
                                val += U_check[i*m+j, t] * V_check[j*p+k, t] * W_check[k*n+i, t]
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                    
                if valid:
                    return U_check, V_check, W_check

    return None

def _to_lists(U, V, W):
    """
    Convert numpy arrays to lists of integers or half-integers.
    """
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Check if it's close to an integer
                if abs(val - round(val)) < 1e-8:
                    row_list.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    # Half-integer
                    half = round(val * 2) / 2
                    if abs(half - int(half)) < 1e-8:
                        row_list.append(int(half))
                    else:
                        row_list.append(float(half))
                else:
                    # Shouldn't happen for valid solutions, but handle gracefully
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
