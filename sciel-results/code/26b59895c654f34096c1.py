import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a rank-minimal exact tensor decomposition for multiplying a 2x4 by a 4x4 matrix.
    
    Strategy:
    1. Build the full matrix-multiplication tensor T[i,j,k] = 1 if k == i + j (0-indexed) with proper indexing.
       Actually, for matmul (A is n x m, B is m x p, C is n x p):
       C[i,k] = sum_j A[i,j] * B[j,k]
       Tensor T[i,j,k] = delta_{i,k} is not right. The tensor has entries T[i,j,k] = 1 always
       (it's a structured tensor). Let me reconsider.
       
       The matrix multiplication tensor M_{i,j,k} for C = A @ B is:
       C[i,k] = sum_j A[i,j] * B[j,k]
       So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity on the middle index).
       More precisely, T is a 3D tensor of shape (n*m*p) where T[i,j,k] = 1.
       
       We want to decompose T = sum_r u^{(r)} \otimes v^{(r)} \otimes w^{(r)}
       where u^{(r)} is in R^{n*m}, v^{(r)} is in R^{m*p}, w^{(r)} is in R^{p*n}
       (with W indexed as W[k*n+i, r] per the trace convention).
       
       Actually, the decomposition is:
       T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
       
       So we need U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
       
    2. Use randomized low-rank tensor decomposition with exact verification.
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Tensor T[i,j,k] = 1 for all i,j,k (the matmul tensor)
    # We'll work in flattened form:
    # U index: i*m + j  (row i, col j of A)
    # V index: j*p + k  (row j, col k of B)
    # W index: k*n + i  (col k, row i of C, per trace convention W[k*n+i, r])
    
    best_rank = 28  # known Strassen-based construction
    best_U = None
    best_V = None
    best_W = None
    
    # First, provide the known rank-28 construction as a safe fallback
    U28 = np.zeros((n * m, 28), dtype=int)
    V28 = np.zeros((m * p, 28), dtype=int)
    W28 = np.zeros((p * n, 28), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = a_col_offset * 2 + c
                g_idx = r * 4 + g_col
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * 4 + g_col
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = c_col_offset * 2 + c
                g_idx = g_col * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff
    
    apply_strassen_block(0, 0, 0, 0, 0, U28, V28, W28)
    apply_strassen_block(1, 1, 0, 0, 7, U28, V28, W28)
    apply_strassen_block(0, 0, 1, 1, 14, U28, V28, W28)
    apply_strassen_block(1, 1, 1, 1, 21, U28, V28, W28)
    
    best_U = U28.copy()
    best_V = V28.copy()
    best_W = W28.copy()
    best_rank = 28
    
    # Now try to find a lower-rank decomposition using tensor decomposition methods
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # Shape: (n*m, m*p, p*n) = (8, 16, 8)
    
    # We can try truncated SVD on matrix unfoldings to get candidate low-rank decompositions
    # Then verify exactness.
    
    # For exact decomposition with integer/half-integer entries, we can try:
    # 1. Build the tensor T as a 3D array
    # 2. Try rank-r decompositions using randomized methods
    # 3. Verify exactness by checking all T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    T = np.ones((n * m, m * p, p * n))  # T[i*m+j, j'*p+k, k'*n+i] - wait, need to be careful
    
    # Actually, the matmul tensor for (n,m,p) = (2,4,4) is:
    # T is a tensor where the "support" is all ones. The entry T[u, v, w] = 1 for all u, v, w
    # where u indexes (i,j), v indexes (j',k), w indexes (k',i').
    # For it to be a valid matmul decomposition, we need:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # But that's not quite right. Let me re-derive.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (sum_c U[i,j,c] * s_c) * (sum_c V[j,k,c] * s_c) ... no, that's not how it works.
    
    # The tensor decomposition of matmul is:
    # The matmul tensor T has T[i,j,k] = 1 (as a formal tensor).
    # A rank-r decomposition is T = sum_{c=1}^r u_c \otimes v_c \otimes w_c
    # where (u_c)_ij = U[i*m+j, c], (v_c)_jk = V[j*p+k, c], (w_c)_ki = W[k*n+i, c]
    # Then: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # But wait, T[i,j,k] is only defined for compatible indices. For matmul, the tensor is:
    # T is supported on all (i,j,k) tuples where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1.
    
    # So the condition is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # That's a system of n*m*p = 2*4*4 = 32 equations in 32*rank unknowns (with the constraint
    # that U, V, W entries are integers or half-integers).
    
    # Actually no - it's n*m*p equations but the unknowns are rank*(n*m + m*p + p*n).
    
    # Let me try a randomized approach:
    # Generate random low-rank candidates and check if they work.
    
    np.random.seed(42)
    
    def verify_decomposition(U_arr, V_arr, W_arr, rank):
        """Check if U, V, W form a valid matmul tensor decomposition."""
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    # Try ranks from 20 down to 28
    # For each rank, try several random initializations and use a gradient-free refinement
    
    # Actually, let's try a smarter approach. The tensor T = ones((8,16,8)) can be written
    # as a sum of rank-1 tensors. The minimal rank is the tensor rank.
    
    # For the specific structure of matmul tensor, known bounds:
    # For 2x4x4, the rank is between some lower bound and 28 (our construction).
    # Let me try to use the fact that we can fix one matrix and solve for the others.
    
    # Approach: Try rank r. Fix U to be random binary {0,1} patterns.
    # Then for each c, the contribution is U[:,c] \otimes V[:,c] \otimes W[:,c]
    # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j), the vector w_{ij} defined by
    # w_{ij}[k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must equal 1 for all k.
    
    # Let me try a different angle: use the known algebraic structure.
    
    # For n=2, m=4, p=4, we can write:
    # A = [[A1, A2]] where A1, A2 are 2x2
    # B = [[B1, B2], [B3, B4]] where B1,...,B4 are 2x2
    # C = A @ B = [A1*B1 + A2*B3, A1*B2 + A2*B4]
    
    # This is what the Strassen construction uses. Each 2x2 matmul has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Can we do better by sharing terms between the block products?
    # The four block products are:
    # C1 = A1*B1 + A2*B3  (needs rank 7 for A1*B1 plus rank 7 for A2*B3, but can we share?)
    # Actually, A1*B1, A1*B2, A2*B3, A2*B4 are four separate 2x2 matmuls.
    # Each needs rank 7, but they share the A matrices (A1 appears in C1 and C2, etc.)
    
    # Let me try to use a rank-r decomposition of each 2x2 matmul and see if the total
    # can be less than 28.
    
    # Alternative: try to find a rank-24, 25, 26, 27 decomposition using numerical methods.
    
    # For small tensors, we can try to use the "border rank" or direct optimization.
    
    # Let me try: for rank r from 27 down to 20:
    # 1. Initialize U, V, W randomly with small integers
    # 2. For each triple (i,j,k), the equation is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # 3. This is a system of 32 polynomial equations in 8*r + 16*r + 8*r = 32*r unknowns
    # 4. For r >= 32, there are more unknowns than equations, so solutions exist generically.
    # 5. For r < 32, it's a constrained problem.
    
    # Actually, for r = 28, we already have a solution. Let me try to find solutions for
    # r = 27, 26, ..., using randomized search.
    
    # Strategy for exact search with small integer entries:
    # Since we need integer or half-integer entries, we can scale by 2 and search in integers.
    
    # Let me try a simple randomized approach with verification.
    
    def try_rank(r, n_tries=500, time_limit=10):
        """Try to find a valid rank-r decomposition with integer entries."""
        np.random.seed(42 + r)
        for trial in range(n_tries):
            if time.time() - t_start > time_limit:
                break
            # Initialize with random values in {-1, 0, 1}
            U_arr = np.random.choice([-1, 0, 1], size=(n*m, r))
            V_arr = np.random.choice([-1, 0, 1], size=(m*p, r))
            W_arr = np.random.choice([-1, 0, 1], size=(p*n, r))
            
            ok, _ = verify_decomposition(U_arr, V_arr, W_arr, r)
            if ok:
                return U_arr, V_arr, W_arr
        
        return None, None, None
    
    # Try ranks 27, 26, 25
    for r in [27, 26, 25, 24, 23, 22, 21, 20]:
        U_new, V_new, W_new = try_rank(r, n_tries=200, time_limit=5)
        if U_new is not None:
            best_rank = r
            best_U = U_new
            best_V = V_new
            best_W = W_new
            break
        if time.time() - t_start > DEADLINE - 20:
            break
    
    # If we found a better rank, verify it properly
    if best_rank < 28:
        ok, _ = verify_decomposition(best_U, best_V, best_W, best_rank)
        if not ok:
            best_rank = 28
            best_U = U28.copy()
            best_V = V28.copy()
            best_W = W28.copy()
    
    # Also try a more structured search for rank 27
    # Use the fact that we can perturb the rank-28 solution and see if we can merge two columns
    
    # Final check
    U_out = best_U.astype(float)
    V_out = best_V.astype(float)
    W_out = best_W.astype(float)
    
    # Verify one more time
    ok, info = verify_decomposition(best_U, best_V, best_W, best_rank)
    if not ok:
        # Fall back to rank 28
        best_rank = 28
        U_out = U28.astype(float)
        V_out = V28.astype(float)
        W_out = W28.astype(float)
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]
    
    # Test with multiple random matrices
    np.random.seed(42)
    all_ok = True
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                for cc in range(rank):
                    acc = 0.0
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc]
                    Cf[ii, kk] += acc * W[kk * n + ii, cc]
        if not np.allclose(C, Cf, atol=1e-10):
            print(f"Test failed for trial {trial}")
            all_ok = False
            break
    if all_ok:
        print(f"OK rank {rank} in {time.time()-t0:.1f}s")
    else:
        print(f"FAILED rank {rank} in {time.time()-t0:.1f}s")
        exit(1)
