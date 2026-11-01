"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Returns U, V, W for schoolbook decomposition (rank 52)."""
    n, m, p = 2, 2, 13
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor():
    """Builds the full target tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 13
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0
    return T

def _strassen_block_decomposition(k0, k1):
    """
    Returns 7 rank-1 terms (u, v, w) for a 2x2x2 sub-block covering
    B columns k0, k1.
    
    Uses the known Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    """
    n, m, p = 2, 2, 13
    # Global index maps
    v_map = [k0, k1, m + k0, m + k1]
    w_map = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
    
    # Strassen's 7 products for 2x2 matmul:
    # P1 = A00 * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = (A00 + A01) * B11
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A10) * (B01 - B11)
    # P6 = (A01 - A00 + A11) * B10
    # P7 = (A01 - A11) * (B00 + B11)
    
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P7
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P6
    
    # Each product P_t is bilinear: P_t = u_t^T A v_t where A is 2x2
    # In tensor decomposition form, each rank-1 term is:
    #   u_t (in U space, size 4) * v_t (in V space, size 4) * w_t (in W space, size 4)
    
    # U space indices: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
    # V space indices: 0=(k=0,j=k0), 1=(k=0,j=k1), 2=(k=1,j=k0), 3=(k=1,j=k1)
    # W space indices: 0=(j=k0,i=0), 1=(j=k0,i=1), 2=(j=k1,i=0), 3=(j=k1,i=1)
    
    # For each Strassen product P_t, we need to express it as a rank-1 term
    # in the tensor decomposition. The key insight is that each product involves
    # specific linear combinations of A entries and B entries.
    
    # Let me define the 7 terms explicitly.
    # Term t contributes to the tensor as:
    #   T[a,b,c] += U[a,t] * V[b,t] * W[c,t]
    
    # For P1 = A00 * (B00 + B11):
    #   A00 corresponds to U index 0
    #   B00 corresponds to V index 0, B11 corresponds to V index 3
    #   The result adds to C00 (W index 0) and C11 (W index 3)
    #   But wait, P1 contributes to both C00 and C11 with different signs.
    #   Actually, in the tensor decomposition, each rank-1 term is a single
    #   outer product. So P1 as a product A00*(B00+B11) is not a single
    #   rank-1 term in the tensor sense.
    
    # Let me think about this differently. The tensor T for 2x2x2 matmul has
    # 8 non-zero entries. We need to decompose it into 7 rank-1 tensors.
    
    # The 8 non-zero entries are:
    # (0,0,0): A00*B00 -> C00
    # (0,1,2): A00*B01 -> C01  (wait, let me recheck)
    
    # Actually, let me re-derive. The matmul tensor T[i,k,j] = 1 if the output
    # C[i,j] depends on A[i,k]*B[k,j].
    
    # For 2x2 * 2x2 -> 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # So the 8 non-zero entries of T are:
    # T[A00, B00, C00] = 1  -> (0, 0, 0)
    # T[A00, B01, C01] = 1  -> (0, 1, 2)
    # T[A01, B10, C00] = 1  -> (1, 2, 0)
    # T[A01, B11, C01] = 1  -> (1, 3, 2)
    # T[A10, B00, C10] = 1  -> (2, 0, 1)
    # T[A10, B01, C11] = 1  -> (2, 1, 3)
    # T[A11, B10, C10] = 1  -> (3, 2, 1)
    # T[A11, B11, C11] = 1  -> (3, 3, 3)
    
    # Now I need to find 7 vectors u_t, v_t, w_t in R^4 such that:
    # sum_{t=1}^7 u_t[a] * v_t[b] * w_t[c] = T[a,b,c]
    
    # This is the classical Strassen decomposition. Let me use a known valid one.
    # After research, here's a valid rank-7 decomposition:
    
    # Term 1: u=[1,0,0,0], v=[1,0,0,1], w=[1,0,0,1]
    #   This gives: T[0,0,0] += 1, T[0,0,3] += 1, T[0,3,0] += 1, T[0,3,3] += 1
    #   But we want T[0,0,0]=1, T[0,1,2]=1, etc. This doesn't match.
    
    # Let me use a different approach: solve the system numerically.
    # I'll use the fact that the tensor is small (4x4x4) and find a rank-7
    # decomposition via optimization.
    
    def find_rank7_decomposition():
        """Find rank-7 decomposition of 2x2x2 matmul tensor via optimization."""
        # Target tensor
        T_loc = np.zeros((4, 4, 4))
        nonzeros = [(0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)]
        for a, b, c in nonzeros:
            T_loc[a, b, c] = 1.0
        
        # Initialize with random vectors
        np.random.seed(12345)
        U_t = np.random.randn(4, 7) * 0.3
        V_t = np.random.randn(4, 7) * 0.3
        W_t = np.random.randn(4, 7) * 0.3
        
        # Optimize using gradient descent
        lr = 0.05
        for iteration in range(300):
            T_approx = np.einsum('at,bt,ct->abc', U_t, V_t, W_t)
            R = T_loc - T_approx
            
            grad_U = np.einsum('abc,bt,ct->at', R, V_t, W_t)
            grad_V = np.einsum('abc,at,ct->bt', R, U_t, W_t)
            grad_W = np.einsum('abc,at,bt->ct', R, U_t, V_t)
            
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
        
        return U_t, V_t, W_t
    
    U_t, V_t, W_t = find_rank7_decomposition()
    
    # Map to global indices
    terms = []
    for t in range(7):
        u_global = np.zeros(4)  # n*m = 4
        v_global = np.zeros(m * p)  # m*p = 26
        w_global = np.zeros(p * n)  # p*n = 26
        
        # U local to global: identity (local index = global index since n*m=4)
        u_global[:] = U_t[:, t]
        
        # V local to global
        for local_idx, global_idx in enumerate(v_map):
            v_global[global_idx] = V_t[local_idx, t]
        
        # W local to global
        for local_idx, global_idx in enumerate(w_map):
            w_global[global_idx] = W_t[local_idx, t]
        
        terms.append((u_global, v_global, w_global))
    
    return terms

def _verify_decomposition(U, V, W, T_target, tol=1e-8):
    """Verify that U, V, W decompose T_target."""
    T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
    error = np.max(np.abs(T_approx - T_target))
    return error < tol, error

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit with safety margin
    
    n, m, p = 2, 2, 13
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Build target tensor for verification
    T_target = _build_target_tensor()
    
    # Try to build a Strassen-based decomposition
    # 6 blocks of 2 columns + 1 block of 1 column
    # Each 2-column block: rank 7
    # 1-column block: rank 4 (schoolbook for 2x2*2x1)
    # Total: 6*7 + 4 = 46
    
    try:
        if time.time() < deadline - 5:
            U_strassen = np.zeros((n * m, 46))
            V_strassen = np.zeros((m * p, 46))
            W_strassen = np.zeros((p * n, 46))
            
            col = 0
            for block in range(6):
                k0 = block * 2
                k1 = block * 2 + 1
                
                terms = _strassen_block_decomposition(k0, k1)
                for u, v, w in terms:
                    U_strassen[:, col] = u
                    V_strassen[:, col] = v
                    W_strassen[:, col] = w
                    col += 1
            
            # Last block: 1 column (k=12)
            # Schoolbook for 2x2 * 2x1: rank 4
            k = 12
            for i in range(n):
                for kk in range(m):
                    U_strassen[i * m + kk, col] = 1.0
                    V_strassen[kk * p + k, col] = 1.0
                    W_strassen[k * n + i, col] = 1.0
                    col += 1
            
            # Verify the decomposition
            valid, error = _verify_decomposition(U_strassen, V_strassen, W_strassen, T_target)
            
            if valid:
                best_rank = 46
                best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
            
            # If not valid, try to improve with more optimization iterations
            elif time.time() < deadline - 10:
                # Re-run with more iterations and different seeds
                for seed in [42, 123, 456, 789, 1000]:
                    if time.time() > deadline - 5:
                        break
                    
                    # Patch the random seed in the decomposition function
                    old_seed = np.random.get_state()
                    np.random.seed(seed)
                    
                    U_strassen2 = np.zeros((n * m, 46))
                    V_strassen2 = np.zeros((m * p, 46))
                    W_strassen2 = np.zeros((p * n, 46))
                    
                    col = 0
                    for block in range(6):
                        k0 = block * 2
                        k1 = block * 2 + 1
                        
                        # Re-derive with current seed
                        v_map = [k0, k1, m + k0, m + k1]
                        w_map = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
                        
                        T_loc = np.zeros((4, 4, 4))
                        nonzeros = [(0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)]
                        for a, b, c in nonzeros:
                            T_loc[a, b, c] = 1.0
                        
                        U_t = np.random.randn(4, 7) * 0.3
                        V_t = np.random.randn(4, 7) * 0.3
                        W_t = np.random.randn(4, 7) * 0.3
                        
                        lr = 0.05
                        for iteration in range(500):
                            T_approx = np.einsum('at,bt,ct->abc', U_t, V_t, W_t)
                            R = T_loc - T_approx
                            
                            grad_U = np.einsum('abc,bt,ct->at', R, V_t, W_t)
                            grad_V = np.einsum('abc,at,ct->bt', R, U_t, W_t)
                            grad_W = np.einsum('abc,at,bt->ct', R, U_t, V_t)
                            
                            U_t -= lr * grad_U
                            V_t -= lr * grad_V
                            W_t -= lr * grad_W
                        
                        for t in range(7):
                            u_global = np.zeros(4)
                            v_global = np.zeros(m * p)
                            w_global = np.zeros(p * n)
                            
                            u_global[:] = U_t[:, t]
                            for local_idx, global_idx in enumerate(v_map):
                                v_global[global_idx] = V_t[local_idx, t]
                            for local_idx, global_idx in enumerate(w_map):
                                w_global[global_idx] = W_t[local_idx, t]
                            
                            U_strassen2[:, col] = u_global
                            V_strassen2[:, col] = v_global
                            W_strassen2[:, col] = w_global
                            col += 1
                    
                    # Last block: 1 column (k=12)
                    k = 12
                    for i in range(n):
                        for kk in range(m):
                            U_strassen2[i * m + kk, col] = 1.0
                            V_strassen2[kk * p + k, col] = 1.0
                            W_strassen2[k * n + i, col] = 1.0
                            col += 1
                    
                    valid2, error2 = _verify_decomposition(U_strassen2, V_strassen2, W_strassen2, T_target)
                    if valid2:
                        best_rank = 46
                        best_U, best_V, best_W = U_strassen2, V_strassen2, W_strassen2
                        break
                    
                    np.random.set_state(old_seed)
    
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Convert to lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
