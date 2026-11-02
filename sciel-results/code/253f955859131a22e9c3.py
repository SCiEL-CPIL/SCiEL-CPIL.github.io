import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Construct the known rank-42 decomposition using Kronecker product of 
       Strassen (2x2x2, rank 7) and Trivial (1x2x3, rank 6).
    2. Apply tensor network disentanglement: check if rank-1 components are
       linearly dependent in the 48-dimensional tensor space, and if so,
       remove redundant components to achieve lower rank.
    3. Return the best valid decomposition found.
    """
    n, m, p = 2, 4, 6
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Core 1: 2x2x2 Strassen Tensor (Rank 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    U1[0, 0] = 1; U1[3, 0] = 1
    U1[2, 1] = 1; U1[3, 1] = 1
    U1[0, 2] = 1
    U1[3, 3] = 1
    U1[0, 4] = 1; U1[1, 4] = 1
    U1[2, 5] = 1; U1[0, 5] = -1
    U1[1, 6] = 1; U1[3, 6] = -1
    
    V1[0, 0] = 1; V1[3, 0] = 1
    V1[0, 1] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    V1[2, 3] = 1; V1[0, 3] = -1
    V1[3, 4] = 1
    V1[0, 5] = 1; V1[1, 5] = 1
    V1[2, 6] = 1; V1[3, 6] = 1
    
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1
    W1[2, 2] = 1; W1[2, 4] = 1
    W1[1, 1] = 1; W1[1, 3] = 1
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1
    
    # Core 2: 1x2x3 Trivial Tensor (Rank 6)
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    U2[0, 0] = 1; U2[0, 1] = 1; U2[0, 2] = 1
    U2[1, 3] = 1; U2[1, 4] = 1; U2[1, 5] = 1
    
    V2[0, 0] = 1; V2[1, 1] = 1; V2[2, 2] = 1
    V2[3, 3] = 1; V2[4, 4] = 1; V2[5, 5] = 1
    
    W2[0, 0] = 1; W2[0, 3] = 1
    W2[1, 1] = 1; W2[1, 4] = 1
    W2[2, 2] = 1; W2[2, 5] = 1
    
    # Kronecker Product Construction
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2 + j1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for j2 in range(2):
                                val2 = U2[i2*2 + j2, r2]
                                if val2 != 0:
                                    i = i1
                                    a = j1 * 2 + j2
                                    idx_u = i * 4 + a
                                    U[idx_u, r] += val1 * val2
            
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2 + k1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                val2 = V2[j2*3 + k2, r2]
                                if val2 != 0:
                                    a = j1 * 2 + j2
                                    b = k1 * 3 + k2
                                    idx_v = a * 6 + b
                                    V[idx_v, r] += val1 * val2
            
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1*2 + i1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for k2 in range(3):
                                val2 = W2[k2*1 + i2, r2]
                                if val2 != 0:
                                    i = i1
                                    b = k1 * 3 + k2
                                    idx_w = b * 2 + i
                                    W[idx_w, r] += val1 * val2
    
    # Verification function
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        np.random.seed(42)
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_mat[:, r]
            v_vec = V_mat[:, r]
            w_vec = W_mat[:, r]
            
            for i in range(n):
                for b in range(p):
                    w_val = w_vec[b*n + i]
                    if w_val == 0:
                        continue
                    sum_a = 0
                    for a in range(m):
                        sum_a += u_vec[i*m + a] * v_vec[a*p + b]
                    C_computed[i, b] += sum_a * w_val
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    # Flatten a rank-1 component into a vector in R^{n*m*p}
    def flatten_component(U_mat, V_mat, W_mat, n, m, p, r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for a in range(m):
                u_val = U_mat[i*m + a, r]
                if u_val == 0:
                    continue
                for b in range(p):
                    v_val = V_mat[a*p + b, r]
                    if v_val == 0:
                        continue
                    w_val = W_mat[b*n + i, r]
                    if w_val == 0:
                        continue
                    vec[i*m*p + a*p + b] = u_val * v_val * w_val
        return vec
    
    # Verify the rank-42 solution
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build matrix of flattened components: 42 x 48
    M = np.zeros((42, n * m * p))
    for r in range(42):
        M[r, :] = flatten_component(U, V, W, n, m, p, r)
    
    # Check rank of the component matrix
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    best_U, best_V, best_W = U, V, W
    best_rank = 42
    
    # If components are linearly dependent, find a minimal independent subset
    if rank_M < 42:
        selected = []
        current_rank = 0
        remaining = list(range(42))
        
        while remaining and current_rank < rank_M:
            best_idx = -1
            best_new_rank = current_rank
            for idx in remaining:
                test_rows = selected + [idx]
                test_M = M[test_rows, :]
                test_rank = np.linalg.matrix_rank(test_M, tol=1e-10)
                if test_rank > best_new_rank:
                    best_new_rank = test_rank
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_rank = best_new_rank
        
        if len(selected) < 42:
            mask = np.zeros(42, dtype=bool)
            mask[selected] = True
            new_U = U[:, mask]
            new_V = V[:, mask]
            new_W = W[:, mask]
            
            if verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = len(selected)
    
    # Try removing one component at a time if still at rank 42
    if best_rank == 42 and time.time() < deadline - 30:
        for r_remove in range(42):
            if time.time() > deadline - 30:
                break
            mask = np.ones(42, dtype=bool)
            mask[r_remove] = False
            new_U = U[:, mask]
            new_V = V[:, mask]
            new_W = W[:, mask]
            if verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = 41
                break
    
    # Try to reduce further if at rank 41
    if best_rank == 41 and time.time() < deadline - 20:
        for r_remove in range(41):
            if time.time() > deadline - 20:
                break
            mask = np.ones(41, dtype=bool)
            mask[r_remove] = False
            new_U = best_U[:, mask]
            new_V = best_V[:, mask]
            new_W = best_W[:, mask]
            if verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = 40
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    construct()
