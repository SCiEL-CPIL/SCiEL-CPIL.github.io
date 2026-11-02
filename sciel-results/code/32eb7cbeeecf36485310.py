import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    time_limit = 140.0
    
    # Build the rank 28 solution
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define the 7 Strassen terms for 2x2 multiplication
    strassen_terms = [
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    ]
    
    u_map_A1 = [0, 1, 4, 5]
    u_map_A2 = [2, 3, 6, 7]
    v_map_B11 = [0, 1, 4, 5]
    v_map_B12 = [2, 3, 6, 7]
    v_map_B21 = [8, 9, 12, 13]
    v_map_B22 = [10, 11, 14, 15]
    w_map_C1 = [0, 2, 1, 3]
    w_map_C2 = [4, 6, 5, 7]
    
    col = 0
    blocks = [
        (u_map_A1, v_map_B11, w_map_C1),
        (u_map_A2, v_map_B21, w_map_C1),
        (u_map_A1, v_map_B12, w_map_C2),
        (u_map_A2, v_map_B22, w_map_C2)
    ]
    
    for u_map, v_map, w_map in blocks:
        for u_t, v_t, w_t in strassen_terms:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
            
    best_U, best_V, best_W = U, V, W
    best_rank = 28
    
    def verify(U, V, W):
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-7)

    # Check for linear dependence to reduce rank to 27
    try:
        # Construct the tensor vectors for each term
        # T_i is a tensor of shape (n, m, p)
        # Flatten to vector of size n*m*p = 32
        tensor_vectors = []
        for c in range(rank):
            T_c = np.einsum('ic,jc,kc->ijk', U[:, c], V[:, c], W[:, c])
            tensor_vectors.append(T_c.ravel())
        
        T_matrix = np.array(tensor_vectors).T # Shape (32, 28)
        
        # Find null space
        # We want c such that T_matrix @ c = 0
        # Use SVD
        U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        # The last row of Vt corresponds to the smallest singular value
        # If the smallest singular value is close to 0, there is a dependency
        if s[-1] < 1e-9:
            c = Vt[-1]
            # c is the coefficient vector
            # T_27 (index 27) is dependent on the others?
            # Actually, any term can be removed if its coefficient is non-zero.
            # Let's pick the term with the largest absolute coefficient to remove?
            # Or just the last one if c[-1] is not zero.
            
            # We want to express T_k as sum_{j != k} (c_j / c_k) T_j
            # Then T = sum_{j != k} (1 + c_j/c_k) T_j
            # Let's remove term 27 (index 27)
            k = 27
            if abs(c[k]) > 1e-9:
                # Scale factors for the remaining terms
                scales = c[:k] / c[k]
                # New U, V, W for rank 27
                U_new = U[:, :k].copy()
                V_new = V[:, :k].copy()
                W_new = W[:, :k].copy()
                
                # Update U (or V or W) with the scales
                # T'_j = (1 + scale_j) T_j
                # We can scale U_j by (1 + scale_j)
                for j in range(k):
                    U_new[:, j] *= (1 + scales[j])
                
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 27
    except Exception:
        pass

    # Round to half-integers if close
    def round_half(x):
        return np.round(x * 2) / 2

    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    U_r = round_half(U_out)
    V_r = round_half(V_out)
    W_r = round_half(W_out)
    
    if verify(U_r, V_r, W_r):
        U_out, V_out, W_out = U_r, V_r, W_r

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    construct()
