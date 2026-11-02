import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s wall-clock limit

    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n

    # Strassen rank-7 coefficients for 2x2x2 (verified identity c = w . (u ⊙ v))
    SU = [
        [0, 3],      # M1 = (a11+a22)
        [2, 3],      # M2 = (a21+a22)
        [0],         # M3 = a11
        [3],         # M4 = a22
        [0, 1],      # M5 = (a11+a12)
        [2, 0],      # M6 = (a21-a11)
        [1, 3],      # M7 = (a12-a22)
    ]
    SV = [
        [0, 3],      # (b11+b22)
        [0],         # b11
        [1, 3],      # (b12-b22)
        [2, 0],      # (b21-b11)
        [3],         # b22
        [0, 1],      # (b11+b12)
        [2, 3],      # (b21+b22)
    ]
    SW = [
        [0, 2, 3, 6],     # c11 = M1 - M4 + M5 - M7
        [6, 4],           # c12 = M5 + M6
        [1, 2],           # c21 = M2 + M3
        [0, 1, 2, 5],     # c22 = M1 - M3 + M4 + M6
    ]
    SWV = [
        [1, -1, 1, -1],
        [1, 1],
        [1, 1],
        [1, -1, 1, 1],
    ]
    SUV = [
        [1, 1], [1, 1], [1], [1], [1, 1], [1, -1], [1, -1],
    ]
    SVV = [
        [1, 1], [1], [1, -1], [1, -1], [1], [1, 1], [1, 1],
    ]

    def build_strassen_rank28():
        """Build the standard Strassen-based rank-28 decomposition."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_block(u_map, v_map, w_map, start_col):
            for k in range(7):
                col = start_col + k
                for idx, val in zip(SU[k], SUV[k]):
                    U[u_map[idx], col] += val
                for idx, val in zip(SV[k], SVV[k]):
                    V[v_map[idx], col] += val
                for idx, val in zip(SW[k], SWV[k]):
                    W[w_map[idx], col] += val

        # U flat: A[i,k] -> i*m + k
        # V flat: B[k,j] -> k*p + j
        # W flat: C[i,j] -> j*n + i   (trace convention)

        # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,:]
        u1 = [0, 1, 4, 5]
        v1 = [0, 1, 4, 5]
        w1 = [0, 2, 1, 3]

        # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,:]
        u2 = [0, 1, 4, 5]
        v2 = [2, 3, 6, 7]
        w2 = [0, 2, 1, 3]

        # Block 3: A[:,2:4] * B[2:4,0:2] -> C[0:2,:]
        u3 = [2, 3, 6, 7]
        v3 = [8, 9, 12, 13]
        w3 = [0, 2, 1, 3]

        # Block 4: A[:,2:4] * B[2:4,2:4] -> C[0:2,:]
        u4 = [2, 3, 6, 7]
        v4 = [10, 11, 14, 15]
        w4 = [0, 2, 1, 3]

        add_block(u1, v1, w1, 0)
        add_block(u2, v2, w2, 7)
        add_block(u3, v3, w3, 14)
        add_block(u4, v4, w4, 21)

        return U, V, W

    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        T = np.einsum('ir,kr,jr->ijk', U, V, W)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def try_rank_reduction(U0, V0, W0, target_rank, max_iter=50):
        """
        Attempt to reduce rank by iteratively zeroing out columns that are
        nearly linearly dependent, then re-optimizing.
        """
        if target_rank >= U0.shape[1]:
            return U0, V0, W0, False

        U = U0[:, :target_rank].copy()
        V = V0[:, :target_rank].copy()
        W = W0[:, :target_rank].copy()

        # Try to find a lower-rank exact decomposition by optimization
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def unflatten_params(params):
            nU = n * m * target_rank
            nV = m * p * target_rank
            Uf = params[:nU].reshape(n * m, target_rank)
            Vf = params[nU:nU + nV].reshape(m * p, target_rank)
            Wf = params[nU + nV:].reshape(p * n, target_rank)
            return Uf, Vf, Wf

        def objective(params):
            Uf, Vf, Wf = unflatten_params(params)
            T = np.einsum('ir,kr,jr->ijk', Uf, Vf, Wf)
            return np.sum((T - 1.0) ** 2)

        # Start from the given matrices
        x0 = flatten_params(U, V, W)
        
        # Use L-BFGS-B for optimization
        best_params = x0
        best_obj = objective(x0)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            result = minimize(objective, best_params, method='L-BFGS-B', 
                            options={'maxiter': 100, 'gtol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
            
            if best_obj < 1e-10:
                break
        
        Uf, Vf, Wf = unflatten_params(best_params)
        
        if best_obj < 1e-8:
            return Uf, Vf, Wf, True
        else:
            return U0, V0, W0, False

    # First, build the standard rank-28 decomposition
    U28, V28, W28 = build_strassen_rank28()
    
    # Verify it works
    if not verify_decomposition(U28, V28, W28):
        # Fallback: trivial rank-32 decomposition
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    col = i * m * p + k * p + j
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28

    # Try to reduce rank from 28 down to some lower bound
    # Theoretical lower bound for 2x4x4 is at least 16 (since 2*4*4=32, 
    # and Strassen gives 28, we can try to go lower)
    # Try ranks: 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16
    
    for target_rank in range(27, 15, -1):
        if time.time() > deadline:
            break
        
        U_t, V_t, W_t, success = try_rank_reduction(U28, V28, W28, target_rank, max_iter=20)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_t, V_t, W_t
            break  # Found a lower rank, stop here
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Fall back to the verified rank-28
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28

    if time.time() > deadline:
        # Still return the best valid result we have
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
