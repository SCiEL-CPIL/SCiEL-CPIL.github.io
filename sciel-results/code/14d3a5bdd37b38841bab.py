"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Target: multiply (n x m) by (m x p) -> (n x p)
    # Tensor T_{i,j,k} = 1 if we compute C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: T = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # Note: W index is k*n + i.
    
    # We will try to find a low rank decomposition.
    # Known bounds:
    # Schoolbook: n*m*p = 96
    # Strassen-like for small dims?
    # For 2x2x2, rank is 7.
    # For 3x3x3, rank is 23 (Landsberg/Bläser).
    # For 3x4x8, we can exploit structure.
    
    # Strategy:
    # 1. Try to use a known decomposition for 3x4x4 and extend to 8?
    #    Or 3x2x4 combined?
    #    Since p=8=4*2, maybe split B into two 4x4 blocks?
    #    C = A * [B1, B2] = [A*B1, A*B2]
    #    This requires decomposing 3x4x4 twice.
    #    Rank(3x4x4) is not trivially small. Schoolbook is 48.
    #    If we use Strassen on 3x4x4? 3x4 is not square.
    
    # Let's try a randomized search for a low rank decomposition.
    # We can formulate this as a system of polynomial equations.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    # Let's try to find a decomposition of rank R.
    # Start with R = 96 (schoolbook) and try to reduce?
    # Or start with a guess for a low rank, e.g., 60, 50, etc.
    
    # Better idea: Use the structure of 3x4x8.
    # Split A into 3 rows. Split B into 8 cols.
    # Maybe use a decomposition for 3x4x2?
    # Rank(3x4x2) <= 3*4*2 = 24.
    # Actually, for 2x2x2, rank is 7.
    # For 3x3x2, rank is ?
    
    # Let's try a generic numerical search using random initialization and gradient descent or Gauss-Newton.
    # Variables: U, V, W.
    # Objective: Minimize || T - sum_r U_r V_r W_r ||_F^2
    # But we need exact integer/half-integer solutions.
    
    # Since the task allows half-integers, we can search over a grid or use rounding after optimization.
    
    # Let's try to find a decomposition of rank R by solving the polynomial system.
    # We can use sympy? No, standard library + numpy/scipy.
    
    # Let's try a specific structured decomposition.
    # For 3x4x8, consider splitting the 8 columns into four 2-column blocks.
    # B = [B0, B1, B2, B3] where each Bi is 4x2.
    # C = [A B0, A B1, A B2, A B3]
    # If we can compute A (3x4) * B_i (4x2) efficiently for each i.
    # Rank(3x4x2) <= ?
    # Schoolbook for 3x4x2 is 24.
    # Can we do better?
    # 3x4x2 is equivalent to 3x4x2.
    # Note that 4x2 matrix multiplication can be seen as 2 independent 4x1 multiplications? No.
    
    # Let's look at 3x4x2.
    # A is 3x4, B is 4x2. C is 3x2.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # There are 3*2 = 6 outputs.
    # Schoolbook: 3*4*2 = 24 terms.
    # Can we share computations?
    # Yes, for fixed i, C_{i,0} and C_{i,1} share A_{i,j}.
    # Let's try to find a rank for 3x4x2.
    
    # Let's try to search for a low rank decomposition of 3x4x8 directly using random restarts and local search.
    # We will parameterize U, V, W with random values, then optimize.
    
    # However, a more robust approach for small sizes is to use the known decomposition for 2x2x2 (rank 7) and combine.
    # 3x4x8 = 3 * 4 * 8.
    # We can view this as 3 blocks of 1x4x8? No.
    # We can view this as 4 blocks of 3x1x8? No.
    
    # Let's try to use the decomposition of 2x2x2 (rank 7) to build 3x4x8.
    # 3 = 2+1, 4 = 2+2, 8 = 2+2+2+2.
    # This seems complex.
    
    # Let's try a simpler approach: Randomized search for rank R.
    # We will try R from 60 down to 30.
    # For each R, we initialize U, V, W randomly and try to minimize the error.
    # If the error is small, we round to nearest half-integer and check exactness.
    
    def get_tensor(n, m, p):
        # T_{i, j, k} = 1
        # Flattened index for U: i*m+j
        # Flattened index for V: j*p+k
        # Flattened index for W: k*n+i
        # The tensor is defined by the multiplication:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # This corresponds to a tensor T such that
        # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k?
        # No, the tensor T_{i,j,k} is 1 if we use A_{i,j} and B_{j,k} to produce C_{i,k}.
        # In the decomposition sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r},
        # the term U_{i,j,r} V_{j,k,r} W_{k,i,r} contributes to C_{i,k}.
        # The condition is that for each i,k, sum_j (sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}) A_{i,j} B_{j,k} = C_{i,k}.
        # This requires that for each i,j,k:
        # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
        # Wait, W depends on i and k.
        # Let's check the indices in the prompt:
        # U: (n*m, rank) -> index i*m+j
        # V: (m*p, rank) -> index j*p+k
        # W: (p*n, rank) -> index k*n+i
        # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This sum must equal 1 for all i, j, k.
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    T = get_tensor(n, m, p)
    
    def decompose_rank(R, max_iters=100, seed=0):
        rng = np.random.RandomState(seed)
        # Initialize U, V, W
        # To avoid symmetry breaking issues, we can initialize with random values.
        U = rng.randn(n*m, R)
        V = rng.randn(m*p, R)
        W = rng.randn(p*n, R)
        
        # We want to minimize sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
        # This is a 3-way tensor factorization problem.
        # We can use ALS (Alternating Least Squares) or gradient descent.
        # ALS is usually faster and more stable.
        
        # Precompute indices
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_flat = u_idx.flatten()
        v_flat = v_idx.flatten()
        w_flat = w_idx.flatten()
        target = np.ones(n*m*p)
        
        # ALS
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            # For each row u in U, we need to solve for U[u, :]
            # The terms involving U[u, :] are sum_r U[u,r] V[v,r] W[w,r] for all (v,w) such that u_idx = u.
            # Let M_u be a matrix where M_u[v', w'] = V[v', r] W[w', r] summed over r? No.
            # The equation is:
            # sum_r U[u,r] (V[v,r] W[w,r]) = 1
            # Let Z_u[v,w] = sum_r V[v,r] W[w,r]
            # Then sum_r U[u,r] Z_u[v,w,r] = 1 is not linear in U.
            # Wait, Z depends on r.
            # Let's rewrite:
            # For a fixed u, we have a set of equations for each (v, w) where u_idx = u.
            # sum_r U[u,r] V[v,r] W[w,r] = 1
            # Let A_u be a matrix of size (num_pairs, R) where each row is V[v,r] * W[w,r] for a pair (v,w).
            # Then A_u @ U[u, :] = 1 (vector of ones)
            
            # We need to group pairs by u.
            # This might be slow if done naively.
            # Given small sizes, we can do it.
            
            # Precompute groups
            groups_u = {}
            for idx in range(len(u_flat)):
                u = u_flat[idx]
                if u not in groups_u:
                    groups_u[u] = []
                groups_u[u].append(idx)
                
            groups_v = {}
            for idx in range(len(v_flat)):
                v = v_flat[idx]
                if v not in groups_v:
                    groups_v[v] = []
                groups_v[v].append(idx)
                
            groups_w = {}
            for idx in range(len(w_flat)):
                w = w_flat[idx]
                if w not in groups_w:
                    groups_w[w] = []
                groups_w[w].append(idx)

            # Update U
            for u in range(n*m):
                if u not in groups_u: continue
                idxs = groups_u[u]
                # Build A matrix
                A = np.zeros((len(idxs), R))
                for k_idx_local, idx in enumerate(idxs):
                    v = v_flat[idx]
                    w = w_flat[idx]
                    A[k_idx_local, :] = V[v, :] * W[w, :]
                # Solve A @ U[u, :] = 1
                # Least squares
                U[u, :] = np.linalg.lstsq(A, np.ones(len(idxs)), rcond=None)[0]
                
            # Update V
            for v in range(m*p):
                if v not in groups_v: continue
                idxs = groups_v[v]
                A = np.zeros((len(idxs), R))
                for k_idx_local, idx in enumerate(idxs):
                    u = u_flat[idx]
                    w = w_flat[idx]
                    A[k_idx_local, :] = U[u, :] * W[w, :]
                V[v, :] = np.linalg.lstsq(A, np.ones(len(idxs)), rcond=None)[0]
                
            # Update W
            for w in range(p*n):
                if w not in groups_w: continue
                idxs = groups_w[w]
                A = np.zeros((len(idxs), R))
                for k_idx_local, idx in enumerate(idxs):
                    u = u_flat[idx]
                    v = v_flat[idx]
                    A[k_idx_local, :] = U[u, :] * V[v, :]
                W[w, :] = np.linalg.lstsq(A, np.ones(len(idxs)), rcond=None)[0]
                
        # Check error
        # Reconstruct
        # We can compute the error efficiently
        # err = 0
        # for idx in range(len(u_flat)):
        #     val = np.dot(U[u_flat[idx]], V[v_flat[idx]] * W[w_flat[idx]])
        #     err += (val - 1)**2
        
        # Vectorized check
        # U_u = U[u_flat] # (N, R)
        # V_v = V[v_flat] # (N, R)
        # W_w = W[w_flat] # (N, R)
        # vals = np.sum(U_u * V_v * W_w, axis=1)
        # err = np.sum((vals - 1)**2)
        
        U_u = U[u_flat]
        V_v = V[v_flat]
        W_w = W[w_flat]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        err = np.sum((vals - 1)**2)
        
        return U, V, W, err

    best_U, best_V, best_W, best_err = None, None, None, float('inf')
    best_R = n*m*p
    
    # Try ranks from low to high
    # Start with a reasonable guess. 3x4x8 schoolbook is 96.
    # Let's try 60, 70, 80, 90, 96.
    # Or try to find the minimum.
    
    for R in range(60, 97):
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds
        for seed in range(5):
            U, V, W, err = decompose_rank(R, max_iters=50, seed=seed)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_R = R
                # If error is very small, we might have found a solution
                if err < 1e-10:
                    break
        if best_err < 1e-10:
            break
            
    # If we found a low error solution, round to half-integers and check
    if best_U is not None and best_err < 1e-6:
        # Round to nearest 0.5
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        
        # Check exactness
        U_u = U_int[u_flat]
        V_v = V_int[v_flat]
        W_w = W_int[w_flat]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        if np.allclose(vals, 1.0, atol=1e-9):
            # Return integer/half-integer matrices
            # Convert to list of lists of floats/ints
            def to_list(mat):
                return [[float(x) for x in row] for row in mat]
            return {"U": to_list(U_int), "V": to_list(V_int), "W": to_list(W_int)}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
