import numpy as np
import time
import random
from scipy.optimize import least_squares

START_TIME = time.time()
BUDGET_SECONDS = 145.0  # Safety margin before 150s limit

def check_time():
    return time.time() - START_TIME < BUDGET_SECONDS

def construct():
    n, m, p = 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Precompute indices for the 60 non-zero entries of the tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
                
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    # Objective function for least_squares
    # We want to find U, V, W such that sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    # This is a non-linear system.
    # Let's try to solve for rank 59.
    # If that fails, try rank 58, etc.
    # Fallback to 60.
    
    def objective(params, rank):
        U_flat = params[:rank * dim_u]
        V_flat = params[rank * dim_u : rank * dim_u + rank * dim_v]
        W_flat = params[rank * dim_u + rank * dim_v :]
        
        U = U_flat.reshape(rank, dim_u).T # (dim_u, rank)
        V = V_flat.reshape(rank, dim_v).T # (dim_v, rank)
        W = W_flat.reshape(rank, dim_w).T # (dim_w, rank)
        
        # Compute errors for all 60 entries
        # U[idx_u, :] shape (60, rank)
        # V[idx_v, :] shape (60, rank)
        # W[idx_w, :] shape (60, rank)
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        
        # Sum over rank dimension
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        return vals - 1.0

    def try_find_decomposition(rank, num_restarts=5):
        size = rank * (dim_u + dim_v + dim_w)
        
        for r in range(num_restarts):
            if not check_time():
                return None
                
            # Initialization:
            # Start with schoolbook-like structure?
            # Or random?
            # Random with small magnitude often works for least_squares if the landscape is well-behaved.
            # But tensor rank is ill-conditioned.
            # Let's try initializing with random integers in {-1, 0, 1} scaled down?
            # Or just random normal.
            
            x0 = np.random.randn(size) * 0.1
            
            try:
                # Use 'lm' (Levenberg-Marquardt) which is good for non-linear least squares
                # But it requires number of params >= number of residuals.
                # Here: params = rank * 47. Residuals = 60.
                # If rank=59, params = 59*47 = 2773. Residuals = 60.
                # So it is underdetermined. LM might struggle or need 'trf'.
                # Let's use 'trf' (Trust Region Reflective) or just 'lm' if it works.
                # Actually, 'lm' fails if m < n. Here m=60, n=2773. So 'lm' will fail.
                # We must use 'trf' or 'dogbox'.
                
                result = least_squares(objective, x0, args=(rank,), method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if result.cost < 1e-8:
                # Extract U, V, W
                U_flat = result.x[:rank * dim_u]
                V_flat = result.x[rank * dim_u : rank * dim_u + rank * dim_v]
                W_flat = result.x[rank * dim_u + rank * dim_v :]
                
                U = U_flat.reshape(rank, dim_u).T
                V = V_flat.reshape(rank, dim_v).T
                W = W_flat.reshape(rank, dim_w).T
                
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                U_sel = U_r[idx_u, :]
                V_sel = V_r[idx_v, :]
                W_sel = W_r[idx_w, :]
                vals = np.sum(U_sel * V_sel * W_sel, axis=1)
                
                if np.all(np.abs(vals - 1.0) < 1e-9):
                    # Also check if any other entries are non-zero?
                    # The tensor is sparse. We only constrained the 60 entries to be 1.
                    # We must ensure other entries are 0.
                    # However, the problem statement says "exactly decompose the matrix multiplication tensor".
                    # This implies T_reconstructed == T_original.
                    # T_original has 1s at the 60 positions and 0s elsewhere.
                    # Our optimization only forced the 60 positions to be 1.
                    # We must check the other positions.
                    
                    # How many other positions?
                    # Total positions = dim_u * dim_v * dim_w = 12 * 20 * 15 = 3600.
                    # We checked 60.
                    # We need to check the remaining 3540.
                    
                    # This is expensive if done naively.
                    # But we can vectorize.
                    # Iterate over all i, j, k and check if (i,j,k) is a valid multiplication index.
                    # If not, sum should be 0.
                    
                    # Let's generate all triples
                    all_u = np.arange(dim_u)
                    all_v = np.arange(dim_v)
                    all_w = np.arange(dim_w)
                    
                    # This might be too slow in Python loop.
                    # Let's check a sample or all if fast enough.
                    # 3600 iterations of vectorized dot products is fast in NumPy.
                    
                    # Actually, we can just check all 3600.
                    # Create indices for all combinations
                    # This is a meshgrid.
                    # 12 * 20 * 15 = 3600.
                    # We can construct the arrays.
                    
                    # But wait, if we find a solution where other entries are non-zero, it's invalid.
                    # Given the symmetry and sparsity, it's likely that if the 60 are 1, the others are close to 0?
                    # Not necessarily.
                    
                    # Let's check all.
                    valid = True
                    # To speed up, we can check in batches or just loop.
                    # 3600 is small.
                    
                    # We need to identify which (u_idx, v_idx, w_idx) are "valid" (should be 1) and which are not (should be 0).
                    # Valid: u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i for some i,j,k.
                    
                    # Let's create a set of valid tuples.
                    valid_tuples = set()
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                valid_tuples.add((i*m+j, j*p+k, k*n+i))
                    
                    # Iterate over all u, v, w
                    # This is 12*20*15 = 3600.
                    # We can vectorize this check too.
                    
                    # Let's just check the 60 valid ones are 1 (done above).
                    # And check a subset of invalid ones?
                    # No, must be exact.
                    
                    # Let's check all 3600.
                    # We can do this:
                    # U_all = U_r (12, R)
                    # V_all = V_r (20, R)
                    # W_all = W_r (15, R)
                    
                    # We need to compute sum_r U[u,r] V[v,r] W[w,r] for all u,v,w.
                    # This is a 3-tensor contraction.
                    # We can do:
                    # T_reconstructed = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    # This creates a 12x20x15 tensor.
                    # Then check if it matches the target tensor.
                    
                    T_rec = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    
                    # Construct target tensor
                    T_target = np.zeros((dim_u, dim_v, dim_w))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    
                    if np.allclose(T_rec, T_target, atol=1e-9):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None

    # Try ranks from 59 down to 60?
    # Start with 59. If found, return.
    # If not, try 58?
    # Given the difficulty, 59 is unlikely.
    # But let's try.
    
    # To save time, we can limit restarts.
    # If rank 59 fails quickly, we return 60.
    
    # Time budget management:
    # We have ~145s.
    # Let's give 10s to rank 59 search.
    # If not found, return 60.
    
    # Actually, least_squares might be slow.
    # Let's try just 1-2 restarts for rank 59 to be safe.
    
    res = try_find_decomposition(59, num_restarts=2)
    if res is not None:
        return res
        
    # Fallback to Rank 60 (Schoolbook)
    rank = 60
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
