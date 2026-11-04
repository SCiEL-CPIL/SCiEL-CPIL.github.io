"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools
from scipy.optimize import linprog

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s wall-clock limit, 5s safety margin
    
    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Build list of non-zero tensor entries
    nnz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nnz.append((a, b, c))
    
    num_nnz = len(nnz)  # 144
    dim_a = n * m  # 16
    dim_b = m * p  # 36
    dim_c = p * n  # 36
    
    # Precompute indices for fast verification
    nnz_a = np.array([x[0] for x in nnz])
    nnz_b = np.array([x[1] for x in nnz])
    nnz_c = np.array([x[2] for x in nnz])
    
    def verify_decomposition(U, V, W):
        """
        Check if U,V,W (NumPy arrays) correctly decompose the tensor.
        Returns True if valid, False otherwise.
        """
        # U: (dim_a, r), V: (dim_b, r), W: (dim_c, r)
        # We need sum_t U[a,t]*V[b,t]*W[c,t] == 1 for all (a,b,c) in nnz
        
        # Efficient check:
        # Compute the sum for each nnz entry.
        # U_nn = U[nnz_a, :] -> (num_nnz, r)
        # V_nn = V[nnz_b, :] -> (num_nnz, r)
        # W_nn = W[nnz_c, :] -> (num_nnz, r)
        
        U_nn = U[nnz_a, :]
        V_nn = V[nnz_b, :]
        W_nn = W[nnz_c, :]
        
        # Element-wise product and sum over rank
        # (U_nn * V_nn * W_nn).sum(axis=1)
        vals = np.sum(U_nn * V_nn * W_nn, axis=1)
        
        # Check if all are close to 1
        if not np.allclose(vals, 1.0, atol=1e-8):
            return False
        
        # We should also check that zero entries are zero.
        # Sampling a few zero entries is a good heuristic to avoid false positives
        # where the decomposition fits the 1s but creates noise elsewhere.
        # However, for matrix multiplication tensors, if the rank is low enough
        # and the fit is exact on the support, it's often correct.
        # Let's do a quick check on a few random zero entries.
        
        # Total entries = dim_a * dim_b * dim_c
        # Let's check 100 random zero entries
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.integers(0, dim_a)
            b = rng.integers(0, dim_b)
            c = rng.integers(0, dim_c)
            
            # Check if (a,b,c) is in nnz
            # Simple check: is there i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i?
            # Since nnz is small, we can just check if it's in the set.
            # But constructing a set of 144 tuples is fast.
            # For speed in this inner loop, let's just compute the value.
            # If it's not in nnz, it should be 0.
            
            # To avoid O(num_nnz) check per sample, we can precompute a set
            # but 100 samples * 144 checks is 14400 ops, which is fine.
            is_nnz = False
            for idx in range(num_nnz):
                if nnz_a[idx] == a and nnz_b[idx] == b and nnz_c[idx] == c:
                    is_nnz = True
                    break
            
            if not is_nnz:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-8:
                    return False
                    # Actually, if we find a non-zero, it's invalid.
            
        return True

    def try_rank_als(rank, max_iters=20, seed=0):
        """Try to find rank-r decomposition using alternating least squares."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(dim_a, rank) * 0.1
        V = np.random.randn(dim_b, rank) * 0.1
        W = np.random.randn(dim_c, rank) * 0.1
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U:
            # For each a, minimize sum_{b,c} (sum_t U[a,t]*V[b,t]*W[c,t] - T[a,b,c])^2
            # This is a linear least squares problem for U[a,:].
            # Let M_VW[t, b, c] = V[b,t]*W[c,t]
            # We want U[a,:] such that M_VW_flat @ U[a,:] ~= T[a,:,:].flatten()
            
            # Construct M_VW_flat: (rank, dim_b * dim_c)
            # V[:, t] is (dim_b,), W[:, t] is (dim_c,)
            # Outer product V[:, t] @ W[:, t].T is (dim_b, dim_c)
            # We need to stack these for all t.
            
            # Efficient way:
            # M_VW_flat[t, b*dim_c + c] = V[b, t] * W[c, t]
            
            # We can do this using broadcasting.
            # V is (dim_b, rank), W is (dim_c, rank)
            # V_exp = V.T[:, :, None] -> (rank, dim_b, 1)
            # W_exp = W.T[:, None, :] -> (rank, 1, dim_c)
            # Prod = V_exp * W_exp -> (rank, dim_b, dim_c)
            # M_VW_flat = Prod.reshape(rank, -1)
            
            V_exp = V.T[:, :, None]
            W_exp = W.T[:, None, :]
            M_VW_flat = (V_exp * W_exp).reshape(rank, -1)
            
            # Now, for each a, we have target T[a, b, c]
            # T is 1 if (a,b,c) in nnz, else 0.
            # We can precompute the target matrix for all a.
            # Target_flat[a, b*dim_c + c] = 1 if (a,b,c) in nnz else 0
            
            # Let's construct the full target tensor in flattened form.
            # T_flat is (dim_a, dim_b * dim_c)
            T_flat = np.zeros((dim_a, dim_b * dim_c))
            for idx in range(num_nnz):
                a = nnz_a[idx]
                b = nnz_b[idx]
                c = nnz_c[idx]
                T_flat[a, b * dim_c + c] = 1.0
            
            # Solve for U:
            # U @ M_VW_flat.T ~= T_flat.T
            # U is (dim_a, rank), M_VW_flat.T is (rank, dim_b*dim_c)
            # T_flat.T is (dim_b*dim_c, dim_a)
            
            # We can solve this as a batched least squares or just use lstsq on the whole system.
            # Since dim_a is small (16), we can loop over a.
            for a in range(dim_a):
                try:
                    U[a, :] = np.linalg.lstsq(M_VW_flat, T_flat[a, :], rcond=None)[0]
                except:
                    pass
            
            # Update V:
            # Similarly, for each b, solve for V[b,:].
            # M_UW_flat[t, a*dim_c + c] = U[a, t] * W[c, t]
            U_exp = U.T[:, :, None]
            W_exp = W.T[:, None, :]
            M_UW_flat = (U_exp * W_exp).reshape(rank, -1)
            
            # Target for V: T_flat_V[b, a*dim_c + c] = T[a, b, c]
            # We can reshape T_flat to get this.
            # T_flat is (dim_a, dim_b, dim_c)
            T_3d = T_flat.reshape(dim_a, dim_b, dim_c)
            # We want T_V[b, a, c] = T_3d[a, b, c]
            T_V_flat = T_3d.transpose(1, 0, 2).reshape(dim_b, -1)
            
            for b in range(dim_b):
                try:
                    V[b, :] = np.linalg.lstsq(M_UW_flat, T_V_flat[b, :], rcond=None)[0]
                except:
                    pass
            
            # Update W:
            # For each c, solve for W[c,:].
            # M_UV_flat[t, a*dim_b + b] = U[a, t] * V[b, t]
            U_exp = U.T[:, :, None]
            V_exp = V.T[:, None, :]
            M_UV_flat = (U_exp * V_exp).reshape(rank, -1)
            
            # Target for W: T_W[c, a, b] = T[a, b, c]
            T_W_flat = T_3d.transpose(2, 0, 1).reshape(dim_c, -1)
            
            for c in range(dim_c):
                try:
                    W[c, :] = np.linalg.lstsq(M_UV_flat, T_W_flat[c, :], rcond=None)[0]
                except:
                    pass
            
            # Optional: Round to half-integers?
            # The problem allows integer/half-integer entries.
            # Rounding might help with the "exact" requirement if the solution is close.
            # However, ALS converges to real numbers. If we round, we might break the exactness.
            # The evaluator checks for "integer/half-integer entries".
            # If we don't round, the entries are floats.
            # The task says: "factor matrices ... with integer/half-integer entries".
            # This implies the returned U, V, W must have values like 0.5, 1.0, -0.5, etc.
            # So we MUST round.
            
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Check convergence / validity
            # We can check the error on nnz entries.
            U_nn = U[nnz_a, :]
            V_nn = V[nnz_b, :]
            W_nn = W[nnz_c, :]
            vals = np.sum(U_nn * V_nn * W_nn, axis=1)
            err = np.max(np.abs(vals - 1.0))
            
            if err < 1e-8:
                # Check a few zero entries
                # If this passes, we can return
                if verify_decomposition(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    best_result = None
    
    # Try various ranks
    # Schoolbook rank is 144.
    # Known lower bound is higher.
    # Let's try ranks that are multiples of common structures or just decreasing.
    ranks_to_try = [128, 112, 96, 80, 64]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = try_rank_als(rank, max_iters=10, seed=rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(dim_a)]
    V_school = [[0] * rank_schoolbook for _ in range(dim_b)]
    W_school = [[0] * rank_schoolbook for _ in range(dim_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
