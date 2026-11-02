"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we think of it as
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In CP form: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # T[i*m+j, j*p+k, k*n+i] = 1 for the "multiplication" pattern
    # Actually T is defined by: for all i,j,k, the entry is 1 (it's the structure tensor)
    # The multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no - the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we identify the tensor such that
    # C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # This is because the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right arrangement)
    
    # Actually, let me be more careful. The matrix multiplication tensor is:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices correspond to
    # A has indices (i,j), B has indices (j,k), C has indices (i,k)
    # So M[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # This is the all-ones 3-tensor of size n x m x p!
    
    # Wait, that can't be right. Let me reconsider.
    # The structure tensor for multiplying an n x m matrix by an m x p matrix
    # is a tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # Its entries are: T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    # So it's a "generalized identity" tensor.
    
    # In flat index terms:
    # U has rows indexed by (i,j) for i in [0,n), j in [0,m) -> n*m rows
    # V has rows indexed by (j,k) for j in [0,m), k in [0,p) -> m*p rows  
    # W has rows indexed by (i,k) for i in [0,n), k in [0,p) -> n*p rows
    # T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'
    
    # So the tensor is: for each triple (i,j,k), there's exactly one nonzero entry
    # at position ((i,j), (j,k), (i,k)) with value 1
    # Total of n*m*p = 63 nonzero entries, each = 1
    
    # The CP decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # means: for each (a,b,c) in the tensor, sum over d of U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
    
    # Let's build the tensor and try to find a low-rank decomposition
    
    nm = n * m  # 9
    mp = m * p  # 21
    np_ = n * p  # 21
    
    # Build the target tensor as a list of (index_a, index_b, index_c, value)
    # where index_a = i*m + j, index_b = j*p + k, index_c = k*n + i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                tensor_entries.append((idx_a, idx_b, idx_c, 1))
    
    # Now try to find a rank-r decomposition
    # Strategy: use random initialization and iterative refinement (like ALS for CP)
    # Or use the "rank-1 peeling" approach
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Try various ranks from the lower bound upward
    # Lower bound: max(nm, mp, np_) = max(9, 21, 21) = 21
    lower_bound = max(nm, mp, np_)
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        if best_decomp is not None and r >= best_rank:
            break
            
        # Try to find a rank-r decomposition
        decomp = _find_cp_decomposition(tensor_entries, nm, mp, np_, r, deadline - 15)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            if r == lower_bound:
                break  # Can't do better than the lower bound
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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

def _find_cp_decomposition(tensor_entries, nm, mp, np_, r, deadline):
    """
    Try to find a rank-r CP decomposition of the tensor.
    Uses random restarts with ALS-style optimization.
    """
    num_restarts = 20
    max_iters = 500
    tol = 1e-10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
        
        # Random initialization
        rng = np.random.default_rng(seed=42 + restart)
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((np_, r))
        
        # Normalize to avoid numerical issues
        for c in range(r):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
        
        # ALS optimization
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each row a, U[a,:] should satisfy
            # sum_c U[a,c] * (V ⊙ W)[a,c] = target
            # where (V ⊙ W)[a,c] = sum over (b,c_idx) where tensor[a,b,c_idx] nonzero of V[b,c]*W[c_idx,c]
            # This is a least squares problem
            
            # Build the system for U
            # For each a in range(nm):
            #   sum_c U[a,c] * S_VW[a,c] = t_a
            # where S_VW[a,c] = sum_{(b,c_idx): tensor[a,b,c_idx]=1} V[b,c] * W[c_idx, c]
            # and t_a = sum_{(b,c_idx): tensor[a,b,c_idx]=1} 1  (since all values are 1)
            
            # More carefully: the tensor equation is
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            # For fixed a, this gives a system in U[a,:]
            
            # Let's use a different approach: direct least squares
            # For each c, we have a rank-1 tensor u_c ⊗ v_c ⊗ w_c
            # The sum should equal the target tensor
            
            # Compute current approximation error
            # And do gradient steps or ALS
            
            # ALS step for U:
            # For each a, solve: minimize sum_{(b,c_idx) in support(a)} (T[a,b,c_idx] - sum_d U[a,d] V[b,d] W[c_idx,d])^2
            # This is linear in U[a,:]:
            # Let M_U[a] be the matrix where M_U[a][d] = sum_{(b,c_idx) in support(a)} V[b,d] * W[c_idx,d]
            # Let rhs[a] = sum_{(b,c_idx) in support(a)} T[a,b,c_idx]
            # Then U[a,:] = M_U[a]^{-1} rhs[a] (least squares)
            
            # Precompute support structure
            # For each a, which (b, c_idx) pairs have T[a,b,c_idx] = 1?
            
            # Build support lists
            support_a = [[] for _ in range(nm)]
            support_b = [[] for _ in range(mp)]
            support_c = [[] for _ in range(np_)]
            for (a, b, c_idx, val) in tensor_entries:
                support_a[a].append((b, c_idx))
                support_b[b].append((a, c_idx))
                support_c[c_idx].append((a, b))
            
            # Update U
            for a in range(nm):
                if not support_a[a]:
                    U[a, :] = 0
                    continue
                # Build M_U[a] : r x r matrix
                # M_U[a][d1, d2] = sum_{(b,c_idx) in support(a)} V[b,d1] * W[c_idx,d2]
                # rhs[a][d] = sum_{(b,c_idx) in support(a)} T[a,b,c_idx] * V[b,d] * W[c_idx,d]
                # Wait, this is wrong. Let me reconsider.
                
                # The equation is: for each (b, c_idx) in support(a):
                #   sum_d U[a,d] * V[b,d] * W[c_idx,d] = 1
                # This is a linear system in U[a,:]:
                #   A_U[a] * U[a,:] = rhs_U[a]
                # where A_U[a] is a |support(a)| x r matrix with entries V[b,d]*W[c_idx,d]
                # and rhs_U[a] is a |support(a)| vector of 1s
                
                if len(support_a[a]) == 0:
                    U[a, :] = 0
                    continue
                    
                A_U = np.zeros((len(support_a[a]), r))
                for idx, (b, c_idx) in enumerate(support_a[a]):
                    A_U[idx, :] = V[b, :] * W[c_idx, :]
                rhs_U = np.ones(len(support_a[a]))
                
                # Solve least squares
                try:
                    U[a, :], residuals, rank_u, sv = np.linalg.lstsq(A_U, rhs_U, rcond=None)
                except:
                    U[a, :] = 0
            
            # Update V
            for b in range(mp):
                if not support_b[b]:
                    V[b, :] = 0
                    continue
                A_V = np.zeros((len(support_b[b]), r))
                for idx, (a, c_idx) in enumerate(support_b[b]):
                    A_V[idx, :] = U[a, :] * W[c_idx, :]
                rhs_V = np.ones(len(support_b[b]))
                try:
                    V[b, :], residuals, rank_v, sv = np.linalg.lstsq(A_V, rhs_V, rcond=None)
                except:
                    V[b, :] = 0
            
            # Update W
            for c_idx in range(np_):
                if not support_c[c_idx]:
                    W[c_idx, :] = 0
                    continue
                A_W = np.zeros((len(support_c[c_idx]), r))
                for idx, (a, b) in enumerate(support_c[c_idx]):
                    A_W[idx, :] = U[a, :] * V[b, :]
                rhs_W = np.ones(len(support_c[c_idx]))
                try:
                    W[c_idx, :], residuals, rank_w, sv = np.linalg.lstsq(A_W, rhs_W, rcond=None)
                except:
                    W[c_idx, :] = 0
            
            # Check convergence
            # Compute error
            error = 0.0
            for (a, b, c_idx, val) in tensor_entries:
                approx = np.dot(U[a, :], V[b, :] * W[c_idx, :])
                error += (approx - val) ** 2
            error = np.sqrt(error / len(tensor_entries))
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        # Final check: is the decomposition exact?
        max_err = 0.0
        for (a, b, c_idx, val) in tensor_entries:
            approx = np.dot(U[a, :], V[b, :] * W[c_idx, :])
            max_err = max(max_err, abs(approx - val))
        
        # Also check that entries NOT in the tensor are zero
        # For all (a,b,c_idx) not in tensor_entries, the sum should be 0
        # But this is a large check. Let's do a random sample.
        sampled_err = 0.0
        num_samples = min(1000, nm * mp * np_ - len(tensor_entries))
        for _ in range(num_samples):
            a = rng.integers(0, nm)
            b = rng.integers(0, mp)
            c_idx = rng.integers(0, np_)
            # Check if this is in the tensor
            in_tensor = any(ta == a and tb == b and tc == c_idx for (ta, tb, tc, tv) in tensor_entries)
            if not in_tensor:
                approx = np.dot(U[a, :], V[b, :] * W[c_idx, :])
                sampled_err = max(sampled_err, abs(approx))
        
        if max_err < 1e-6 and sampled_err < 1e-6:
            # Convert to lists
            U_list = [[float(U[a, c]) for c in range(r)] for a in range(nm)]
            V_list = [[float(V[b, c]) for c in range(r)] for b in range(mp)]
            W_list = [[float(W[c_idx, c]) for c in range(r)] for c_idx in range(np_)]
            return (U_list, V_list, W_list)
    
    return None
# EVOLVE-BLOCK-END
