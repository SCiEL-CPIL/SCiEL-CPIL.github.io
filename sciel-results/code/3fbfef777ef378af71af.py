"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x5 matmul."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time
import random

N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N
TENSOR_SIZE = N * M * P  # 75 target tensor entries

# Precompute the 3D structure of the matmul tensor.
# For each (i, j, k) in [N]x[M]x[P], the term U[i*M+j] * V[j*P+k] * W[k*N+i] should equal 1.
# So the tensor T[i,j,k] = 1 for all valid (i,j,k).
# A decomposition with factor matrices U (NM x r), V (MP x r), W (PN x r) is valid iff
#   sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] == 1  for all (i,j,k).

# Precompute index maps
def _idx_U(i, j): return i * M + j
def _idx_V(j, k): return j * P + k
def _idx_W(k, i): return k * N + i

# List of all (i,j,k) triples
TRIPLES = [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]

def _target_tensor():
    """Return the target tensor as a flat array of ones (all entries = 1)."""
    return np.ones(TENSOR_SIZE, dtype=np.float64)

def _compute_residual(U, V, W, r):
    """Compute the tensor produced by the decomposition minus the target (should be 0)."""
    # For each (i,j,k): sum_c U[iM+j,c] * V[jP+k,c] * W[kN+i,c] - 1
    res = np.empty(TENSOR_SIZE, dtype=np.float64)
    for t, (i, j, k) in enumerate(TRIPLES):
        ui = U[_idx_U(i, j), :]
        vk = V[_idx_V(j, k), :]
        wk = W[_idx_W(k, i), :]
        res[t] = np.dot(ui, vk * wk) - 1.0
    return res

def _verify(U, V, W, tol=1e-8):
    """Check if the decomposition is valid."""
    r = U.shape[1]
    res = _compute_residual(U, V, W, r)
    return float(np.max(np.abs(res))) < tol

def _schoolbook():
    """Rank-75 schoolbook decomposition."""
    r = TENSOR_SIZE
    U = np.zeros((NM, r), dtype=np.float64)
    V = np.zeros((MP, r), dtype=np.float64)
    W = np.zeros((PN, r), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[_idx_U(i, j), col] = 1.0
                V[_idx_V(j, k), col] = 1.0
                W[_idx_W(k, i), col] = 1.0
                col += 1
    return U, V, W

def _try_eliminate(U, V, W, r, deadline):
    """
    Try to reduce rank by one: find a column index c such that we can set U[:,c]=0, V[:,c]=0, W[:,c]=0
    and compensate by adjusting other columns. This is hard, so instead we use a different strategy:
    Try to find a linear dependence among the rank-1 terms.
    
    Alternative: use the "tensor rank reduction" via finding a column that can be expressed
    as a combination of others. For now, we use a simpler approach:
    Randomly perturb and try to find a lower-rank solution via optimization.
    """
    pass

def _local_search(U, V, W, r, deadline, n_iter=200):
    """
    Local search: randomly pick a column and try to zero it out by adjusting others.
    Strategy: for a candidate column c, try to find scalars alpha_c, beta_c, gamma_c and
    adjustments to other columns such that the tensor is preserved.
    
    Simpler approach: use gradient-based optimization to minimize the residual while
    encouraging sparsity in one column.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_residual = float(np.max(np.abs(_compute_residual(U, V, W, r))))
    
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        # Pick a random column to try to eliminate
        c = random.randint(0, r - 1)
        
        # Try: set column c to zero, and see if we can adjust other columns to compensate.
        # This is a linear system problem. For each (i,j,k), we need:
        #   sum_{c' != c} U[iM+j, c'] V[jP+k, c'] W[kN+i, c'] = 1
        # The current contribution of column c is U[iM+j,c] V[jP+k,c] W[kN+i,c].
        # If we set column c to zero, the residual for each (i,j,k) becomes:
        #   -U[iM+j,c] V[jP+k,c] W[kN+i,c]
        # We need to distribute this correction across other columns.
        
        # This is complex. Instead, let's try a simpler heuristic:
        # If column c has a very small norm, just zero it out and check.
        col_norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if col_norm < 1e-6:
            U[:, c] = 0.0
            V[:, c] = 0.0
            W[:, c] = 0.0
            res = _compute_residual(U, V, W, r)
            max_res = float(np.max(np.abs(res)))
            if max_res < 1e-6:
                # Column c was already negligible, remove it
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                r -= 1
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_residual = max_res
                continue
        
        # Try a small perturbation approach:
        # Randomly perturb column c and see if residual decreases
        if random.random() < 0.3:
            U_save, V_save, W_save = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
            scale = random.uniform(-0.1, 0.1)
            U[:, c] += scale * random.randn(NM)
            V[:, c] += scale * random.randn(MP)
            W[:, c] += scale * random.randn(PN)
            res = _compute_residual(U, V, W, r)
            max_res = float(np.max(np.abs(res)))
            if max_res < best_residual:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_residual = max_res
            else:
                U[:, c], V[:, c], W[:, c] = U_save, V_save, W_save
    
    return best_U, best_V, best_W, best_residual

def _greedy_construct(r, deadline):
    """
    Construct a rank-r decomposition by greedily adding rank-1 terms.
    Start with zero residual, and for each of r steps, find the best rank-1 term
    to add to minimize the remaining residual.
    """
    target = np.ones(TENSOR_SIZE, dtype=np.float64)
    current = np.zeros(TENSOR_SIZE, dtype=np.float64)
    
    U = np.zeros((NM, r), dtype=np.float64)
    V = np.zeros((MP, r), dtype=np.float64)
    W = np.zeros((PN, r), dtype=np.float64)
    
    for c in range(r):
        if time.time() > deadline:
            break
        residual = target - current
        
        # Find the best rank-1 term u * v * w that best matches the residual.
        # This is a 3-way tensor approximation problem. Use a simple greedy:
        # For each possible combination of "dominant" indices, try to fit.
        
        # Simple approach: for each (i,j,k) with largest residual, try to fit a rank-1 term
        # that covers it. A rank-1 term u*v*w where u[iM+j], v[jP+k], w[kN+i] are non-zero
        # will affect all (i',j',k') where u[i'M+j'], v[j'P+k'], w[k'N+i'] are non-zero.
        
        # Greedy: pick the (i,j,k) with largest |residual|, and set up a rank-1 term
        # that is 1 at (i,j,k) and 0 elsewhere (as much as possible).
        
        # Actually, a rank-1 term u*v*w where u is supported on {iM+j}, v on {jP+k}, w on {kN+i}
        # will only affect (i,j,k) if u is a unit vector at iM+j, v at jP+k, w at kN+i.
        # But then it only affects (i,j,k) and not others. This gives the schoolbook.
        
        # To get a better decomposition, we need rank-1 terms that overlap.
        # Let's try: for each step, randomly generate a rank-1 term and check if it helps.
        
        # Better: use the fact that we want sum of rank-1 terms = target.
        # Try random rank-1 terms and keep those that reduce the residual.
        
        best_improvement = 0
        best_u, best_v, best_w = None, None, None
        best_res = residual.copy()
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Generate a random rank-1 term
            # u is a random vector in R^NM, v in R^MP, w in R^PN
            # Scale so that the term has reasonable magnitude
            u = random.randn(NM)
            v = random.randn(MP)
            w = random.randn(PN)
            
            # Compute the tensor contribution of this rank-1 term
            contrib = np.zeros(TENSOR_SIZE, dtype=np.float64)
            for t, (i, j, k) in enumerate(TRIPLES):
                contrib[t] = u[_idx_U(i, j)] * v[_idx_V(j, k)] * w[_idx_W(k, i)]
            
            # Try to find the best scalar alpha such that current + alpha * contrib approximates target
            # alpha = <residual, contrib> / <contrib, contrib>
            dot_rc = np.dot(residual, contrib)
            dot_cc = np.dot(contrib, contrib)
            if abs(dot_cc) < 1e-12:
                continue
            alpha = dot_rc / dot_cc
            
            new_res = residual - alpha * contrib
            new_err = np.sum(new_res**2)
            old_err = np.sum(residual**2)
            
            if new_err < old_err - 1e-12:
                improvement = old_err - new_err
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_u, best_v, best_w = u, v, w
                    best_res = new_res
                    best_alpha = alpha
        
        if best_u is not None:
            U[:, c] = best_alpha * best_u
            V[:, c] = best_v
            W[:, c] = best_w
            current += best_alpha * np.array([
                best_u[_idx_U(i, j)] * best_v[_idx_V(j, k)] * best_w[_idx_W(k, i)]
                for (i, j, k) in TRIPLES
            ])
        else:
            # No improvement, break
            break
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook (rank 75)
    U, V, W = _schoolbook()
    r = TENSOR_SIZE
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    # Verify schoolbook
    assert _verify(U, V, W), "Schoolbook should be valid"
    
    # Try to reduce rank via local search
    U, V, W, res = _local_search(U, V, W, r, deadline, n_iter=100)
    if _verify(U, V, W, tol=1e-7):
        actual_rank = U.shape[1]
        if actual_rank < best_rank:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = actual_rank
    
    # Try greedy construction with various ranks
    for target_rank in [60, 55, 50, 45, 40]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        U_g, V_g, W_g = _greedy_construct(target_rank, deadline)
        # Check if the greedy construction achieved a low residual
        res = _compute_residual(U_g, V_g, W_g, target_rank)
        max_res = float(np.max(np.abs(res)))
        if max_res < 1e-6:
            if target_rank < best_rank:
                best_U, best_V, best_W = U_g.copy(), V_g.copy(), W_g.copy()
                best_rank = target_rank
    
    # Return the best found
    # Convert to list format as expected
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
