import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    
    # Compute tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Note: The task description says output factor follows W[k*n+i, r].
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually matrix multiplication C = A * B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # Rank 1 term: u_i v_j w_k.
    # U has shape (n*m, rank)? No, standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} is NOT correct for standard matmul tensor.
    # Standard matmul tensor M_{i,j,k} = 1 if indices match appropriately?
    # Let's look at the current program's verify:
    # T[i * m + j, j * p + k, k * n + i] 
    # This implies the tensors U, V, W are flattened vectors.
    # U is (n*m, rank). Index i*m+j corresponds to (i,j) of A.
    # V is (m*p, rank). Index j*p+k corresponds to (j,k) of B.
    # W is (p*n, rank). Index k*n+i corresponds to (k,i) of C?
    # Wait, C[i,k] = sum_j A[i,j] B[j,k].
    # So the "output" index is (i,k). The problem says W[k*n+i, r].
    # So W index is k*n+i. This matches (k,i) ordering for the third mode.
    # The tensor entry for a rank 1 term r is:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This product should sum to 1 for all valid (i,j,k) in the matmul definition?
    # No, the matmul tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # This is a linear map. The tensor T_{i,j,k} represents the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # For fixed i, k, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 if the indices are "compatible" in the summation?
    # Actually, the tensor of matrix multiplication is the set of triples (i,j,k) such that
    # the entry (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # Since j is the summed index, for ANY i, j, k, the term A[i,j]B[j,k] contributes to C[i,k].
    # So T_{i,j,k} = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # Let's check the current program's verify again.
    # It checks T[i * m + j, j * p + k, k * n + i] == 1.0.
    # And it checks that other entries are 0?
    # "T2 = T.copy(); ... T2[...] = 0.0; if np.max(np.abs(T2)) > 1e-9: return False"
    # This implies that for the flattened indices, only the specific triplets (i*m+j, j*p+k, k*n+i)
    # should be non-zero?
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then the flattened tensor would have many 1s.
    # But the indices are coupled.
    # Let's trace: 
    # U index: u = i*m + j
    # V index: v = j*p + k
    # W index: w = k*n + i
    # The condition for a non-zero entry in the tensor T[u, v, w] is that there exists i,j,k 
    # such that u = i*m+j, v = j*p+k, w = k*n+i.
    # This requires the j in U and V to match, the k in V and W to match, and the i in W and U to match.
    # So, yes, the tensor is sparse. It has 1s at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # And 0 elsewhere.
    # My optimization target must reconstruct this specific sparse tensor.
    
    # Flatten the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    # Compute current tensor
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # We can compute this efficiently.
    # T = np.einsum('ur,vr,wr->uvw', Ua, Va, Wa)
    
    # For speed in optimization, we might only check the non-zero entries and a sample of zero entries?
    # No, for exactness we need all. But for optimization, we can use a subset.
    # However, the final verification must be exact.
    
    # Let's compute the full error for verification.
    T = np.einsum('ur,vr,wr->uvw', Ua, Va, Wa)
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if abs(T[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                    return False
    
    # Check zero entries (sample or all? All is safer but slower. 
    # Given nm=16, mp=36, pn=36. Total 20736 entries. 
    # Non-zero: 4*4*9 = 144.
    # Zero: 20736 - 144 = 20592.
    # Checking all is fine for verification (once).
    
    # Create a mask of target indices
    # Actually, just comparing the whole tensor is easiest if memory allows.
    # 16 * 36 * 36 * 8 bytes = 165 KB. Very small.
    if not np.allclose(T, target, atol=1e-9):
        return False
    
    return True

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _optimize_rank(n, m, p, rank, deadline, seed=0):
    """
    Attempt to find a rank 'rank' decomposition using optimization.
    Returns (U, V, W) as lists of lists of floats if found, else None.
    """
    if time.time() > deadline:
        return None
        
    np.random.seed(seed)
    
    # Initialize factors
    # Using random normal initialization
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    # To speed up error calculation, we can precompute the indices of non-zeros
    # and use a random subset of zeros for the gradient/loss during optimization.
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i * m + j, j * p + k, k * n + i))
    nz_indices = np.array(nz_indices)
    
    # Total possible indices
    total_indices = n*m * m*p * p*n
    # We will sample 1000 zero indices for the loss function to keep it fast
    # but ensure we cover the space.
    
    def compute_loss(U, V, W):
        # T = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This is expensive if done fully.
        # Let's compute only on the sampled indices.
        
        # Non-zero part
        # T_nz = np.einsum('ur,vr,wr->n', U[nz_indices[:,0]], V[nz_indices[:,1]], W[nz_indices[:,2]])
        # Actually einsum with advanced indexing is tricky.
        # Let's do:
        U_nz = U[nz_indices[:, 0]] # (144, rank)
        V_nz = V[nz_indices[:, 1]]
        W_nz = W[nz_indices[:, 2]]
        T_nz = np.einsum('nr,nr,nr->n', U_nz, V_nz, W_nz)
        loss_nz = np.sum((T_nz - 1.0)**2)
        
        # Zero part (sampled)
        # Generate random indices not in nz_indices? 
        # Simpler: just generate random indices and ignore if they happen to be nz (prob 144/20736 ~ 0.7%)
        # Or just generate random indices from the full space.
        num_samples = 2000
        idx_u = np.random.randint(0, n*m, num_samples)
        idx_v = np.random.randint(0, m*p, num_samples)
        idx_w = np.random.randint(0, p*n, num_samples)
        
        U_s = U[idx_u]
        V_s = V[idx_v]
        W_s = W[idx_w]
        T_s = np.einsum('nr,nr,nr->n', U_s, V_s, W_s)
        loss_z = np.sum(T_s**2)
        
        return loss_nz + loss_z

    # Simulated Annealing / Random Descent
    current_loss = compute_loss(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = current_loss
    
    temp = 1.0
    cooling = 0.99
    
    steps = 0
    max_steps = 20000 # Adjust based on time
    
    while time.time() < deadline - 5.0:
        # Random perturbation
        # Pick a random factor and a random element to perturb
        factor_idx = np.random.randint(0, 3)
        if factor_idx == 0:
            row = np.random.randint(0, n*m)
            col = np.random.randint(0, rank)
            delta = np.random.randn() * temp
            U[row, col] += delta
            factors = (U, V, W)
        elif factor_idx == 1:
            row = np.random.randint(0, m*p)
            col = np.random.randint(0, rank)
            delta = np.random.randn() * temp
            V[row, col] += delta
            factors = (U, V, W)
        else:
            row = np.random.randint(0, p*n)
            col = np.random.randint(0, rank)
            delta = np.random.randn() * temp
            W[row, col] += delta
            factors = (U, V, W)
            
        new_loss = compute_loss(U, V, W)
        
        if new_loss < current_loss:
            current_loss = new_loss
            if new_loss < best_loss:
                best_loss = new_loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        else:
            # Accept with probability exp(-(new_loss - current_loss) / temp)
            if np.random.rand() < np.exp(-(new_loss - current_loss) / temp):
                current_loss = new_loss
            else:
                # Reject: revert change
                if factor_idx == 0:
                    U[row, col] -= delta
                elif factor_idx == 1:
                    V[row, col] -= delta
                else:
                    W[row, col] -= delta
        
        temp *= cooling
        steps += 1
        
        # Early termination if loss is very small
        if best_loss < 1e-10:
            break
            
    # Final check with full verification
    if best_loss < 1e-6:
        # Convert to lists of floats for the interface
        # The problem asks for integer/half-integer entries?
        # "integer/half-integer entries that exactly decompose"
        # If the optimization finds a solution with floats, we might need to round?
        # Rounding might break exactness.
        # However, if the solution is exact, the floats should be very close to rationals.
        # Let's try to return the floats first. If the evaluator is strict about types,
        # we might fail. But the current program returns ints.
        # The prompt says "integer/half-integer entries".
        # This suggests the solution space is discrete.
        # Optimization finds continuous solutions.
        # If it finds a solution, we can try to snap to nearest half-integers.
        
        U_snapped = np.round(best_U * 2) / 2
        V_snapped = np.round(best_V * 2) / 2
        W_snapped = np.round(best_W * 2) / 2
        
        # Verify snapped version
        if _verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
        # If snapping fails, try original floats? 
        # The interface says "integer/half-integer". 
        # If I return floats, it might be invalid.
        # I will return the snapped version if it's close, otherwise None.
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0 # 5s safety margin
    
    # Start with schoolbook as fallback
    res = _build_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # Known lower bound? 
    # Rank of 4x4x4 is 49 (Strassen).
    # Rank of 4x4x9?
    # We can try ranks lower than 114 (current best known from Strassen blocks).
    # Let's try a range of ranks.
    
    # Since optimization is expensive, we limit the number of ranks we try.
    # We know 114 works. Let's try to beat it.
    # Possible candidates: 100, 110, 114.
    
    # To save time, we will only try a few specific ranks.
    # If we find one, we return it.
    
    ranks_to_try = [100, 110, 114]
    
    # Randomize order? No, lower is better.
    for r in ranks_to_try:
        if time.time() > deadline - 30.0:
            break
            
        # Try optimization for this rank
        # Use multiple seeds/restarts
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
                
            result = _optimize_rank(n, m, p, r, deadline, seed=seed)
            if result is not None:
                U, V, W = result
                # Final strict verification
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
    # If no improvement found, return the best known (Schoolbook or Strassen if we had it)
    # The current program has a Strassen implementation that gives 114.
    # I will include a simplified Strassen construction to ensure we at least match 114.
    
    # Re-implementing the Strassen logic from the current program to ensure baseline 114
    # This is safer than just schoolbook (144).
    
    def _build_rank114(n, m, p):
        rank = 114
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen 2x2x2
        _STRASSEN_222 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
            for (u, v, w) in _STRASSEN_222:
                for t in range(4):
                    if u[t]:
                        U[a_base + t][col] += u[t]
                    if v[t]:
                        V[b_base + t][col] += v[t]
                    if w[t]:
                        W[c_base + t][col] += w[t]
                col += 1
            return col

        def _add_444(U, V, W, col, a_base, b_base, c_base):
            # Use 2x2x2 Strassen blocks to build 4x4x4?
            # The current program's _add_444 seems to use a specific set of sums.
            # Let's just copy the logic from the current program's _add_444.
            U_sums = [
                [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
                [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
            ]
            V_sums = [
                [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
                [0, -1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
            ]
            W_sums = [
                [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
            ]
            for t in range(7):
                u, v, w = U_sums[t], V_sums[t], W_sums[t]
                for s in range(4):
                    if u[s]:
                        U[a_base + s][col] += u[s]
                    if v[s]:
                        V[b_base + s][col] += v[s]
                    if w[s]:
                        W[c_base + s][col] += w[s]
                col += 1
            return col

        def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
            for i in range(ar):
                for j in range(ac):
                    U[a_base + i * ac + j][col] += 1
                    V[b_base + j * bc][col] += 1
                    W[c_base + i * bc][col] += 1
                    col += 1
            return col

        def _add_449(U, V, W, col, a_base, b_base, c_base):
            col = _add_444(U, V, W, col, a_base, b_base, c_base)
            col = _add_444(U, V, W, col, a_base, b_base + 4, c_base + 4)
            col = _add_matrix_vector_block(U, V, W, col, a_base, b_base + 8, c_base + 8, 4, 4, 1)
            return col

        col = _add_449(U, V, W, 0, 0, 0, 0)
        if col != rank:
            rank = col
            U = [row[:rank] for row in U]
            V = [row[:rank] for row in V]
            W = [row[:rank] for row in W]
        return {"U": U, "V": V, "W": W}

    try:
        res_strassen = _build_rank114(n, m, p)
        if _verify(res_strassen["U"], res_strassen["V"], res_strassen["W"], n, m, p):
            return res_strassen
    except Exception:
        pass
        
    return res

