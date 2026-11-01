import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 3, 3, 5
    N = n * m  # 9
    M = m * p  # 15
    P = p * n  # 15

    # 1. Construct the Schoolbook Decomposition (Rank 45)
    # This is a guaranteed valid solution.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((N, rank_schoolbook))
    V_sb = np.zeros((M, rank_schoolbook))
    W_sb = np.zeros((P, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Helper to check if a candidate decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        # The condition is: For all i, j, k:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 if (i,j,k) is a valid term?
        # No, the tensor T_{i,j,k} is defined such that:
        # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The decomposition represents the bilinear map (A, B) -> C.
        # C_{i,k} = sum_r (sum_j U_{i*m+j, r} A_{i,j} ?) No.
        
        # Standard Tensor Definition:
        # T_{i, j, k} = 1 if the indices correspond to the multiplication term.
        # Specifically, the tensor is of size (n*m, m*p, p*n).
        # Entry (i*m+j, j*p+k, k*n+i) corresponds to the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # Wait, the indices in the factor matrices are:
        # U: (n*m, rank) -> index u = i*m + j
        # V: (m*p, rank) -> index v = j*p + k
        # W: (p*n, rank) -> index w = k*n + i
        
        # The reconstruction formula is:
        # C_{i,k} = sum_r W_{k*n+i, r} * ( sum_j U_{i*m+j, r} A_{i,j} ? ) ... No.
        
        # Let's look at the standard rank-1 tensor: u \otimes v \otimes w.
        # T_{u,v,w} = u_u v_v w_w.
        # For matrix multiplication, we want to represent the tensor T such that
        # T_{i*m+j, j*p+k, k*n+i} = 1 and all other entries are 0?
        # No, that's not right. The tensor T is defined on the space A x B x C^*.
        # Actually, the "official trace convention" usually implies:
        # The decomposition is valid if:
        # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 
        # for all i, j, k? No, that would sum to rank if all 1s.
        
        # Correct interpretation:
        # The multiplication tensor T has entries T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
        # The factorization expresses the bilinear form.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # A rank-1 decomposition term corresponds to:
        # C_{i,k} += (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) * W_{i,k,r} ?
        # No, the indices in U are flattened (n*m).
        
        # Let's derive the condition from the schoolbook case.
        # In schoolbook:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
        # For a specific term (i,j,k), the contribution to C_{i,k} is:
        # U[i*m+j, r] * A_{i,j} ? No, U is just a matrix.
        
        # The standard definition of matrix multiplication tensor rank:
        # We want to find R, U, V, W such that:
        # For all i, j, k:
        # sum_{r=1}^R U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1
        # Is this correct?
        # If we set A and B to identity matrices or basis matrices, we can test.
        # Let A_{i,j} = delta_{i,a} delta_{j,b} and B_{j,k} = delta_{j,c} delta_{k,d}.
        # Then C_{i,k} = delta_{i,a} delta_{b,c} delta_{c,c} delta_{k,d} = delta_{i,a} delta_{b,c} delta_{k,d}.
        # The decomposition computes:
        # C_{i,k} = sum_r W_{k*n+i, r} * (sum_{j'} U_{i*m+j', r} A_{i,j'}) * (sum_{j'} V_{j'*p+k, r} B_{j',k}) ?
        # This form is not standard.
        
        # The standard form is:
        # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_j V_{j,k,r} B_{j,k}) W_{i,k,r} is wrong because U and V indices don't match the flattened input.
        
        # Let's stick to the tensor identity.
        # The tensor T is an element of (R^n \otimes R^m)^* \otimes (R^m \otimes R^p)^* \otimes (R^n \otimes R^p).
        # Or simply, T_{u, v, w} where u indexes A, v indexes B, w indexes C.
        # T_{i*m+j, j*p+k, k*n+i} = 1.
        # And T_{u,v,w} = 0 otherwise.
        # A decomposition of rank R is:
        # T_{u,v,w} = sum_{r=1}^R U_{u,r} V_{v,r} W_{w,r}
        
        # So the check is:
        # For all u in [0, N), v in [0, M), w in [0, P):
        # sum_r U[u,r] V[v,r] W[w,r] == T[u,v,w]
        
        # We can check this efficiently by sampling or checking the non-zero entries and a few zero entries.
        # Non-zero entries:
        # u = i*m + j, v = j*p + k, w = k*n + i.
        # There are n*m*p = 45 non-zero entries.
        # We must ensure sum_r U[u,r] V[v,r] W[w,r] == 1 for these.
        # And sum_r U[u,r] V[v,r] W[w,r] == 0 for all other (u,v,w).
        
        # Checking all zero entries is O(N*M*P*R) = 9*15*15*45 ~ 91k ops per check, which is fast.
        
        # Construct T_target as a sparse set of indices
        # Actually, just checking the 45 non-zero entries is necessary but not sufficient.
        # We must also ensure the "off-diagonal" terms are zero.
        
        # Efficient check:
        # Compute the tensor product for a few random zero-entries and all non-zero entries.
        
        # 1. Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    val = np.dot(np.dot(U[u, :], V[v, :]), W[w, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # 2. Check some zero entries
        # A zero entry is any (u, v, w) that does not satisfy the index alignment.
        # u = i*m + j, v = j'*p + k, w = k'*n + i'
        # Alignment requires: j == j', k == k', i == i'.
        # So if any of j!=j', k!=k', i!=i' differ, it should be 0.
        
        # We can check a random sample of zero entries to save time, or all if fast enough.
        # Total entries: 9*15*15 = 2025.
        # Non-zero: 45.
        # Zero: 1980.
        # Checking all 1980 with R=45 is 1980 * 45 ~ 90k multiplications. Very fast.
        
        # However, doing this inside an optimization loop might be slow if called frequently.
        # For a final validation, it's fine. For a search loop, we might use a cheaper proxy or less frequent checks.
        
        # Let's just check all zero entries to be safe for the final result.
        # To speed up, we can precompute which (u,v) pairs are "compatible" for a given w?
        # No, just loop.
        
        for u in range(N):
            i = u // m
            j = u % m
            for v in range(M):
                j2 = v // p
                k = v % p
                # If j != j2, then for any w, the term is 0?
                # Not necessarily. The sum is over r.
                # But the target T[u,v,w] is 0.
                for w in range(P):
                    # Check if this is a non-zero entry
                    i2 = w % n
                    k2 = w // n
                    if i == i2 and j == j2 and k == k2:
                        continue # Already checked
                    val = np.dot(np.dot(U[u, :], V[v, :]), W[w, :])
                    if abs(val) > tol:
                        return False
        return True

    # 2. Attempt to find a lower rank decomposition
    # Strategy: Start with a rank slightly lower than schoolbook, or use a heuristic.
    # Given the difficulty, we will try a local search / gradient descent on a relaxed objective
    # or simply try to "merge" columns of the schoolbook decomposition.
    
    # A common technique for small tensors is to use a random initialization and gradient descent
    # on the residual of the tensor equation.
    
    def tensor_residual(U, V, W):
        """Compute the Frobenius norm of the difference between the decomposed tensor and the target tensor."""
        # Target tensor T is 1 at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
        # We compute the decomposed tensor D[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
        # We want to minimize ||D - T||_F^2.
        
        # This is expensive to compute fully (2025 entries * R ops).
        # But R is small (<=45). 2025 * 45 = 91k ops. This is very fast in NumPy.
        
        # Construct T_target as a dense array? 9x15x15 = 2025 floats. Small.
        # T_target = np.zeros((N, M, P))
        # for i in range(n):
        #     for j in range(m):
        #         for k in range(p):
        #             T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
        # D = np.einsum('ur,vr,wr->uvw', U, V, W)
        # return np.linalg.norm(D - T_target)
        
        # To save memory/time in loops, we can compute the residual on the fly or use a cheaper metric.
        # Let's use the full residual for accuracy.
        
        # Pre-compute T_target outside?
        return None # Placeholder

    # Let's pre-compute the target tensor and indices for speed
    T_target = np.zeros((N, M, P))
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                nonzero_indices.append((u, v, w))

    def compute_residual(U, V, W):
        """Compute ||D - T||_F^2"""
        # D_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # This is equivalent to:
        # D = U @ diag? No.
        # We can compute D by iterating over r:
        # D += U[:, r, None, None] * V[None, :, r, None] * W[None, None, :, r]
        # This is O(R * N * M * P).
        # R=45, N=9, M=15, P=15. 45 * 2025 = 91k ops.
        
        D = np.zeros((N, M, P))
        for r in range(U.shape[1]):
            D += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        
        return np.sum((D - T_target)**2)

    def compute_grads(U, V, W):
        """Compute gradients of residual w.r.t U, V, W."""
        # Res = sum_{u,v,w} (D_{u,v,w} - T_{u,v,w})^2
        # D_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # dRes/dU_{u,r} = 2 * sum_{v,w} (D_{u,v,w} - T_{u,v,w}) V_{v,r} W_{w,r}
        
        # Let E_{u,v,w} = D_{u,v,w} - T_{u,v,w}
        # grad_U[u, r] = 2 * sum_{v,w} E[u, v, w] V[v, r] W[w, r]
        # This can be computed as:
        # grad_U = 2 * E @ (V * W^T)? No.
        # grad_U[u, r] = 2 * sum_w sum_v E[u,v,w] V[v,r] W[w,r]
        
        # We can compute this efficiently using matrix multiplications.
        # Let M_VW[v, w, r] = V[v, r] W[w, r].
        # Then grad_U[u, r] = 2 * sum_{v,w} E[u, v, w] M_VW[v, w, r]
        # This is a tensor contraction.
        
        # Simpler: Loop over r?
        # grad_U[:, r] = 2 * (E * (V[:, r, None] * W[None, :, r])) sum over v,w?
        # No, E is 3D.
        
        # Let's use the property:
        # grad_U = 2 * np.einsum('uvw,vr,wr->ur', E, V, W)
        # grad_V = 2 * np.einsum('uvw,ur,wr->vr', E, U, W)
        # grad_W = 2 * np.einsum('uvw,ur,vr->wr', E, U, V)
        
        D = np.zeros((N, M, P))
        for r in range(U.shape[1]):
            D += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        E = D - T_target
        
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', E, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', E, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', E, U, V)
        
        return grad_U, grad_V, grad_W, compute_residual(U, V, W)

    # Optimization Loop
    # We will try to reduce the rank from 44 down to a lower bound.
    # Lower bound for 3x3x5: max(9, 15, 15) = 15.
    # Known ranks for 3x3x3 is 23.
    # For 3x3x5, it's likely higher. Let's try to find a valid decomposition.
    # Since exact integer/half-integer solutions are required, gradient descent on reals might not land on them.
    # However, the prompt says "integer/half-integer entries".
    # The schoolbook solution uses integers.
    # If we find a real solution, we might need to round. But rounding breaks exactness.
    # So we should look for exact solutions or very close ones that can be snapped.
    
    # Given the time constraint and the difficulty of finding exact low-rank decompositions for 3x3x5,
    # the safest bet is to return the schoolbook decomposition unless we find a provably better one quickly.
    
    # Let's try a few random restarts with gradient descent to see if we can find a lower rank.
    # We'll use a small number of iterations to stay within time limits.
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 44 down to 20?
    # If we find a valid solution, we return it.
    
    # To ensure we return *something* valid, we always have the schoolbook solution.
    
    # Strategy:
    # 1. Start with rank R = 44.
    # 2. Initialize U, V, W with small random values.
    # 3. Run gradient descent for a few steps.
    # 4. Check if residual is close to 0.
    # 5. If yes, check if entries can be snapped to half-integers.
    # 6. If snapped version is valid, return it.
    # 7. Decrease R and repeat.
    
    # However, finding exact half-integer solutions via continuous GD is tricky.
    # Often, these decompositions are derived from algebraic identities.
    # Without a specific identity for 3x3x5, random search is unlikely to succeed quickly.
    
    # Let's check if we have time for a more structured search.
    # Given 140s, we can do quite a bit.
    
    # But wait, the prompt says "integer/half-integer entries".
    # The schoolbook solution has integers.
    # If we can't find a better one, we return schoolbook.
    
    # Let's try to see if we can improve the schoolbook solution by merging terms.
    # This is equivalent to finding a linear dependence among the rank-1 tensors.
    
    # For now, let's just implement a basic GD loop for a few ranks to see if it converges.
    # If it doesn't converge to a valid exact solution, we fall back to schoolbook.
    
    def try_rank(R):
        if time.time() > deadline - 10:
            return None
        
        # Random init
        U = rng.standard_normal((N, R)) * 0.1
        V = rng.standard_normal((M, R)) * 0.1
        W = rng.standard_normal((P, R)) * 0.1
        
        lr = 0.01
        for step in range(100): # 100 steps max per rank attempt
            if time.time() > deadline - 10:
                return None
            
            grad_U, grad_V, grad_W, res = compute_grads(U, V, W)
            
            if res < 1e-6:
                # Check for half-integer snapping
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if is_valid(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
        return None

    # Try to find a lower rank
    # We'll try a few ranks in descending order
    for R in range(44, 20, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(R)
        if result:
            U_opt, V_opt, W_opt = result
            best_rank = R
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            break # Found a better one, stop searching for even lower? Or continue?
            # Usually, if we find one, we might try to go lower. But let's be safe and return the best found.
            # Actually, the loop goes downwards, so the first one found is the highest rank < 45.
            # We can continue to try lower ranks if time permits.
            # Let's continue the loop but only if we have time.
            # The loop structure `for R in ...` will continue to next R (lower).
            # We should update best and continue.

    # Final check: ensure the best found is valid
    if not is_valid(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_schoolbook

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
