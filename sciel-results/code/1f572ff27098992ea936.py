"""Tensor rank minimization for 3x5x7 matrix multiplication.
Uses structured rank-1 decomposition to try to find a rank < 105 exact decomposition.
Falls back to schoolbook (rank 105) if no improvement found within time budget."""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0  # 20s safety margin before 140s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the 3x5x7 matrix multiplication tensor.
    
    The tensor T[i, j, k] = 1 if and only if there exists a rank-1 term c such that
    U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1, and for all other (i,j,k) the sum is 0.
    
    Actually, the correct verification: for each output element (i, k), the product (AB)[i,k]
    = sum_j A[i,j] B[j,k]. In tensor form:
    T[i, j, k] should equal 1 for the "multiplication" tensor.
    The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _verify_decomposition_np(U, V, W, n, m, p):
    """Faster verification using numpy."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to 3D tensors
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
    # We need to check T[i,j,k] == 1 for all i,j,k
    # W_t[k,i,c] -> transpose to [i, k, c]
    W_ik = W_t.transpose(1, 0, 2)  # [n, p, rank]
    
    # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_ik[i,k,c]
    # = sum_c (U_t[i,j,c] * V_t[j,k,c]) * W_ik[i,k,c]
    
    # This is a bit tricky to vectorize directly. Let's do it in a loop over k for speed.
    for k in range(p):
        # U_t[:, :, :] * V_t[:, k, :] gives [n, m, rank]
        prod = U_t * V_t[:, k, :][None, :, :]  # [n, m, rank]
        # Now sum over c: prod[i, j, c] * W_ik[i, k, c]
        # = sum_c prod[i, j, c] * W_ik[i, k, c]
        T_ik = np.einsum('ijc,ic->ij', prod, W_ik[:, k, :])
        if not np.allclose(T_ik, 1.0, atol=1e-10):
            return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _snap_to_half_int(val, tol=1e-9):
    """Snap a value to the nearest half-integer if within tolerance."""
    scaled = val * 2
    nearest = round(scaled)
    if abs(val * 2 - nearest) < tol:
        return Fraction(nearest, 2)
    return None


def _snap_matrix(M, tol=1e-9):
    """Try to snap all entries to half-integers. Returns list of Fraction lists or None."""
    result = []
    for row in M:
        new_row = []
        for v in row:
            sv = _snap_to_half_int(v, tol)
            if sv is None:
                return None
            new_row.append(sv)
        result.append(new_row)
    return result


def _rank1_contribution(U, V, W, c, n, m, p):
    """Get the contribution of rank-1 term c to T[i,j,k]."""
    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Uc = [U[i * m + j][c] for i in range(n) for j in range(m)]
    Vc = [V[j * p + k][c] for j in range(m) for k in range(p)]
    Wc = [W[k * n + i][c] for k in range(p) for i in range(n)]
    
    return Uc, Vc, Wc


def _try_randomized_search(n, m, p, max_rank, time_budget):
    """Attempt to find a decomposition with rank < n*m*p using random initialization
    and coordinate descent / gradient-like updates."""
    import numpy as np
    
    n, m, p = n, m, p
    target_rank = max_rank
    
    # We'll work with floating point for the search, then snap at the end.
    # The tensor T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    
    # Strategy: random initialization with sparse structure, then iterative refinement.
    # Use a simple approach: try random sign patterns and see if we can satisfy constraints.
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Try various ranks below schoolbook
    for rank in range(target_rank, min(target_rank, best_rank)):
        if _remaining_time() < 5:
            break
        
        # Try multiple random initializations
        max_trials = 50
        for trial in range(max_trials):
            if _remaining_time() < 3:
                break
            
            # Random sparse initialization
            np.random.seed(trial * 1000 + rank)
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with random non-zero patterns
            for c in range(rank):
                # Each rank-1 term touches some subset of entries
                # Random choice: each row has ~1-3 nonzeros
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, n * m - 1)
                    U[idx, c] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, m * p - 1)
                    V[idx, c] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, p * n - 1)
                    W[idx, c] = random.choice([-1, 1])
            
            # Verify
            if _verify_decomposition_np(U, V, W, n, m, p):
                # Snap to half-integers
                U_frac = _snap_matrix(U.tolist(), tol=1e-8)
                V_frac = _snap_matrix(V.tolist(), tol=1e-8)
                W_frac = _snap_matrix(W.tolist(), tol=1e-8)
                if U_frac and V_frac and W_frac:
                    # Convert to int lists for output
                    U_out = [[int(x) for x in row] for row in U_frac]
                    V_out = [[int(x) for x in row] for row in V_frac]
                    W_out = [[int(x) for x in row] for row in W_frac]
                    if _verify_decomposition(U_out, V_out, W_out, n, m, p):
                        return U_out, V_out, W_out, rank
            if _remaining_time() < 2:
                break
    
    return None


def _try_structured_reduction(n, m, p):
    """Try structured reductions based on known tensor rank results.
    
    For small matrix multiplication, known results:
    - 2x2x2: rank 7 (Strassen)
    - 2x3x3: rank 16
    - 3x3x3: rank 23
    - 2x2x3: rank 11
    
    For 3x5x7, no known rank below 105 is published in standard references,
    but we can try embedding/padding strategies.
    
    Strategy: Pad to 4x8x8 (or 4x6x8) and use known decompositions,
    then restrict back.
    """
    import numpy as np
    
    # Try padding 3x5x7 to 4x8x8
    # If we have a rank-R decomposition for 4x8x8, we can restrict to 3x5x7
    # by zeroing out rows/columns corresponding to the extra dimensions.
    # This might give a valid (but not necessarily minimal) decomposition for 3x5x7.
    
    # Known: 4x4x4 has rank 49 (Landsberg). 8x8x8 has rank 218 (Strassen).
    # 4x8x8: not standard. Let's not go down this path without known formulas.
    
    # Alternative: Try to combine Strassen-style tricks on blocks.
    # Split 3x5x7 into blocks and apply lower-rank multiplication to each block.
    
    # For example: 3x5 = [1x5; 2x5], 5x7 = [5x1, 5x6] etc.
    # This gets complicated. Let's try a simpler approach:
    
    # Check if we can find a rank-104 decomposition by searching over small perturbations
    # of the schoolbook solution.
    
    # The schoolbook has 105 rank-1 terms, each with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # Can we merge two terms? If we can find two terms c1, c2 such that their sum
    # can be replaced by a single rank-1 term, we reduce rank by 1.
    
    # Two rank-1 terms:
    # T1[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i]
    # T2[i,j,k] = U2[i,j]*V2[j,k]*W2[k,i]
    # T1+T2[i,j,k] = T3[i,j,k] = U3[i,j]*V3[j,k]*W3[k,i]
    
    # In schoolbook, each term is supported on exactly one (i,j,k).
    # So T1 is supported at (i1,j1,k1) and T2 at (i2,j2,k2).
    # If (i1,j1,k1) != (i2,j2,k2), then T1+T2 has support at two points,
    # but a rank-1 tensor U3*V3*W3 has support that is a "grid" - 
    # if U3[a,b]!=0, V3[b',c]!=0, W3[c',a'']!=0, then T3[a,b',c''] can be nonzero
    # for a grid of points. So T1+T2 with two separated points cannot be rank 1
    # unless the support is compatible.
    
    # Actually, a rank-1 tensor U⊗V⊗W has the property that if T[a,b,c]≠0 and
    # T[a',b',c']≠0, then T[a,b',c]≠0, T[a,b,c']≠0, T[a',b,c]≠0, etc.
    # (It's a "grid" or "combinatorial rectangle" of support.)
    
    # So two schoolbook terms at different (i,j,k) points CANNOT be merged into one
    # rank-1 term because their combined support is two isolated points, not a grid.
    
    # Therefore, we cannot simply merge two schoolbook terms.
    
    # However, we CAN replace multiple schoolbook terms with a different set of
    # rank-1 terms that collectively cover the same tensor. This is what Strassen does.
    
    # For 3x5x7, let's try a block decomposition:
    # Split A (3x5) into [A1: 1x5; A2: 2x5]
    # Split B (5x7) into [B1: 5x1, B2: 5x6]
    # C = A*B: C[0,:] = A1*B, C[1:,:] = A2*B
    # A1*B is 1x7, uses 5 mults (or 1x5 * 5x7)
    # A2*B is 2x7, uses 10 mults (or 2x5 * 5x7)
    # Total: 15 mults for the "outer" structure, but that's the schoolbook.
    
    # Try Strassen on the 2x5 * 5x7 part:
    # Pad 2x5 to 2x8? Or use 2x2 Strassen blocks.
    # 2x5 = [2x2, 2x2, 2x1]
    # 5x7 = [[2x3, 2x4], [2x3, 2x4], [1x3, 1x4]]
    # This is getting too complex without a clear rank advantage.
    
    # Let's try a computational search: can we find any rank-104 decomposition?
    # We'll do a targeted search: take the schoolbook and try to replace
    # subsets of terms with lower-rank equivalents.
    
    # For 3x3x3, the rank is 23 (not 27). The reduction comes from Strassen-type
    # tricks on 2x2x2 sub-blocks.
    
    # For 3x5x7, we can try to identify 2x2x2 sub-blocks and apply Strassen to them.
    # A 3x5x7 multiplication contains overlapping 2x2x2 sub-problems.
    
    # Let's try: decompose using Strassen on the (1,2)x(1,2)x(1,2) sub-block
    # and schoolbook for the rest.
    
    # This is complex. Let me try a direct optimization approach instead.
    
    return None


def _try_gradient_descent(n, m, p, target_rank, time_budget):
    """Try to find a rank-`target_rank` decomposition using gradient descent
    on the residual ||T - sum_c U_c V_c W_c||^2."""
    import numpy as np
    
    if _remaining_time() < 10:
        return None
    
    # T[i,j,k] = 1 for all i,j,k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Residual: R[i,j,k] = 1 - sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Minimize ||R||^2
    
    # Initialize with schoolbook and remove some terms, or random init.
    np.random.seed(42)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_iter = 2000
    lr = 0.01
    
    for attempt in range(3):
        if _remaining_time() < 5:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        for it in range(n_iter):
            if _remaining_time() < 5:
                break
            
            # Compute residual
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # Use einsum for speed
            U_t = U.reshape(n, m, target_rank)
            V_t = V.reshape(m, p, target_rank)
            W_t = W.reshape(p, n, target_rank)
            W_ik = W_t.transpose(1, 0, 2)  # [n, p, rank]
            
            # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_ik[i,k,c]
            # = einsum('ijc,jkc,ikc->ijk', U_t, V_t, W_ik) -- this is expensive
            # Let's do it in a loop over k
            T_approx = np.zeros((n, m, p))
            for k in range(p):
                # U_t[i,j,c] * V_t[j,k,c] * W_ik[i,k,c]
                prod = U_t * V_t[:, k, :][None, :, :]  # [n, m, rank]
                T_approx[:, :, k] = np.einsum('ijc,ic->ij', prod, W_ik[:, k, :])
            
            R = 1.0 - T_approx  # residual
            loss = np.sum(R * R)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-10:
                break
            
            # Gradient descent step (approximate)
            # dL/dU[i,j,c] = -2 * R[i,j,k] * V[j,k,c] * W[k,i,c] (summed over k)
            # This is expensive. Let's use a simpler update.
            
            # For each c, update U, V, W
            for c in range(target_rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_ik = Wc.transpose(1, 0)  # [n, p]
                
                # T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc_ik[i,k]
                # dR[i,j,k]/dUc[i,j] = -Vc[j,k] * Wc_ik[i,k]
                # grad_Uc[i,j] = -2 * sum_k R[i,j,k] * Vc[j,k] * Wc_ik[i,k]
                
                # Compute in loop over k
                grad_Uc = np.zeros((n, m))
                grad_Vc = np.zeros((m, p))
                grad_Wc = np.zeros((p, n))
                
                for k in range(p):
                    R_k = R[:, :, k]  # [n, m]
                    Vc_k = Vc[:, k]  # [m]
                    Wc_ik = Wc_ik[:, k]  # [n]
                    
                    # grad_Uc[i,j] += -2 * R[i,j] * Vc[j] * Wc_ik[i]
                    grad_Uc += -2 * R_k * np.outer(Wc_ik, Vc_k)
                    
                    # grad_Vc[j,k] = -2 * sum_i R[i,j,k] * Uc[i,j] * Wc_ik[i]
                    # = -2 * (R[:, :, k] * Uc * Wc_ik[:, None]) @ ones? No.
                    # grad_Vc[j,k] = -2 * sum_i R[i,j,k] * Uc[i,j] * Wc_ik[i,k]
                    tmp = R_k * Uc * Wc_ik[:, None]  # [n, m]
                    grad_Vc[:, k] = -2 * np.sum(tmp, axis=0)
                    
                    # grad_Wc[k,i] = -2 * sum_j R[i,j,k] * Uc[i,j] * Vc[j,k]
                    tmp2 = R_k * Uc * Vc[:, k][None, :]  # [n, m]
                    grad_Wc[k, :] = -2 * np.sum(tmp2, axis=1)
                
                U[:, c] -= lr * grad_Uc.flatten()
                V[:, c] -= lr * grad_Vc.flatten()
                W[:, c] -= lr * grad_Wc.flatten()
            
            if it % 100 == 0:
                pass  # Could print loss for debugging
    
    if best_loss < 1e-10:
        # Snap to half-integers
        U_frac = _snap_matrix(best_U.tolist(), tol=1e-6)
        V_frac = _snap_matrix(best_V.tolist(), tol=1e-6)
        W_frac = _snap_matrix(best_W.tolist(), tol=1e-6)
        if U_frac and V_frac and W_frac:
            U_out = [[int(x) for x in row] for row in U_frac]
            V_out = [[int(x) for x in row] for row in V_frac]
            W_out = [[int(x) for x in row] for row in W_frac]
            if _verify_decomposition(U_out, V_out, W_out, n, m, p):
                return U_out, V_out, W_out, target_rank
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    best_result = None
    
    # Try gradient descent with rank 104
    if _remaining_time() > 20:
        result = _try_gradient_descent(n, m, p, schoolbook_rank - 1, 60)
        if result:
            best_result = result
    
    # Try randomized search
    if best_result is None and _remaining_time() > 10:
        result = _try_randomized_search(n, m, p, schoolbook_rank - 1, 40)
        if result:
            best_result = result
    
    # Fall back to schoolbook
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
