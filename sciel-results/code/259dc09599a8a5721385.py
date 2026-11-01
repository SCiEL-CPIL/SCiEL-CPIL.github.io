"""Matrix multiplication 2x2 x 2x9: find rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from itertools import product as iproduct

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if the (i,k) output
    depends on A[i,j]*B[j,k]. Shape: (n, m, p)."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrix(T, n, m, p):
    """Reshape tensor T (n,m,p) into matrix of shape (n*m, p) for the
    W[k*n+i, r] convention. The tensor element T[i,j,k] maps to
    row = i*m + j (U index), col = k (V index part), and W index = k*n + i.
    
    Actually, let's think about this more carefully.
    
    The matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    In the tensor decomposition:
    C[i,k] = sum_r U[i*m+j... wait, let me re-read.
    
    U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    
    The decomposition says:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So for each (i,j,k), the sum over r of the product equals 1 (since T[i,j,k]=1).
    """
    # We need to verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # This is the standard matrix multiplication tensor.
    pass

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact: 
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """Try to remove a single rank-1 term (column) from the decomposition.
    Check if the remaining terms can be adjusted to still satisfy the equations.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Remove column term_idx
    mask = [r for r in range(rank) if r != term_idx]
    U_red = U[:, mask]
    V_red = V[:, mask]
    W_red = W[:, mask]
    new_rank = rank - 1
    
    # Now we need to check if there exist adjustments to U_red, V_red, W_red
    # such that the tensor equation still holds.
    # This is a nonlinear problem. Instead, let's check if the removed term
    # is linearly dependent in a specific way.
    
    # For a single term removal to work with the SAME other terms, we need:
    # The contribution of term r0 must be zero for all (i,j,k), which is impossible
    # unless U[i*m+j, r0]*V[j*p+k, r0]*W[k*n+i, r0] = 0 for all i,j,k,
    # which means one of the three vectors is identically zero.
    
    # A more general approach: allow the remaining terms to change.
    # This is a system of equations. Let's set up the linear system.
    # 
    # We have n*m*p = 36 equations (one per (i,j,k)) and 
    # 3 * (n*m + m*p + p*n) * (new_rank) variables if we allow all to change.
    # That's too many. 
    #
    # Alternative: fix two of the three matrices and solve for the third.
    # Fix U_red and V_red, solve for W_red.
    # For each (k,i) pair (W row), we have equations:
    # For each (j): sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is linear in W[k*n+i, :] for fixed (i,k).
    # 
    # Actually, for fixed (i,k), the equation is:
    # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's define for each (i,k): a vector w_ik of length new_rank
    # The equation becomes: sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * w_ik[r] = 1
    # So: A_ik @ w_ik = 1, where A_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
    
    # This gives n*p = 18 independent linear systems, each of size new_rank x new_rank
    # (actually just one equation per (i,k), so it's underdetermined if new_rank > 1).
    
    # Wait, I have one equation per (i,k) but new_rank unknowns per (i,k).
    # So it's underdetermined. We can find a solution if the system is consistent.
    
    # Let's just check: for each (i,k), does there exist w_ik such that A_ik @ w_ik = 1?
    # This is possible iff 1 is in the range of A_ik^T, i.e., iff A_ik is not the zero vector
    # (if A_ik is zero, we need 0=1 which fails; if A_ik is nonzero, we can always find w_ik).
    
    # Actually that's not quite right either. Let me just solve it.
    
    success = True
    W_new = np.zeros((p * n, new_rank))
    
    for i in range(n):
        for k in range(p):
            # Build the coefficient vector for this (i,k)
            a_vec = np.zeros(new_rank)
            for j in range(m):
                for r_idx, r in enumerate(mask):
                    a_vec[r_idx] += U[i*m+j, r] * V[j*p+k, r]
            
            # Solve a_vec @ w_ik = 1
            if np.allclose(a_vec, 0, atol=1e-10):
                success = False
                break
            else:
                # Find w_ik such that a_vec @ w_ik = 1
                # Minimum norm solution: w_ik = a_vec / (a_vec @ a_vec)
                denom = a_vec @ a_vec
                if abs(denom) < 1e-10:
                    success = False
                    break
                W_new[k*n+i, :] = a_vec / denom
        if not success:
            break
    
    if not success:
        return False, None, None, None
    
    # Verify the full decomposition
    ok, _ = _verify(U_red, V_red, W_new, n, m, p, tol=1e-6)
    if not ok:
        return False, None, None, None
    
    return True, U_red, V_red, W_new

def _try_remove_pair(U, V, W, n, m, p, term1, term2):
    """Try to remove two rank-1 terms simultaneously."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    mask = [r for r in range(rank) if r != term1 and r != term2]
    U_red = U[:, mask]
    V_red = V[:, mask]
    W_red = W[:, mask]
    new_rank = rank - 2
    
    # Fix U_red and V_red, solve for W_new
    success = True
    W_new = np.zeros((p * n, new_rank))
    
    for i in range(n):
        for k in range(p):
            a_vec = np.zeros(new_rank)
            for j in range(m):
                for r_idx, r in enumerate(mask):
                    a_vec[r_idx] += U[i*m+j, r] * V[j*p+k, r]
            
            if np.allclose(a_vec, 0, atol=1e-10):
                success = False
                break
            else:
                denom = a_vec @ a_vec
                if abs(denom) < 1e-10:
                    success = False
                    break
                W_new[k*n+i, :] = a_vec / denom
        if not success:
            break
    
    if not success:
        return False, None, None, None
    
    ok, _ = _verify(U_red, V_red, W_new, n, m, p, tol=1e-6)
    if not ok:
        return False, None, None, None
    
    return True, U_red, V_red, W_new

def _randomized_search(n, m, p, target_rank, max_iters=5000, timeout=None):
    """Randomized search for a decomposition of the given rank.
    
    Strategy: randomly initialize U, V, W with small entries, then use
    alternating least squares to converge.
    """
    start_time = time.time()
    
    for it in range(max_iters):
        if timeout and (time.time() - start_time) > timeout:
            break
        
        random_rank = target_rank
        U = np.random.randn(n * m, random_rank) * 0.5
        V = np.random.randn(m * p, random_rank) * 0.5
        W = np.random.randn(p * n, random_rank) * 0.5
        
        # Alternating least squares
        converged = False
        for inner in range(200):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Update W: for each (k,i), solve linear system
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            for i in range(n):
                for k in range(p):
                    a_vec = np.zeros(random_rank)
                    for j in range(m):
                        a_vec += U[i*m+j, :] * V[j*p+k, :]
                    if np.linalg.norm(a_vec) > 1e-8:
                        W[k*n+i, :] = a_vec / (a_vec @ a_vec)
            
            # Update V: for each (j,k), solve
            # sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: sum_r (sum_i U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] = 1
            for j in range(m):
                for k in range(p):
                    a_vec = np.zeros(random_rank)
                    for i in range(n):
                        a_vec += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(a_vec) > 1e-8:
                        V[j*p+k, :] = a_vec / (a_vec @ a_vec)
            
            # Update U: for each (i,j), solve
            # sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: sum_r (sum_k V[j*p+k, r] * W[k*n+i, r]) * U[i*m+j, r] = 1
            for i in range(n):
                for j in range(m):
                    a_vec = np.zeros(random_rank)
                    for k in range(p):
                        a_vec += V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.norm(a_vec) > 1e-8:
                        U[i*m+j, :] = a_vec / (a_vec @ a_vec)
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(random_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-8:
                converged = True
                break
        
        if converged:
            # Convert to lists
            U_list = [[float(U[i, r]) for r in range(random_rank)] for i in range(n*m)]
            V_list = [[float(V[i, r]) for r in range(random_rank)] for i in range(m*p)]
            W_list = [[float(W[i, r]) for r in range(random_rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _greedy_reduce(U, V, W, n, m, p, max_pairs_to_try=100):
    """Greedy reduction: try to remove terms one at a time, then in pairs."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    removed = []
    
    # Try removing single terms
    for attempt in range(rank):
        if not removed:
            current_rank = rank
        else:
            current_rank = rank - len(removed)
        
        # Try each remaining column
        found = False
        for term in range(current_rank):
            if term in removed:
                continue
            ok, U_new, V_new, W_new = _try_remove_term(
                U, V, W, n, m, p, term)
            if ok:
                U, V, W = U_new, V_new, W_new
                removed.append(term)
                found = True
                break
        
        if not found:
            break
    
    # Try removing pairs
    current_rank = rank - len(removed)
    for attempt in range(50):
        found = False
        for t1 in range(current_rank):
            if t1 in removed:
                continue
            for t2 in range(t1 + 1, current_rank):
                if t2 in removed:
                    continue
                ok, U_new, V_new, W_new = _try_remove_pair(
                    U, V, W, n, m, p, t1, t2)
                if ok:
                    U, V, W = U_new, V_new, W_new
                    removed.extend([t1, t2])
                    current_rank = rank - len(removed)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    # Build result
    mask = [r for r in range(rank) if r not in removed]
    U_final = U[:, mask]
    V_final = V[:, mask]
    W_final = W[:, mask]
    
    U_list = [[float(U_final[i, r]) for r in range(U_final.shape[1])] for i in range(U_final.shape[0])]
    V_list = [[float(V_final[i, r]) for r in range(V_final.shape[1])] for i in range(V_final.shape[0])]
    W_list = [[float(W_final[i, r]) for r in range(W_final.shape[1])] for i in range(W_final.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook (rank 36)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try greedy reduction on schoolbook
    if time.time() < deadline - 5:
        reduced = _greedy_reduce(
            np.array(best["U"], dtype=float),
            np.array(best["V"], dtype=float),
            np.array(best["W"], dtype=float),
            n, m, p)
        # Verify
        ok, _ = _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p)
        if ok:
            new_rank = len(reduced["U"][0])
            if new_rank < best_rank:
                best = reduced
                best_rank = new_rank
    
    # Try randomized search for lower ranks
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = _randomized_search(n, m, p, target_rank, max_iters=200,
                                     timeout=deadline - time.time() - 5)
        if result:
            ok, _ = _verify(result["U"], result["V"], result["W"], n, m, p)
            if ok:
                best = result
                best_rank = target_rank
                break
    
    # Final verification
    ok, _ = _verify(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best

# EVOLVE-BLOCK-END
