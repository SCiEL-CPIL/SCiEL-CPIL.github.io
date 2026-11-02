"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
import random
from collections import defaultdict

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n
SCHOOBOOK_RANK = NM * MP * PN  # 75

def _build_schoolbook():
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = NM * MP * PN
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] should satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,j,k pattern)
    
    Actually the decomposition means:
    (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j*p+k',c]*B[j,k'])
    
    The standard check: for each term c, the trilinear form
    f_c(A,B) = (U_c . A_vec) * (V_c . B_vec) * W_c
    should sum to give the correct matrix product.
    
    More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    and we need: sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * A[i,j] * sum_j' V[j'*p+k,c] * B[j',k]
    
    Wait, let me be more careful. The convention is:
    U has shape (n*m, rank): U[i*m + j, c] multiplies A[i,j]
    V has shape (m*p, rank): V[j*p + k, c] multiplies B[j,k]  
    W has shape (p*n, rank): W[k*n + i, c] is the coefficient for C[i,k]
    
    So: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    
    For this to equal sum_j A[i,j]*B[j,k] for all A,B:
    sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,k,j,j'
    
    This is a set of bilinear equations.
    """
    # Build the tensor: T[i,j,j',k] = sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c]
    # Should equal delta[j,j'] for all i,k
    rank = U.shape[1]
    # For each (i,k) pair, check that the matrix M_jj' = sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] is identity
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for c in range(rank):
                w = W[k * n + i, c]
                if abs(w) < 1e-15:
                    continue
                for j in range(m):
                    uj = U[i * m + j, c]
                    if abs(uj) < 1e-15:
                        continue
                    for jp in range(m):
                        vjp = V[jp * p + k, c]
                        if abs(vjp) < 1e-15:
                            continue
                        M[j, jp] += w * uj * vjp
            if not np.allclose(M, np.eye(m), atol=tol):
                return False
    return True

def _rank_from_decomposition(U, V, W):
    """Return the number of non-zero columns (terms)."""
    rank = U.shape[1]
    count = 0
    for c in range(rank):
        if (np.abs(U[:, c]) > 1e-12).any() or \
           (np.abs(V[:, c]) > 1e-12).any() or \
           (np.abs(W[:, c]) > 1e-12).any():
            count += 1
    return count

def _try_merge_terms(U, V, W, target_rank, deadline):
    """Try to find a decomposition with rank <= target_rank by merging/perturbing terms."""
    # Strategy: start from schoolbook, try to zero out some terms and adjust others
    # Use the fact that if we can express some basis vectors as combinations, we reduce rank.
    
    # Approach: Random linear combinations of terms
    # Pick a subset of terms to "absorb" and try to solve for the rest
    
    rank0 = U.shape[1]
    if target_rank >= rank0:
        return None
    
    # Try random perturbations: add small random vectors to some columns
    # and see if we can maintain the identity property with fewer effective terms
    
    # Better approach: use the bilinear structure
    # For fixed (i,k), we need sum_c W[ik,c] * u_c * v_c^T = I_m
    # where u_c = U[i*m: (i+1)*m, c] and v_c = V[k*p: k*p + ? , c]
    # Wait, V is indexed by j*p+k, so for fixed k, the relevant V entries are V[j*p+k, c] for j=0..m-1
    
    # Group terms by which (i,k) they contribute to
    # Actually every term contributes to potentially all (i,k) pairs
    
    # Let me try a different approach: random search in the space of decompositions
    # Start from schoolbook and apply random rank-reducing operations
    
    best = None
    best_rank = rank0
    
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        
        # Random perturbation: pick a few columns and add random combinations
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Pick 1-3 pairs of columns to merge
        n_merge = random.randint(1, min(3, rank0 - target_rank))
        for _ in range(n_merge):
            c1, c2 = random.sample(range(rank0), 2)
            # Try to absorb column c2 into column c1 with a random coefficient
            # This won't generally work, but let's try solving locally
            pass
        
        # Simpler: just check if any random subset of columns of size target_rank
        # can be adjusted to still satisfy the equations
        if random.random() < 0.3:
            # Try keeping only target_rank random columns and solving
            kept = sorted(random.sample(range(rank0), target_rank))
            U_k = U_try[:, kept]
            V_k = V_try[:, kept]
            W_k = W_try[:, kept]
            
            # Check if this subset alone can satisfy the equations
            # For each (i,k): sum_{c in kept} W_k[ik,c] * u_c * v_c^T = I
            # This is a linear system in W_k[ik,c] for each (i,k)
            
            valid = True
            W_new = np.zeros((PN, target_rank))
            for i in range(n):
                for k in range(p):
                    # Build matrix: for each c in kept, the outer product u_c * v_c^T
                    # We need sum_c W_new[ik,c] * (u_c * v_c^T) = I_m
                    # This is: W_new[ik,:] @ (stack of u_c * v_c^T) = vec(I)
                    # Stack: for each c, vec(u_c * v_c^T) is a vector of length m*m
                    # So we have a (m*m) x (target_rank) matrix A where A[:,c] = vec(u_c @ v_c.T)
                    # And we need A @ W_new[ik,:]^T = vec(I)
                    
                    A = np.zeros((m * m, target_rank))
                    for idx, c in enumerate(kept):
                        u_c = U_k[i * m: (i+1) * m, idx]
                        v_c = V_k[c * p: c * p + m, idx]  # WRONG: V is indexed j*p+k
                        # V[j*p+k, c] so for fixed k, entries are V[0*p+k, c], V[1*p+k, c], ..., V[(m-1)*p+k, c]
                        v_c = np.array([V_k[j * p + k, idx] for j in range(m)])
                        A[:, idx] = np.outer(u_c, v_c).flatten()
                    
                    b = np.eye(m).flatten()
                    # Solve A @ w = b (least squares or exact)
                    try:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        if rank_sol < target_rank and not np.allclose(A @ sol, b, atol=1e-9):
                            valid = False
                            break
                        W_new[k * n + i, :] = sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and _verify(U_k, V_k, W_new, tol=1e-8):
                r = _rank_from_decomposition(U_k, V_k, W_new)
                if r < best_rank:
                    best_rank = r
                    best = (U_k.copy(), V_k.copy(), W_new.copy())
        
        if time.time() >= deadline:
            break
    
    return best

def _structured_search(deadline):
    """Search for structured low-rank decompositions."""
    # Try decompositions with symmetry: e.g., terms that share the same (i,k) pattern
    # or where U, V have a Kronecker structure
    
    best = None
    best_rank = NM * MP * PN
    
    # Try rank values from a few below schoolbook down
    for target_rank in range(NM * MP * PN - 5, max(NM * MP * PN - 30, 50)):
        if time.time() > deadline:
            break
        if target_rank < best_rank:
            result = _try_merge_terms(*_build_schoolbook(), target_rank, min(deadline, time.time() + 5))
            if result is not None:
                U_r, V_r, W_r = result
                r = _rank_from_decomposition(U_r, V_r, W_r)
                if r < best_rank:
                    best_rank = r
                    best = result
                    print(f"Found rank {r} at target {target_rank}")
    
    return best

def _local_polish(U, V, W, deadline):
    """Try to reduce rank further from a given decomposition."""
    rank = U.shape[1]
    best_rank = _rank_from_decomposition(U, V, W)
    best = (U.copy(), V.copy(), W.copy())
    
    # Try removing each column and solving for it
    for c_remove in range(rank):
        if time.time() > deadline:
            break
        if best_rank <= 50:  # Don't bother if already low
            break
        kept = [c for c in range(rank) if c != c_remove]
        U_k = U[:, kept]
        V_k = V[:, kept]
        W_k = W[:, kept]
        
        new_rank = len(kept)
        W_new = np.zeros((PN, new_rank))
        valid = True
        for i in range(n):
            for k in range(p):
                A = np.zeros((m * m, new_rank))
                for idx, c in enumerate(kept):
                    u_c = U_k[i * m: (i+1) * m, idx]
                    v_c = np.array([V_k[j * p + k, idx] for j in range(m)])
                    A[:, idx] = np.outer(u_c, v_c).flatten()
                b = np.eye(m).flatten()
                try:
                    sol, res, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-8):
                        valid = False
                        break
                    W_new[k * n + i, :] = sol
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and _verify(U_k, V_k, W_new, tol=1e-8):
            r = _rank_from_decomposition(U_k, V_k, W_new)
            if r < best_rank:
                best_rank = r
                best = (U_k.copy(), V_k.copy(), W_new.copy())
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    best_rank = NM * MP * PN
    best_result = (U, V, W)
    
    # Try structured search
    if time.time() < deadline - 10:
        result = _structured_search(deadline - 5)
        if result is not None:
            U_r, V_r, W_r = result
            r = _rank_from_decomposition(U_r, V_r, W_r)
            if r < best_rank:
                best_rank = r
                best_result = (U_r, V_r, W_r)
    
    # Try local polish
    if time.time() < deadline - 5:
        polished = _local_polish(best_result[0], best_result[1], best_result[2], deadline)
        r = _rank_from_decomposition(polished[0], polished[1], polished[2])
        if r < best_rank:
            best_rank = r
            best_result = polished
    
    # Convert to list format as required
    U_final, V_final, W_final = best_result
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Ensure entries are clean (round small values)
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for i in range(len(row)):
                if abs(row[i]) < 1e-10:
                    row[i] = 0
                else:
                    # Try to find a simple fraction
                    from fractions import Fraction
                    f = Fraction(row[i]).limit_denominator(1000)
                    if abs(float(f) - row[i]) < 1e-9:
                        row[i] = float(f)
    
    return {"U": U_list, "V": V_list, "W": W_list}
