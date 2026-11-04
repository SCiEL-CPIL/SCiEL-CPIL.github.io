import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    nm, mp, np_ = n * m, m * p, n * p
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm, mp, np_ = n * m, m * p, n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = _build_target_tensor(n, m, p)
    
    # Reconstruct using einsum for speed
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.linalg.norm(T_recon - T_target) > tol:
        return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, max_iter=5000, time_limit=20.0):
    """Try to reduce rank using structured search and ALS with better initialization."""
    start = time.time()
    rank = U.shape[1]
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    T_target = _build_target_tensor(n, m, p)
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    nm, mp, np_ = n * m, m * p, n * p
    
    # Strategy 1: Try to find linear dependencies by checking if any rank-1 term
    # is in the span of others (after appropriate rescaling)
    # For each pair of terms, check if they can be merged
    
    # Strategy 2: Use structured initialization based on block decomposition
    # and run ALS to try to converge to lower rank
    
    for target_rank in range(max(15, rank - 10), best_rank):
        if time.time() - start > time_limit:
            break
        
        for attempt in range(8):
            if time.time() - start > time_limit:
                break
            
            # Initialize using a subset of the current decomposition
            # Take the first target_rank terms as initial guess
            if target_rank <= rank:
                U_rand = U[:, :target_rank].copy()
                V_rand = V[:, :target_rank].copy()
                W_rand = W[:, :target_rank].copy()
            else:
                U_rand = np.random.randn(nm, target_rank) * 0.1
                V_rand = np.random.randn(mp, target_rank) * 0.1
                W_rand = np.random.randn(np_, target_rank) * 0.1
            
            # ALS iterations with convergence check
            prev_err = np.inf
            for it in range(200):
                if time.time() - start > time_limit:
                    break
                
                # Update U
                for a in range(nm):
                    VW = np.einsum('br,cr->bc', V_rand, W_rand)
                    if np.linalg.matrix_rank(VW) >= target_rank:
                        U_rand[a, :] = np.linalg.lstsq(VW, T_target[a, :, :], rcond=None)[0]
                
                # Update V
                for b in range(mp):
                    UW = np.einsum('ar,cr->ac', U_rand, W_rand)
                    if np.linalg.matrix_rank(UW) >= target_rank:
                        V_rand[b, :] = np.linalg.lstsq(UW, T_target[:, b, :], rcond=None)[0]
                
                # Update W
                for c in range(np_):
                    UV = np.einsum('ar,br->ab', U_rand, V_rand)
                    if np.linalg.matrix_rank(UV) >= target_rank:
                        W_rand[c, :] = np.linalg.lstsq(UV, T_target[:, :, c], rcond=None)[0]
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand)
                err = np.linalg.norm(T_recon - T_target)
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            # Check if we found an exact decomposition
            if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None, None, None, rank

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, n * p
    
    # Build Strassen-based rank-36 decomposition
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for block_j in range(2):
        for block_k in range(3):
            if block_k < 2:
                Ub, Vb, Wb = _strassen_2x2x2()
                
                U_full = np.zeros((nm, 7))
                V_full = np.zeros((mp, 7))
                W_full = np.zeros((np_, 7))
                
                for i in range(2):
                    for j in range(2):
                        U_full[i*m + block_j*2 + j, :] = Ub[i*2 + j, :]
                
                for j in range(2):
                    for k in range(2):
                        V_full[(block_j*2 + j)*p + block_k*2 + k, :] = Vb[j*2 + k, :]
                
                for k in range(2):
                    for i in range(2):
                        W_full[(block_k*2 + k)*n + i, :] = Wb[k*2 + i, :]
                
                U_blocks.append(U_full)
                V_blocks.append(V_full)
                W_blocks.append(W_full)
            else:
                U_full = np.zeros((nm, 4))
                V_full = np.zeros((mp, 4))
                W_full = np.zeros((np_, 4))
                
                for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                    U_full[i*m + block_j*2 + j, idx] = 1
                    V_full[(block_j*2 + j)*p + block_k*2, idx] = 1
                    W_full[(block_k*2)*n + i, idx] = 1
                    
                U_blocks.append(U_full)
                V_blocks.append(V_full)
                W_blocks.append(W_full)
    
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    rank = U.shape[1]
    
    # Verify the Strassen-based decomposition
    if not _check_decomposition(U, V, W, n, m, p):
        # Something is wrong; fall back to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((np_, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Snap to half-integers (they're already integers, but be safe)
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    
    if not _check_decomposition(U_s, V_s, W_s, n, m, p):
        U_s, V_s, W_s = U, V, W
    
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank
    
    # Try to reduce rank if we have time
    if time.time() < deadline - 30:
        reduced = _try_rank_reduction(U_s, V_s, W_s, n, m, p,
                                       max_iter=5000,
                                       time_limit=min(30.0, deadline - time.time() - 5))
        if reduced[0] is not None and reduced[3] < best_rank:
            best_rank = reduced[3]
            best_result = {"U": reduced[0].tolist(), "V": reduced[1].tolist(), "W": reduced[2].tolist()}
    
    return best_result
