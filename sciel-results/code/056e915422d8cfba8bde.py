"""Matrix multiplication 3x4x8 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T = sum_r U[:,r] * V[:,r] * W[:,r]^T (tensor outer product sum)
    # i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Strategy: Try to find a decomposition with rank < schoolbook_rank
    # using randomized search with local refinement
    
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try various ranks from low to high, stopping at first valid one
    # But first, let's try to find a good rank via optimization
    
    # Try Strassen-like or other known lower bounds
    # For 3x4x8, let's try ranks from a reasonable lower bound up
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Also, the tensor rank is at least the matrix rank of any unfolding
    
    # Let's try to find a decomposition using iterative methods
    
    # First, try a few candidate ranks
    candidate_ranks = []
    # Start from lower bound and go up
    lb = max(n*m, m*p, p*n)
    for r in range(lb, min(schoolbook_rank, lb + 40)):
        candidate_ranks.append(r)
    
    # Also include some specific values
    for r in [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88]:
        if r not in candidate_ranks and lb <= r < schoolbook_rank:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks))
    
    # For each candidate rank, try to find a decomposition
    # Use a combination of random initialization and gradient-free local search
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        success = _try_decomposition(T, n, m, p, rank, deadline)
        if success:
            best_rank = rank
            best_result = success
            # If we found one, try to go lower
            break
    
    # If no better than schoolbook was found, return schoolbook
    if best_rank >= schoolbook_rank:
        return best_result
    
    return best_result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_decomposition(T, n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition of tensor T."""
    nm, mp, pn = T.shape
    
    # Multiple random restarts
    max_restarts = 20
    max_iters = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Alternating least squares
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each column r, solve U[:,r] to minimize
            # ||T - sum_r U[:,r] outer V[:,r] outer W[:,r]||^2
            # Fix V and W, solve for U
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed r, U[:,r] should satisfy:
            # sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is a linear system if we treat it properly
            
            # Actually, let's use a simpler approach: 
            # Residual R = T - sum_r U[:,r] outer V[:,r] outer W[:,r]
            # Then update each factor using least squares
            
            # Compute residual
            R = T.copy()
            for r in range(rank):
                R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            # Update U[:, r] for each r
            for r in range(rank):
                # We want to find u such that sum_{a} u[a] * V[b,r]*W[c,r] approximates R[a,b,c]
                # For fixed r, define M_r[b,c] = V[b,r]*W[c,r]
                # Then we want u to minimize ||R[:, :, :] - u outer M_r||^2
                # This is: min_u ||R - u outer M_r||_F^2
                # Solution: u = (M_r^T M_r)^{-1} M_r^T R (reshaped)
                
                Mr = np.outer(V[:, r], W[:, r])  # shape (mp, pn)
                # R has shape (nm, mp, pn)
                # We want u of shape (nm,) such that u outer Mr approximates R
                # u[a] * Mr[b,c] approx R[a,b,c]
                # For each a: u[a] * Mr = R[a, :, :]
                # So u[a] = <R[a,:,:], Mr> / ||Mr||^2
                
                mr_norm_sq = np.sum(Mr * Mr)
                if mr_norm_sq < 1e-12:
                    continue
                for a in range(nm):
                    u_val = np.sum(R[a, :, :] * Mr) / mr_norm_sq
                    U[a, r] = u_val
            
            # Similarly update V and W
            R = T.copy()
            for r in range(rank):
                R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            for r in range(rank):
                Mr = np.outer(U[:, r], W[:, r])  # shape (nm, pn)
                mr_norm_sq = np.sum(Mr * Mr)
                if mr_norm_sq < 1e-12:
                    continue
                for b in range(mp):
                    v_val = np.sum(R[:, b, :] * Mr) / mr_norm_sq
                    V[b, r] = v_val
            
            R = T.copy()
            for r in range(rank):
                R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            for r in range(rank):
                Mr = np.outer(U[:, r], V[:, r])  # shape (nm, mp)
                mr_norm_sq = np.sum(Mr * Mr)
                if mr_norm_sq < 1e-12:
                    continue
                for c in range(pn):
                    w_val = np.sum(R[:, :, c] * Mr) / mr_norm_sq
                    W[c, r] = w_val
            
            # Check convergence
            R = T.copy()
            for r in range(rank):
                R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(R)
            if err < 1e-6:
                # Found a decomposition, now try to make entries rational
                return _rationalize(U, V, W, n, m, p)
        
        # After ALS, check if we got close
        R = T.copy()
        for r in range(rank):
            R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(R)
        if err < 1e-4:
            result = _rationalize(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None


def _rationalize(U, V, W, n, m, p):
    """Try to convert float decomposition to exact rational/half-integer entries."""
    # For each entry, try to find the nearest fraction with small denominator
    def to_rational(x, max_denom=100):
        # Try common fractions
        if abs(x) < 1e-10:
            return 0
        # Try integers and halves
        for d in range(1, max_denom + 1):
            num = round(x * d)
            if abs(x - num/d) < 1e-8:
                return num/d
        return x
    
    U_r = [[to_rational(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_r = [[to_rational(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_r = [[to_rational(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Verify the decomposition with rational entries
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank = U.shape[1]
    T_rec = np.zeros((nm, mp, pn))
    for r in range(rank):
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    T_rec[a, b, c] += U_r[a][r] * V_r[b][r] * W_r[c][r]
    
    if np.allclose(T_rec, T, atol=1e-8):
        return {"U": U_r, "V": V_r, "W": W_r}
    
    return None


# EVOLVE-BLOCK-END
