"""Stronger program: SAT-style search over finite field GF(3) for rank-23 decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
MOD = 3  # work over GF(3)

def _schoolbook():
    n, m, p = 3, 3, 3
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

def _verify_decomposition(U, V, W, n=3, m=3, p=3, mod=MOD):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 (mod mod) for all i,j,k."""
    rank = U[0].__len__() if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    if isinstance(U, list):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    else:
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s % mod != 1:
                    return False
    return True

def _verify_real(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify over reals: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(S, np.ones((n, m, p)), atol=tol)

def _try_sat_search(rank, n=3, m=3, p=3, mod=MOD, time_budget=8.0):
    """
    Search for a rank-`rank` decomposition over GF(mod).
    We fix W to be the identity-like structure and search for U, V.
    
    Strategy: We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 (mod mod) for all i,j,k.
    
    A common approach: set W[k*n+i, r] = delta_{k,i} * w_{k,r} for some pattern,
    or use a structured W.
    
    Let's try: W[k*n+i, r] = 1 if k == i, else 0. Then:
    sum_r U[i,j,r] * V[j,i,r] * 1 = 1 (mod mod) for all i,j (where k=i).
    And for k != i: sum_r U[i,j,r] * V[j,k,r] * 0 = 0, which is automatically satisfied.
    
    So we need: for each i, the matrix U_i (row i of U reshaped) dotted with V_j appropriately = 1.
    Specifically: sum_r U[i*m+j, r] * V[j*p+i, r] = 1 (mod mod) for all i,j.
    
    This means: for each i, let U_i = U[i*m:(i+1)*m, :] (m x rank) and V_i = V[0*p+i : p*p : p, :] (p x rank)
    Wait, V[j*p+i, r] for fixed i and varying j is V[:, i] if we reshape V to (m, p, rank).
    
    So V_reshaped[j, i, r] = V[j*p+i, r]. We need sum_r U[i*m+j, r] * V[j*p+i, r] = 1 for all i,j.
    This is: (U_i * V_i^T)[j, j] ... no.
    
    Actually, for fixed i: sum_j' U[i*m+j', r] * V[j'*p+i, r] summed over r = 1 for each j.
    Wait no: for each (i, j), sum_r U[i*m+j, r] * V[j*p+i, r] = 1.
    
    Let's define: for each i, A_i = U[i*m:(i+1)*m, :] (shape m x rank)
    B_i = V[:, i] where V is reshaped to (m, p, rank), so B_i[j, r] = V[j*p+i, r] (shape m x rank)
    
    We need: sum_r A_i[j, r] * B_i[j, r] = 1 for all j.
    i.e., element-wise dot product of rows of A_i and B_i equals 1.
    
    This is a constraint satisfaction problem. Let's try a randomized local search.
    """
    start = time.time()
    
    # Initialize randomly
    np.random.seed(12345 + rank)
    U = np.random.randint(0, mod, (n*m, rank)) % mod
    V = np.random.randint(0, mod, (m*p, rank)) % mod
    W = np.zeros((p*n, rank), dtype=int)
    for i in range(n):
        for k in range(p):
            if i == k:
                W[k*n+i, :] = 1  # W[k*n+i, r] = 1 for all r when k==i
    
    # Actually, let's not fix W. Let's do a more general search.
    # We'll use a randomized local search: randomly flip entries and check.
    
    def _check(U, V, W):
        """Check if the decomposition is valid over GF(mod)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s % mod != 1:
                        return False
        return True
    
    # Try random initialization + local search
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
        
        np.random.seed(12345 + rank * 100 + attempt)
        U = np.random.randint(0, mod, (n*m, rank)) % mod
        V = np.random.randint(0, mod, (m*p, rank)) % mod
        W = np.random.randint(0, mod, (p*n, rank)) % mod
        
        # Local search: randomly flip entries
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            # Randomly pick a matrix and entry to flip
            mat = np.random.choice(3)
            if mat == 0:
                row, col = np.random.randint(0, n*m), np.random.randint(0, rank)
                old = U[row, col]
                U[row, col] = (old + 1) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                U[row, col] = old
                # Try flipping by 2
                U[row, col] = (old + 2) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                U[row, col] = old
            elif mat == 1:
                row, col = np.random.randint(0, m*p), np.random.randint(0, rank)
                old = V[row, col]
                V[row, col] = (old + 1) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                V[row, col] = old
                V[row, col] = (old + 2) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                V[row, col] = old
            else:
                row, col = np.random.randint(0, p*n), np.random.randint(0, rank)
                old = W[row, col]
                W[row, col] = (old + 1) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                W[row, col] = old
                W[row, col] = (old + 2) % mod
                if _check(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_U, best_V, best_W
                W[row, col] = old
        
        # If we made progress in terms of satisfying more constraints, keep going
        # For now, just restart
    
    return None, None, None

def _try_gradient_descent(rank, n=3, m=3, p=3, max_iters=2000, tol=1e-8, seed=42, time_budget=10.0):
    """Try to find a rank-r decomposition using gradient descent over reals."""
    start = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Target: S[i,j,k] = 1 for all i,j,k where S[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Note: the tensor is T[i,j,k] = 1 for all i,j,k (the all-ones tensor in the (i,j,k) space)
    # Wait, I need to re-examine. The constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # This is the all-ones tensor of shape (n,m,p).
    
    lr = 0.05
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        residual = S - np.ones((n, m, p))
        
        if np.linalg.norm(residual) < tol:
            return U, V, W, True
        
        # Gradients
        grad_U = np.einsum('jkr,kir,ijk->ijr', V_r, W_r, residual)
        grad_V = np.einsum('ijr,kir,ijk->jkr', U_r, W_r, residual)
        grad_W = np.einsum('ijr,jkr,ijk->kir', U_r, V_r, residual)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    final_S = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
    final_res = np.linalg.norm(final_S - np.ones((n, m, p)))
    return U, V, W, final_res < tol

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p  # 27
    
    # Try to find lower rank decompositions
    # Known: rank 23 is achievable for 3x3x3
    for r in range(23, best_rank):
        if time.time() > deadline:
            break
        
        # Try SAT-style search over GF(3)
        try:
            U_sat, V_sat, W_sat = _try_sat_search(r, n, m, p, mod=MOD, time_budget=5.0)
            if U_sat is not None:
                # Verify over reals (convert from GF(3) to reals: 0->0, 1->1, 2->-1)
                U_real = U_sat.astype(float)
                V_real = V_sat.astype(float)
                W_real = W_sat.astype(float)
                # Map 2 -> -1
                U_real[U_real == 2] = -1
                V_real[V_real == 2] = -1
                W_real[W_real == 2] = -1
                
                if _verify_real(U_real, V_real, W_real, n, m, p, tol=1e-6):
                    best = {"U": U_real.tolist(), "V": V_real.tolist(), "W": W_real.tolist()}
                    best_rank = r
                    break
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        # Try gradient descent
        try:
            U_gd, V_gd, W_gd, success = _try_gradient_descent(
                r, n, m, p, max_iters=500, tol=1e-7, seed=42+r, time_budget=5.0
            )
            if success:
                best = {"U": U_gd.tolist(), "V": V_gd.tolist(), "W": W_gd.tolist()}
                best_rank = r
                break
        except Exception:
            continue
    
    return best
# EVOLVE-BLOCK-END
