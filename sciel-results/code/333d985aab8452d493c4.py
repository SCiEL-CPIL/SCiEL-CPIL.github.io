"""Matrix multiplication 2,3,5 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Indices for the tensor
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # Tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i) = 1
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    def snap_to_half(x):
        """Snap a value to the nearest half-integer (0, 0.5, 1, 1.5, ...)."""
        # Multiply by 2, round, divide by 2
        return round(x * 2) / 2.0

    def snap_matrix(M):
        """Snap a numpy array to half-integers."""
        return np.round(M * 2) / 2.0

    def verify(U, V, W):
        """
        Verify if the decomposition is exact using Fraction arithmetic.
        U, V, W are lists of lists of floats (snapped to half-integers).
        """
        r = len(U[0])
        # Convert to Fractions for exact check
        # Optimization: only check if it looks close first? No, just do it.
        # 30 entries to check.
        for idx in indices:
            a, b, c = idx
            s = Fraction(0)
            for rr in range(r):
                s += Fraction(U[a][rr]) * Fraction(V[b][rr]) * Fraction(W[c][rr])
            if s != 1:
                return False
        return True

    def check_numerical(U, V, W, tol=1e-9):
        """Fast numerical check before exact check."""
        r = U.shape[2]
        # U, V, W are (n*m, r), etc.
        # We need U[a, :] dot V[b, :] dot W[c, :] ? No, it's element-wise product sum.
        # sum_r U[a, r] V[b, r] W[c, r]
        # This is the trace of U[a,:].T * diag(V[b,:]) * W[c,:].T ? No.
        # It's the Hadamard product sum.
        
        # Efficient check:
        # For each (a,b,c) in indices, compute sum(U[a,:]*V[b,:]*W[c,:])
        # We can precompute M_ab = U[a,:] * V[b,:] (element-wise)
        # Then check dot(M_ab, W[c,:]) == 1
        
        # Let's just loop, it's small.
        for a, b, c in indices:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(s - 1.0) > tol:
                return False
        return True

    def run_als(U, V, W, max_iter=100):
        """
        Run Alternating Least Squares to minimize the error.
        We want sum_r U[a,r] V[b,r] W[c,r] = 1 for all (a,b,c) in indices.
        
        Fix V, W: Solve for U.
        The equation is linear in U[a, :].
        For a fixed a, the terms involving U[a, r] are those where the first index is a.
        Let S_a = set of (b, c) such that (a, b, c) is in indices.
        We want: sum_{(b,c) in S_a} (V[b, :] * W[c, :]) dot U[a, :] = 1 for each (b,c)?
        No. The constraint is:
        For each (a,b,c) in indices: U[a, :] dot (V[b, :] * W[c, :]) = 1.
        
        So for a fixed 'a', we have a system of linear equations for the vector U[a, :].
        Let X_a be a matrix where each row is (V[b, :] * W[c, :]) for some (b,c) compatible with a.
        We want X_a @ U[a, :] = 1 (vector of ones).
        
        Since the system might be underdetermined (rank < number of constraints) or overdetermined,
        we use least squares.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute compatibility maps
        # For each a (0..n*m-1), list of (b, c)
        comp_a = {a: [] for a in range(n * m)}
        comp_b = {b: [] for b in range(m * p)}
        comp_c = {c: [] for c in range(p * n)}
        
        for a, b, c in indices:
            comp_a[a].append((b, c))
            comp_b[b].append((a, c))
            comp_c[c].append((a, b))
            
        for it in range(max_iter):
            # Update U
            for a in range(n * m):
                if not comp_a[a]: continue
                # Build matrix X
                rows = []
                for b, c in comp_a[a]:
                    # V[b, :] * W[c, :]
                    rows.append(V[b, :] * W[c, :])
                X = np.array(rows) # shape (num_constraints, r)
                y = np.ones(len(rows))
                # Solve X U[a,:] = y
                # If X is rank deficient, use pinv or lstsq
                try:
                    U[a, :] = np.linalg.lstsq(X, y, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(m * p):
                if not comp_b[b]: continue
                rows = []
                for a, c in comp_b[b]:
                    rows.append(U[a, :] * W[c, :])
                X = np.array(rows)
                y = np.ones(len(rows))
                try:
                    V[b, :] = np.linalg.lstsq(X, y, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(p * n):
                if not comp_c[c]: continue
                rows = []
                for a, b in comp_c[c]:
                    rows.append(U[a, :] * V[b, :])
                X = np.array(rows)
                y = np.ones(len(rows))
                try:
                    W[c, :] = np.linalg.lstsq(X, y, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            if check_numerical(U, V, W, tol=1e-6):
                # Snap and verify exactly
                U_snap = snap_matrix(U)
                V_snap = snap_matrix(V)
                W_snap = snap_matrix(W)
                # Convert to list of lists for verify
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                if verify(U_list, V_list, W_list):
                    return U_list, V_list, W_list, True
        return None

    best_result = None
    best_rank = n * m * p + 1
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 15
    # Upper bound is 30
    # Let's start from 15 and go up.
    # But 15 is very tight. Random init might not work well.
    # Let's try a range.
    
    for rank in range(15, 31):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        max_restarts = 50 if rank < 25 else 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            # Using normal distribution
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            res = run_als(U, V, W, max_iter=200)
            if res:
                U_l, V_l, W_l, _ = res
                best_result = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = rank
                found = True
                break # Found for this rank, move to next lower rank? No, we are going up.
                # Actually we want the MINIMAL rank. So if we find one at rank R, we should stop?
                # Yes, because we are iterating from low to high.
                # But wait, the loop is `for rank in range(15, 31)`.
                # If we find a solution at 15, we return it.
                # If not, we try 16, etc.
                break
        
        if found:
            return best_result

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
