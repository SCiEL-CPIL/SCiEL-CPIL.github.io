import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R, V: 12 x R, W: 12 x R
    
    # Build target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
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

    def verify_exact(U, V, W):
        """
        Verify if U, V, W (lists of lists of numbers) form an exact decomposition.
        Entries must be half-integers (integer or x.5).
        """
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # val should be float or int
                    # Check if 2*val is an integer
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        # Check tensor decomposition
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Target is 1 if (i,j,k) matches, 0 otherwise.
        
        # To be fast and exact, we can use Fractions or just careful integer arithmetic.
        # Since entries are half-integers, 2*U, 2*V, 2*W are integers.
        # Let U' = 2U, V' = 2V, W' = 2W.
        # sum (U'/2) (V'/2) (W'/2) = (1/8) sum U' V' W'
        # We need sum U' V' W' = 8 for target 1, and 0 for target 0.
        
        U_int = [[int(round(2*val)) for val in row] for row in U]
        V_int = [[int(round(2*val)) for val in row] for row in V]
        W_int = [[int(round(2*val)) for val in row] for row in W]
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                    if s != 8:
                        return False
        
        # Check zero entries
        # Iterate over all a, b, c
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                if j1 != j2:
                    # Must be 0
                    for c in range(p*n):
                        s = 0
                        for r in range(rank):
                            s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                        if s != 0:
                            return False
                else:
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        if i1 == i2 and k1 == k2:
                            continue # Already checked
                        else:
                            s = 0
                            for r in range(rank):
                                s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                            if s != 0:
                                return False
        return True

    def snap_to_half_int(val):
        return round(val * 2) / 2.0

    def fast_als_decompose(T, R, max_iters=100, tol=1e-9, seed=42, use_snap=True):
        """
        ALS with optional snapping to half-integers.
        """
        np.random.seed(seed)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        if use_snap:
            U = np.round(U * 2) / 2.0
            V = np.round(V * 2) / 2.0
            W = np.round(W * 2) / 2.0

        # Precompute index mappings
        a_to_bc = {}
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a_to_bc[a] = [(j * p + k, k * n + i) for k in range(p)]
        
        b_to_ac = {}
        for j in range(m):
            for k in range(p):
                b = j * p + k
                b_to_ac[b] = [(i * m + j, k * n + i) for i in range(n)]
        
        c_to_ab = {}
        for k in range(p):
            for i in range(n):
                c = k * n + i
                c_to_ab[c] = [(i * m + j, j * p + k) for j in range(m)]
        
        err = float('inf')
        for it in range(max_iters):
            # Update U
            for a in range(n*m):
                bc_list = a_to_bc[a]
                A = np.zeros((len(bc_list), R))
                y = np.zeros(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A[idx, :] = V[b, :] * W[c, :]
                    y[idx] = T[a, b, c]
                if np.linalg.matrix_rank(A) < R:
                     # Handle underdetermined or rank-deficient
                     sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                else:
                     sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                if use_snap:
                    U[a, :] = np.round(U[a, :] * 2) / 2.0

            # Update V
            for b in range(m*p):
                ac_list = b_to_ac[b]
                A = np.zeros((len(ac_list), R))
                y = np.zeros(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A[idx, :] = U[a, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                if use_snap:
                    V[b, :] = np.round(V[b, :] * 2) / 2.0

            # Update W
            for c in range(p*n):
                ab_list = c_to_ab[c]
                A = np.zeros((len(ab_list), R))
                y = np.zeros(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A[idx, :] = U[a, :] * V[b, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
                if use_snap:
                    W[c, :] = np.round(W[c, :] * 2) / 2.0

            # Calculate error
            # Only check target entries for speed during optimization
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < tol:
                # Check zero entries before returning
                zero_err = 0.0
                for a in range(n*m):
                    i1 = a // m
                    j1 = a % m
                    for b in range(m*p):
                        j2 = b // p
                        k2 = b % p
                        if j1 != j2:
                            for c in range(p*n):
                                s = 0.0
                                for r in range(R):
                                    s += U[a, r] * V[b, r] * W[c, r]
                                zero_err += s * s
                        else:
                            for c in range(p*n):
                                k1 = c // n
                                i2 = c % n
                                if i1 != i2 or k1 != k2:
                                    s = 0.0
                                    for r in range(R):
                                        s += U[a, r] * V[b, r] * W[c, r]
                                    zero_err += s * s
                if zero_err < tol:
                    return U, V, W, err
                else:
                    # If zero entries are not satisfied, keep going
                    err += np.sqrt(zero_err)
                    
        return U, V, W, err

    def try_rank(R, deadline, num_restarts=20):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
            seed = restart * 13 + 7
            U, V, W, err = fast_als_decompose(T, R, max_iters=50, tol=1e-5, seed=seed, use_snap=True)
            
            # Convert to lists for exact verification
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
            # If floating point error is small, try to rationalize more aggressively
            if err < 1e-3:
                # Try to find nearest half-integers that work
                # This is a local search on the lattice
                # Simple hill climbing: try flipping bits of U, V, W?
                pass
        
        return None

    # Search for minimal R
    # Lower bound 27? Try from 27 to 35.
    best_result = schoolbook()
    
    # We want to find R < 36
    # Allocate time to lower ranks first? Or spread?
    # Since 36 is the baseline, any R < 36 is an improvement.
    # Try R from 27 to 35.
    
    # Time budget: 110s.
    # 9 ranks. ~12s per rank.
    for R in range(27, 36):
        if time.time() > deadline - 5:
            break
        res = try_rank(R, deadline - 5, num_restarts=10)
        if res is not None:
            return res

    return best_result
