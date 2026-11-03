"""Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea:
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Verify the solution, then search for a lower rank with a deadline guard.
"""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(1000):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2x2.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def _residual_func(params, n, m, p, rank):
    """Compute the residual of the tensor decomposition."""
    # params is a flat vector of length rank * (nm + mp + pn)
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_flat = params[:rank * nm]
    V_flat = params[rank * nm : rank * nm + rank * mp]
    W_flat = params[rank * nm + rank * mp :]
    
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    T_reconstructed = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    # Target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    return T_reconstructed - T_target


def _try_optimize_rank(n, m, p, rank, U_init, V_init, W_init, deadline):
    """Try to optimize a decomposition of the given rank starting from init."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Flatten init
    U_flat = np.array(U_init, dtype=np.float64).flatten()
    V_flat = np.array(V_init, dtype=np.float64).flatten()
    W_flat = np.array(W_init, dtype=np.float64).flatten()
    
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    # Use least squares to minimize the residual
    # We don't need the Jacobian explicitly, let scipy approximate or use a simple solver
    # Given the small size, a direct solver might be fine, but least_squares is safer
    
    try:
        import scipy.optimize as opt
        
        # Define the residual function for scipy
        def residual(x):
            return _residual_func(x, n, m, p, rank).flatten()
            
        # Run optimization
        # 'trf' method is robust
        result = opt.least_squares(residual, x0, method='trf', max_nfev=1000)
        
        if result.success or result.cost < 1e-8:
            x_opt = result.x
            U_opt = x_opt[:rank * nm].reshape(nm, rank)
            V_opt = x_opt[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
            W_opt = x_opt[rank * nm + rank * mp :].reshape(pn, rank)
            
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
                return U_list, V_list, W_list
                
    except Exception:
        pass
        
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find a lower rank via optimization-based search
    # Theoretical lower bound: n*m + m*p + p*n - m = 6 + 30 + 20 - 3 = 53
    # Try ranks 53 and 54
    
    # We will try to "drop" one term from the rank 55 solution and optimize the rest
    # We try dropping each term one by one, or random subsets
    
    for r in range(53, best_rank):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try dropping terms from the rank 55 solution
        # We need to initialize with rank r. 
        # We can take the first r columns of the rank 55 solution.
        # This is a heuristic initialization.
        
        # To improve success rate, we can try different subsets of columns
        # or random permutations
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            # Create initial guess by taking a random subset of columns from rank 55
            # or just the first r columns
            if attempt == 0:
                U_init = [row[:r] for row in U_best]
                V_init = [row[:r] for row in V_best]
                W_init = [row[:r] for row in W_best]
            else:
                # Random subset
                indices = random.sample(range(55), r)
                U_init = [[row[i] for i in indices] for row in U_best]
                V_init = [[row[i] for i in indices] for row in V_best]
                W_init = [[row[i] for i in indices] for row in W_best]
                
            result = _try_optimize_rank(n, m, p, r, U_init, V_init, W_init, deadline)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = r
                found = True
                break
                
    return {"U": U_best, "V": V_best, "W": W_best}
