"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor.
    For all a in [0,n*m), b in [0,m*p), c in [0,p*n):
    sum_r U[a,r] V[b,r] W[c,r] == 1 if (a%m == b//p and b%p == c//n and a//m == c%n) else 0.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - expected))
    return max_err < tol


def _build_expected_tensor(n, m, p):
    """Build the expected multiplication tensor as a 3D array."""
    nm = n * m
    mp = m * p
    pn = p * n
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    return expected


def _try_strassen_block(n, m, p):
    """
    Try a Strassen-like 2x2 block decomposition.
    Split A (n x m) into 2x2 blocks of sizes (n1,n2) x (m1,m2)
    Split B (m x p) into 2x2 blocks of sizes (m1,m2) x (p1,p2)
    Use 7 multiplications instead of 8.
    
    For 4x5x5: n=4, m=5, p=5
    Split: n1=2, n2=2, m1=2, m2=3, p1=2, p2=3
    """
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # 7 Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mx is a product of two smaller matrices.
    # We need to decompose each product using schoolbook (or recursively).
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block_prod(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, sign=1):
        """Add schoolbook terms for A_rows x A_cols * B_rows x B_cols -> C_rows x C_cols."""
        # A_rows: list of row indices in A, A_cols: list of col indices in A
        # B_rows: list of row indices in B, B_cols: list of col indices in B
        # C_rows: list of row indices in C, C_cols: list of col indices in C
        for i in C_rows:
            for k in C_cols:
                for j in A_cols:
                    # A[i,j] * B[j,k] -> C[i,k]
                    U_row = i * m + j
                    V_row = j * p + k
                    W_row = k * n + i
                    U_list.append([0.0] * len(U_list[0]) if U_list else [0.0])
                    # We'll build these differently - use a dict approach
        # This approach is getting complex. Let me use a different method.
    
    # Simpler approach: build the 7 Strassen products as separate schoolbook decompositions
    # with appropriate sign and offset
    
    # Define the 7 products with their A-submatrix, B-submatrix, and contribution to C
    # Each product: (A_sub)(B_sub) contributes with a coefficient to a C_sub
    
    products = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11: rows [0,n1), cols [0,m1)
    # A22: rows [n1,n), cols [m1,m)
    # B11: rows [0,m1), cols [0,p1)
    # B22: rows [m1,m), cols [p1,p)
    # M1 contributes: +1 to C11, +1 to C22
    
    # This is getting very complex to handle the sums. Let me use a linear algebra approach.
    # Instead, I'll construct the full decomposition by expressing each of the 7 Strassen
    # products as a sum of schoolbook terms with appropriate coefficients.
    
    # Actually, let me take a step back. The Strassen algorithm for 4x5x5:
    # We can pad to 4x8x8 or use a different blocking.
    # 
    # Alternative: Use the known fact that for 4x5x5, we can use a 2x2 block with
    # sizes (2,2)x(2,3) and (2,3)x(2,3), giving 7 products of sizes:
    # M1: (2x2)(2x2) -> 4 mults
    # M2: (2x3)(2x2) -> 6 mults  
    # M3: (2x2)(3x3) -> 6 mults
    # M4: (2x3)(3x2) -> 6 mults
    # M5: (2x3)(3x3) -> 9 mults
    # M6: (2x2)(2x3) -> 6 mults
    # M7: (2x3)(3x3) -> 9 mults
    # Total: 4+6+6+6+9+6+9 = 46 mults (vs 100 schoolbook)
    
    # Let me build this properly.
    # A is 4x5, B is 5x5
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x3, C21 is 2x2, C22 is 2x3
    
    # Strassen:
    # M1 = (A11 + A22)(B11 + B22) : (2x5)(5x5) -> wait, that's not right
    # 
    # Actually for rectangular Strassen, we need to be more careful.
    # Let me just try a direct rank-70 search instead.
    
    return None  # Too complex to implement correctly in this form


def _try_integer_search(n, m, p, target_rank, num_trials=20, seed=42):
    """
    Try to find a rank-target_rank decomposition with integer/half-integer entries.
    Strategy: 
    1. Initialize with random values
    2. Use gradient descent to minimize the tensor error
    3. Round to nearest half-integer
    4. Check if the rounded solution still satisfies the tensor identity
    """
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    expected = _build_expected_tensor(n, m, p)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    best_rounded = False
    
    for trial in range(num_trials):
        # Initialize with small random integers or half-integers
        scale = rng.choice([1, 2, 4])
        U = rng.randint(-scale, scale + 1, (nm, target_rank)).astype(np.float64)
        V = rng.randint(-scale, scale + 1, (mp, target_rank)).astype(np.float64)
        W = rng.randint(-scale, scale + 1, (pn, target_rank)).astype(np.float64)
        
        # Gradient descent
        lr = 0.001
        for step in range(500):
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = T - expected
            loss = np.sum(err**2)
            
            if loss < best_err:
                best_err = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 1e-10:
                eff_lr = lr * 100 / (1 + grad_norm)
            else:
                eff_lr = lr
            
            U -= eff_lr * dU
            V -= eff_lr * dV
            W -= eff_lr * dW
        
        # Try rounding to half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        T_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
        err_round = np.max(np.abs(T_round - expected))
        
        if err_round < 1e-9:
            best_rounded = True
            best_U, best_V, best_W = U_round, V_round, W_round
            best_err = err_round
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def _try_polish(n, m, p, U_init, V_init, W_init, max_steps=2000, tol=1e-12):
    """High-precision polish pass using Newton-like iterations."""
    nm = n * m
    mp = m * p
    pn = p * n
    expected = _build_expected_tensor(n, m, p)
    
    U = U_init.copy().astype(np.float64)
    V = V_init.copy().astype(np.float64)
    W = W_init.copy().astype(np.float64)
    
    for step in range(max_steps):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = T - expected
        loss = np.sum(err**2)
        
        if loss < tol:
            break
        
        # Compute gradients
        dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
        
        # Adam-like update with momentum
        # Simple: use a small fixed step
        lr = 0.0001
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    final_T = np.einsum('ar,br,cr->abc', U, V, W)
    final_err = np.max(np.abs(final_T - expected))
    
    if final_err < 1e-9:
        return U, V, W, final_err
    return None, None, None, final_err


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 100
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Target ranks to try, in order of preference
    target_ranks = [70, 72, 75, 80, 85, 90, 95]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try integer search
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            U, V, W, err = _try_integer_search(n, m, p, target_rank, num_trials=5, seed=target_rank*100+attempt*7)
            
            if U is not None and err < 1e-9:
                if _verify(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # If we found a solution, try to polish it
    if best_rank < 100:
        if time.time() < deadline - 10:
            U_p, V_p, W_p, err_p = _try_polish(n, m, p, best_U, best_V, best_W, max_steps=1000)
            if U_p is not None and err_p < 1e-9:
                best_U, best_V, best_W = U_p, V_p, W_p
    
    # Convert to lists for the return interface
    U_list = [[float(x) for x in row] for row in np.array(best_U).tolist()]
    V_list = [[float(x) for x in row] for row in np.array(best_V).tolist()]
    W_list = [[float(x) for x in row] for row in np.array(best_W).tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
