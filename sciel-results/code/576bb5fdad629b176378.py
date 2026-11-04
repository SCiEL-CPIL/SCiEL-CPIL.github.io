import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 135
    
    # Pre-compute target tensor T flattened
    # T[i, j, k] = 1 if j == k else 0
    # Shape (n, m, p)
    T = np.zeros((n, m, p))
    for j in range(m):
        for k in range(p):
            if j == k:
                T[:, j, k] = 1.0
    
    # Flatten T for least squares
    # We need to map (i, j, k) to index.
    # U is (n*m, r). Index i*m+j.
    # V is (m*p, r). Index j*p+k.
    # W is (p*n, r). Index k*n+i.
    
    # For ALS:
    # Fix V, W. Solve for U.
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    # For fixed i, j:
    # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = T[i, j, k] for all k.
    # This is a linear system for U[i*m+j, :].
    # Let A be matrix where A[k, r] = V[j*p+k, r] * W[k*n+i, r].
    # Let b be vector b[k] = T[i, j, k].
    # Solve A * U_row = b.
    
    def solve_U(U, V, W):
        new_U = np.zeros_like(U)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                # Construct A (p x r) and b (p)
                A = np.zeros((p, r))
                b = np.zeros(p)
                for k in range(p):
                    b[k] = T[i, j, k]
                    # A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A x = b
                # Use least squares
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = sol
                except:
                    new_U[i*m+j, :] = U[i*m+j, :]
        return new_U

    def solve_V(U, V, W):
        new_V = np.zeros_like(V)
        r = V.shape[1]
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r)) # Wait, equation is for U[i, j, r] * V[j, k, r] * W[k, i, r]
                # For fixed j, k, we solve for V[j*p+k, :].
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k] for all i.
                # Let A[i, r] = U[i*m+j, r] * W[k*n+i, r].
                # Let b[i] = T[i, j, k].
                # Solve A * V_row = b.
                A = np.zeros((n, r))
                b = np.zeros(n)
                for i in range(n):
                    b[i] = T[i, j, k]
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = sol
                except:
                    new_V[j*p+k, :] = V[j*p+k, :]
        return new_V

    def solve_W(U, V, W):
        new_W = np.zeros_like(W)
        r = W.shape[1]
        for k in range(p):
            for i in range(n):
                # Solve for W[k*n+i, :].
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k] for all j.
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r].
                # Let b[j] = T[i, j, k].
                # Solve A * W_row = b.
                A = np.zeros((m, r))
                b = np.zeros(m)
                for j in range(m):
                    b[j] = T[i, j, k]
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = sol
                except:
                    new_W[k*n+i, :] = W[k*n+i, :]
        return new_W

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    target = 1.0 if j == k else 0.0
                    err += (val - target) ** 2
        return err

    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]

    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst

    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try lower ranks
    for target_rank in range(schoolbook_rank - 1, 70, -1):
        if time.time() > deadline:
            break
            
        found = False
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(42 + restart * 1000 + target_rank)
            
            # Initialize randomly
            U_try = np.random.randn(n*m, target_rank)
            V_try = np.random.randn(m*p, target_rank)
            W_try = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(20):
                if time.time() > deadline:
                    break
                U_try = solve_U(U_try, V_try, W_try)
                V_try = solve_V(U_try, V_try, W_try)
                W_try = solve_W(U_try, V_try, W_try)
                
            err = compute_error(U_try, V_try, W_try)
            if err < 1e-6:
                # Verify and clean
                # Check if rounding works
                U_clean = clean_list(to_list(U_try))
                V_clean = clean_list(to_list(V_try))
                W_clean = clean_list(to_list(W_try))
                
                # Verify cleaned version
                err_clean = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(np.array(U_clean[i*m+j]), np.array(V_clean[j*p+k]) * np.array(W_clean[k*n+i]))
                            target = 1.0 if j == k else 0.0
                            err_clean += (val - target) ** 2
                
                if err_clean < 1e-9:
                    best_U = np.array(U_clean)
                    best_V = np.array(V_clean)
                    best_W = np.array(W_clean)
                    best_rank = target_rank
                    found = True
                    break

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
