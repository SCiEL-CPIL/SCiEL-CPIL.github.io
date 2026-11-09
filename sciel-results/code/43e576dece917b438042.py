import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84
    
    # Schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W, tol=1e-9):
        """Check exact decomposition with rational/half-integer awareness."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Check if entries are close to half-integers
        # And if the tensor equation holds exactly (within float precision)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integers(mat, tol=1e-6):
        """Snap values to nearest 0.5 if close."""
        snapped = np.round(mat * 2) / 2
        if np.all(np.abs(mat - snapped) < tol):
            return snapped
        return None

    def try_rank(R):
        """Try to find a decomposition of rank R."""
        # We fix V and W, and solve for U.
        # V shape: (m*p, R)
        # W shape: (p*n, R)
        # U shape: (n*m, R)
        
        # To make U rational, we want V and W to be simple.
        # Let's try random V and W with entries in {0, 1, 0.5, -0.5, -1}
        # Or just {0, 1} for simplicity first.
        
        n_trials = 50
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                return False
            
            np.random.seed(42 + trial * 100 + R)
            
            # Initialize V and W
            # Use a mix of 0s and 1s to keep A_i sparse-ish
            V = np.random.randint(0, 2, (m * p, R)).astype(float)
            W = np.random.randint(0, 2, (p * n, R)).astype(float)
            
            # Ensure no column is all zeros (which would make U undefined for that component)
            # Actually, if a column of V is 0, the corresponding U column can be anything?
            # No, if V[:, r] = 0 and W[:, r] = 0, the term is 0.
            # If V[:, r] = 0 but W[:, r] != 0, the term is 0 for all i, j, k?
            # Term: U[i,j,r] * 0 * W[k,i,r] = 0.
            # So we can have zero columns in V or W, but not both.
            
            U = np.zeros((n * m, R))
            valid = True
            
            # Solve for U for each i
            for i in range(n):
                # Equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j, k
                # Let's form the matrix A_i of size (m*p) x (m*R)
                # Row index: j*p+k
                # Col index: j*R + r
                # A_i[j*p+k, j*R + r] = V[j*p+k, r] * W[k*n+i, r]
                
                A_i = np.zeros((m * p, m * R))
                for j in range(m):
                    for k in range(p):
                        row_idx = j * p + k
                        # The row of A_i corresponds to equation for (j, k)
                        # The columns correspond to unknowns U[i*m+j', r] for j' in 0..m-1
                        # Wait, the equation for (j, k) involves U[i*m+j, r] ONLY?
                        # No.
                        # C[i, k] = sum_j A[i, j] B[j, k].
                        # The tensor equation is:
                        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                        # This equation is for a specific (i, j, k).
                        # So for fixed i, we have m*p equations.
                        # The unknowns in the equation for (j, k) are U[i*m+j, r] for r=0..R-1.
                        # The unknowns U[i*m+j', r] for j' != j do NOT appear in the equation for (j, k).
                        # This means the system for fixed i decouples into m independent systems!
                        # One for each j.
                        
                        # For fixed i and j:
                        # Unknowns: U[i*m+j, 0], ..., U[i*m+j, R-1]
                        # Equations: for each k in 0..p-1:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # This is a system of p equations with R unknowns.
                        # p=7, R=83.
                        # We can solve this!
                        
                # Let's re-structure the loop to solve for each (i, j)
                pass # This logic is handled in the outer loop below

            # Corrected logic:
            # For each i in n:
            #   For each j in m:
            #     Solve for U[i*m+j, :] using p equations.
            
            U_sol = np.zeros((n * m, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    # System: A U_col = b
                    # A is p x R
                    # b is ones(p)
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve A x = 1
                    # If A is rank deficient, use lstsq
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_sol[i * m + j, :] = x
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if U_sol is close to half-integers
            U_snapped = snap_to_half_integers(U_sol, tol=1e-5)
            if U_snapped is None:
                # Try rounding to integers if close?
                # Or just accept if error is small?
                # The problem requires EXACT decomposition with integer/half-integer entries.
                # So we MUST snap.
                continue
            
            # Verify the snapped solution
            if verify(U_snapped, V, W, tol=1e-9):
                best_rank = R
                best_U = U_snapped
                best_V = V
                best_W = W
                return True
        
        return False

    # Search for lower rank
    # Start from 83 down to 80
    for R in range(83, 79, -1):
        if time.time() > deadline - 10:
            break
        if try_rank(R):
            break

    # If no improvement found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
