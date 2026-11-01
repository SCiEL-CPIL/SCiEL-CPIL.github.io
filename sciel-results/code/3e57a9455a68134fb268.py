import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target ranks to try, from low to high
    # We know 30 works. Let's try 24, 25, 26, 27, 28, 29.
    # If we find one, we return it.
    
    # Time limit guard
    start_time = time.time()
    time_limit = 100.0 # Leave margin for 150s limit
    
    # Define the tensor T indices
    # T[i, j, k] = 1 if j == k? No.
    # The tensor is the multiplication map.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor components are T[i, j, k] = 1 for all i, j, k?
    # No, the tensor is sum_{i,j,k} e_{i,j} \otimes f_{j,k} \otimes g_{i,k}
    # The condition is that the reconstructed tensor matches this.
    # The equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k?
    # Wait. The product is A*B.
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined by T(A, B) = AB.
    # In the basis, T(e_{i,j}, f_{j',k}) = g_{i,k} if j=j', else 0.
    # So the non-zero components are T_{(i,j), (j,k), (i,k)} = 1.
    # There are n*m*p such components.
    
    # We want to find U, V, W of rank R such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # This is a system of n*m*p = 30 equations.
    # Unknowns: U, V, W.
    
    # We can use ALS (Alternating Least Squares).
    # Fix U, V, solve for W.
    # Fix U, W, solve for V.
    # Fix V, W, solve for U.
    
    def solve_for_W(U, V, R):
        # Equations: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # For fixed i, k, the equation is:
        # sum_r (U[i,j,r] V[j,k,r]) W[k,i,r] = 1 for all j?
        # No, the equation is for each (i, j, k).
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is not linear in W if we sum over j?
        # No, for a fixed (i, k), we have m equations (one for each j).
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0..m-1.
        # Let A_{j, r} = U[i*m+j, r] * V[j*p+k, r].
        # Then sum_r A_{j, r} * W[k*n+i, r] = 1 for all j.
        # This is a linear system: A * W_col = 1_vector.
        # Where A is m x R, W_col is R x 1, 1_vector is m x 1.
        # We solve for W_col using least squares.
        
        W = np.zeros((p * n, R))
        for k in range(p):
            for i in range(n):
                # Construct A matrix (m x R)
                A = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b = np.ones(m)
                # Solve A * W_col = b
                # W_col is W[k*n+i, :]
                try:
                    W_col, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = W_col
                except:
                    W[k * n + i, :] = 1.0 / R
        return W

    def solve_for_U(V, W, R):
        # Equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For fixed i, j, we have p equations (one for each k).
        # sum_r (V[j*p+k, r] * W[k*n+i, r]) * U[i*m+j, r] = 1 for all k.
        # Let A_{k, r} = V[j*p+k, r] * W[k*n+i, r].
        # Then sum_r A_{k, r} * U[i*m+j, r] = 1 for all k.
        # A * U_col = 1_vector.
        
        U = np.zeros((n * m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                b = np.ones(p)
                try:
                    U_col, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = U_col
                except:
                    U[i * m + j, :] = 1.0 / R
        return U

    def solve_for_V(U, W, R):
        # Equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For fixed j, k, we have n equations (one for each i).
        # sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] = 1 for all i.
        # Let A_{i, r} = U[i*m+j, r] * W[k*n+i, r].
        # Then sum_r A_{i, r} * V[j*p+k, r] = 1 for all i.
        # A * V_col = 1_vector.
        
        V = np.zeros((m * p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                b = np.ones(n)
                try:
                    V_col, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = V_col
                except:
                    V[j * p + k, :] = 1.0 / R
        return V

    def check_solution(U, V, W, R, tol=1e-6):
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def round_solution(U, V, W, R):
        # Try to round to integers or half-integers
        # Multiply by 2 to make integers, then round
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        if check_solution(U_int, V_int, W_int, R, tol=1e-9):
            return U_int, V_int, W_int
        return None

    # Try ranks from 24 to 29
    for R in range(24, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Multiple random restarts
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(restart)
            U = np.random.rand(n * m, R)
            V = np.random.rand(m * p, R)
            W = np.random.rand(p * n, R)
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                W = solve_for_W(U, V, R)
                U = solve_for_U(V, W, R)
                V = solve_for_V(U, W, R)
                
                if check_solution(U, V, W, R, tol=1e-8):
                    # Try to round
                    rounded = round_solution(U, V, W, R)
                    if rounded:
                        U_int, V_int, W_int = rounded
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
            # If not exact, check if close
            if check_solution(U, V, W, R, tol=1e-4):
                rounded = round_solution(U, V, W, R)
                if rounded:
                    U_int, V_int, W_int = rounded
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # If no better rank found, return schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
