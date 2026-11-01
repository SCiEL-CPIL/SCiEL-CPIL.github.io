"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # --- Schoolbook decomposition (rank = n*m*p) ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # --- Fast verification using einsum ---
    def verify_fast(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j."""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Check for integer/half-integer entries ---
    def is_valid_entries(*mats):
        for mat in mats:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # --- Convert to list format ---
    def to_list(mat):
        # Round to nearest half-integer for cleanliness
        mat_rounded = np.round(mat * 2) / 2.0
        # Check if all are integers
        if np.allclose(mat_rounded, np.round(mat_rounded), atol=1e-9):
            return np.round(mat_rounded).astype(int).tolist()
        else:
            # Return as float list with half-integer values
            return mat_rounded.tolist()

    # --- Constructive decomposition using Strassen for 2x2 blocks ---
    def construct_strassen_blocks():
        """
        Decomposes M_{2,2,7} using:
        - 3 blocks of 2 columns using Strassen (rank 7 each)
        - 1 block of 1 column using Schoolbook (rank 4)
        Total rank = 3*7 + 4 = 25.
        """
        r = 25
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(j_start):
            nonlocal col
            # j_start is 0, 2, 4
            j1 = j_start
            j2 = j_start + 1
            
            # M1 = (A00 + A11) * (B0,j1 + B1,j2)
            U[0*m+0][col] += 1
            U[1*m+1][col] += 1
            V[0*p+j1][col] += 1
            V[1*p+j2][col] += 1
            W[j1*n+0][col] += 1
            W[j2*n+1][col] += 1
            col += 1
            
            # M2 = (A10 + A11) * B0,j1
            U[1*m+0][col] += 1
            U[1*m+1][col] += 1
            V[0*p+j1][col] += 1
            W[j1*n+1][col] += 1
            W[j2*n+1][col] += -1
            col += 1
            
            # M3 = A00 * (B1,j1 - B1,j2)
            U[0*m+0][col] += 1
            V[1*p+j1][col] += 1
            V[1*p+j2][col] += -1
            W[j1*n+0][col] += 1
            W[j2*n+0][col] += 1
            col += 1
            
            # M4 = A11 * (B0,j2 - B0,j1)
            U[1*m+1][col] += 1
            V[0*p+j2][col] += 1
            V[0*p+j1][col] += -1
            W[j1*n+0][col] += 1
            W[j1*n+1][col] += 1
            col += 1
            
            # M5 = (A00 + A01) * B1,j2
            U[0*m+0][col] += 1
            U[0*m+1][col] += 1
            V[1*p+j2][col] += 1
            W[j1*n+0][col] += -1
            W[j2*n+0][col] += 1
            col += 1
            
            # M6 = (A10 - A00) * (B0,j1 + B0,j2)
            U[1*m+0][col] += 1
            U[0*m+0][col] += -1
            V[0*p+j1][col] += 1
            V[0*p+j2][col] += 1
            W[j2*n+1][col] += 1
            col += 1
            
            # M7 = (A01 - A11) * (B1,j1 + B1,j2)
            U[0*m+1][col] += 1
            U[1*m+1][col] += -1
            V[1*p+j1][col] += 1
            V[1*p+j2][col] += 1
            W[j1*n+0][col] += 1
            W[j2*n+1][col] += 1
            col += 1
            
        def add_schoolbook_block(j_start):
            nonlocal col
            j1 = j_start
            
            # A00 * B0,j1 -> C0,j1
            U[0*m+0][col] += 1
            V[0*p+j1][col] += 1
            W[j1*n+0][col] += 1
            col += 1
            
            # A01 * B1,j1 -> C0,j1
            U[0*m+1][col] += 1
            V[1*p+j1][col] += 1
            W[j1*n+0][col] += 1
            col += 1
            
            # A10 * B0,j1 -> C1,j1
            U[1*m+0][col] += 1
            V[0*p+j1][col] += 1
            W[j1*n+1][col] += 1
            col += 1
            
            # A11 * B1,j1 -> C1,j1
            U[1*m+1][col] += 1
            V[1*p+j1][col] += 1
            W[j1*n+1][col] += 1
            col += 1
            
        # Apply blocks
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        add_schoolbook_block(6)
        
        return {"U": U, "V": V, "W": W}

    # Initialize best with Strassen blocks (rank 25)
    best = construct_strassen_blocks()
    best_rank = 25
    
    U_mat = np.array(best["U"], dtype=float)
    V_mat = np.array(best["V"], dtype=float)
    W_mat = np.array(best["W"], dtype=float)
    
    if not verify_fast(U_mat, V_mat, W_mat):
        # Fallback to schoolbook if construction fails
        best = schoolbook()
        best_rank = n * m * p
        U_mat = np.array(best["U"], dtype=float)
        V_mat = np.array(best["V"], dtype=float)
        W_mat = np.array(best["W"], dtype=float)
    else:
        # Ensure entries are clean integers
        best["U"] = np.round(U_mat).astype(int).tolist()
        best["V"] = np.round(V_mat).astype(int).tolist()
        best["W"] = np.round(W_mat).astype(int).tolist()

    # --- New Strategy: Structured Random Search with Analytic Completion ---
    
    def compute_residual(U_mat, V_mat):
        """
        Computes the residual norm for a given U, V pair.
        We solve for W analytically to minimize the residual.
        Residual is sum over (j,i) of || A_{ji} W_{ji} - 1 ||^2
        where A_{ji}[k, c] = U[i,k,c] * V[k,j,c]
        """
        residual = 0.0
        for i in range(n):
            for j in range(p):
                # A is (m, r)
                A = np.zeros((m, U_mat.shape[1]))
                for k in range(m):
                    A[k, :] = U_mat[i * m + k, :] * V_mat[k * p + j, :]
                
                b = np.ones(m)
                
                # Solve least squares
                try:
                    sol, residuals, rank, s = LA.lstsq(A, b, rcond=None)
                    if residuals.size > 0:
                        residual += residuals[0]
                    else:
                        # If rank < m, check if consistent
                        # If consistent, residual is 0. If not, it's in 'residuals' only if m > rank?
                        # Actually, lstsq returns residuals only if m > rank.
                        # If m <= rank, residuals is empty, but solution exists if consistent.
                        # We need to check A @ sol - b
                        err = A @ sol - b
                        residual += np.dot(err, err)
                except:
                    return float('inf')
        return residual

    def random_sparse_matrix(rows, cols, density, seed):
        """Generate a sparse matrix with entries in {-1, 0, 1}."""
        rng = np.random.RandomState(seed)
        mat = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                if rng.random() < density:
                    mat[r, c] = rng.choice([-1, 1])
        return mat

    def hill_climb(U_mat, V_mat, max_steps=1000):
        """
        Local search: randomly flip entries in U or V, accept if residual decreases.
        """
        current_res = compute_residual(U_mat, V_mat)
        if current_res == 0:
            return U_mat, V_mat, 0.0
            
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Randomly select a matrix to perturb
            if np.random.random() < 0.5:
                target_mat = U_mat
                other_mat = V_mat
            else:
                target_mat = V_mat
                other_mat = U_mat
            
            # Create a copy
            new_mat = target_mat.copy()
            
            # Pick a random entry to flip (or set to 0/1/-1)
            r_idx = np.random.randint(0, new_mat.shape[0])
            c_idx = np.random.randint(0, new_mat.shape[1])
            
            # Try changing to a different value
            current_val = new_mat[r_idx, c_idx]
            options = [-1, 0, 1]
            if current_val in options:
                options.remove(current_val)
            
            if options:
                new_val = np.random.choice(options)
                new_mat[r_idx, c_idx] = new_val
                
                # Compute new residual
                new_res = compute_residual(new_mat, other_mat)
                
                if new_res < current_res:
                    target_mat[:] = new_mat
                    current_res = new_res
                    if current_res == 0:
                        break
        
        return U_mat, V_mat, current_res

    # Start search from rank 24 down to 10
    for r in range(best_rank - 1, 9, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for init in range(20):
            if time.time() > deadline:
                break
            
            seed = 12345 + r * 100 + init
            # Density: need enough non-zeros to be solvable. 
            # For rank r, we have n*m=4 rows in U, m*p=14 rows in V.
            # Higher density increases chance of full rank A_{ji}
            density = 0.6 
            
            U_rand = random_sparse_matrix(n * m, r, density, seed)
            V_rand = random_sparse_matrix(m * p, r, density, seed + 1)
            
            # Initial check
            res = compute_residual(U_rand, V_rand)
            if res < 1e-9:
                # Verify full decomposition
                # Solve for W
                W_rand = np.zeros((p * n, r))
                for i in range(n):
                    for j in range(p):
                        A = np.zeros((m, r))
                        for k in range(m):
                            A[k, :] = U_rand[i * m + k, :] * V_rand[k * p + j, :]
                        b = np.ones(m)
                        sol, _, _, _ = LA.lstsq(A, b, rcond=None)
                        W_rand[j * n + i, :] = sol
                        
                if verify_fast(U_rand, V_rand, W_rand):
                    if is_valid_entries(U_rand, V_rand, W_rand):
                        best = {
                            "U": to_list(U_rand),
                            "V": to_list(V_rand),
                            "W": to_list(W_rand)
                        }
                        best_rank = r
                        found = True
                        break
            
            # If not immediately zero, try hill climbing
            U_rand, V_rand, final_res = hill_climb(U_rand.copy(), V_rand.copy(), max_steps=50)
            
            if final_res < 1e-9:
                # Solve for W
                W_rand = np.zeros((p * n, r))
                for i in range(n):
                    for j in range(p):
                        A = np.zeros((m, r))
                        for k in range(m):
                            A[k, :] = U_rand[i * m + k, :] * V_rand[k * p + j, :]
                        b = np.ones(m)
                        sol, _, _, _ = LA.lstsq(A, b, rcond=None)
                        W_rand[j * n + i, :] = sol
                        
                if verify_fast(U_rand, V_rand, W_rand):
                    if is_valid_entries(U_rand, V_rand, W_rand):
                        best = {
                            "U": to_list(U_rand),
                            "V": to_list(V_rand),
                            "W": to_list(W_rand)
                        }
                        best_rank = r
                        found = True
                        break
        
        if found:
            # If we found a lower rank, we can try to go lower, but we break to save time?
            # No, we continue the loop to try lower ranks.
            pass

    return best
# EVOLVE-BLOCK-END
