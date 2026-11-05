from fractions import Fraction
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 19  # Target minimal rank
    
    U_size = n * m
    V_size = m * p
    W_size = p * n

    def solve_square_system(M, b):
        """
        Solves M x = b where M is K x K and b is K x 1.
        Returns x as a list of Fractions of length K.
        Assumes M is invertible.
        """
        K = len(b)
        # Augmented matrix
        M_aug = [M[i][:] + [b[i]] for i in range(K)]
        
        # Gaussian Elimination
        for i in range(K):
            # Find pivot
            pivot_row = -1
            for r in range(i, K):
                if M_aug[r][i] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return None # Singular
                
            # Swap
            if pivot_row != i:
                M_aug[i], M_aug[pivot_row] = M_aug[pivot_row], M_aug[i]
                
            # Normalize
            pivot_val = M_aug[i][i]
            M_aug[i] = [val / pivot_val for val in M_aug[i]]
            
            # Eliminate
            for r in range(K):
                if r == i:
                    continue
                factor = M_aug[r][i]
                if factor != 0:
                    M_aug[r] = [M_aug[r][j] - factor * M_aug[i][j] for j in range(K + 1)]
                    
        return [M_aug[i][K] for i in range(K)]

    def solve_sparse_system(M, b):
        """
        Solves M x = b where M is K x rank, b is K x 1.
        Finds a sparse solution by selecting K independent columns.
        """
        K = len(b)
        r = len(M[0]) if K > 0 else 0
        
        if K == 0:
            return [Fraction(0)] * r

        # Find K pivot columns
        M_copy = [row[:] for row in M]
        pivot_cols = []
        row_idx = 0
        for col in range(r):
            if row_idx >= K:
                break
            # Find a row below row_idx with non-zero entry in col
            swap_idx = -1
            for i in range(row_idx, K):
                if M_copy[i][col] != 0:
                    swap_idx = i
                    break
            if swap_idx == -1:
                continue
            # Swap
            M_copy[row_idx], M_copy[swap_idx] = M_copy[swap_idx], M_copy[row_idx]
            pivot_cols.append(col)
            
            # Eliminate below and above? Just below for finding pivots is enough, 
            # but for solving we need the reduced system.
            # Let's just extract the submatrix and solve it separately.
            row_idx += 1
            
        if row_idx < K:
            return [Fraction(0)] * r

        # Extract submatrix M_sub (K x K) and b is same
        M_sub = []
        for i in range(K):
            M_sub.append([M[i][col] for col in pivot_cols])
            
        sol_sub = solve_square_system(M_sub, b)
        if sol_sub is None:
            return [Fraction(0)] * r
            
        sol = [Fraction(0)] * r
        for i, col in enumerate(pivot_cols):
            sol[col] = sol_sub[i]
            
        return sol

    def random_init():
        U = [[Fraction(random.randint(-1, 1)) for _ in range(rank)] for _ in range(U_size)]
        V = [[Fraction(random.randint(-1, 1)) for _ in range(rank)] for _ in range(V_size)]
        W = [[Fraction(random.randint(-1, 1)) for _ in range(rank)] for _ in range(W_size)]
        return U, V, W

    def check_convergence(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    sum_val = Fraction(0)
                    ui_idx = i * m + j
                    vj_idx = j * p + k
                    wk_idx = k * n + i
                    for c in range(rank):
                        sum_val += U[ui_idx][c] * V[vj_idx][c] * W[wk_idx][c]
                    if sum_val != 1:
                        return False
        return True

    max_iterations = 100
    attempts = 50
    
    for attempt in range(attempts):
        random.seed(attempt * 100 + 42)
        U, V, W = random_init()
        
        for _ in range(max_iterations):
            if check_convergence(U, V, W):
                break
                
            # 1. Update U
            for i in range(n):
                for j in range(m):
                    ui_idx = i * m + j
                    M = []
                    b = [Fraction(1)] * p
                    for k in range(p):
                        vj_idx = j * p + k
                        wk_idx = k * n + i
                        row = [V[vj_idx][c] * W[wk_idx][c] for c in range(rank)]
                        M.append(row)
                    sol = solve_sparse_system(M, b)
                    U[ui_idx] = sol

            # 2. Update V
            for j in range(m):
                for k in range(p):
                    vj_idx = j * p + k
                    M = []
                    b = [Fraction(1)] * n
                    for i in range(n):
                        ui_idx = i * m + j
                        wk_idx = k * n + i
                        row = [U[ui_idx][c] * W[wk_idx][c] for c in range(rank)]
                        M.append(row)
                    sol = solve_sparse_system(M, b)
                    V[vj_idx] = sol

            # 3. Update W
            for k in range(p):
                for i in range(n):
                    wk_idx = k * n + i
                    M = []
                    b = [Fraction(1)] * m
                    for j in range(m):
                        ui_idx = i * m + j
                        vj_idx = j * p + k
                        row = [U[ui_idx][c] * V[vj_idx][c] for c in range(rank)]
                        M.append(row)
                    sol = solve_sparse_system(M, b)
                    W[wk_idx] = sol

        if check_convergence(U, V, W):
            # Check if entries are integers or half-integers
            valid = True
            for mat in [U, V, W]:
                for row in mat:
                    for val in row:
                        if val.denominator not in [1, 2]:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return {
                    "U": [[float(val) for val in row] for row in U],
                    "V": [[float(val) for val in row] for row in V],
                    "W": [[float(val) for val in row] for row in W]
                }
            else:
                # If we have an exact solution but denominators are not 1 or 2,
                # we can try to scale? No, scaling changes the tensor value.
                # However, if the solution is exact, the denominators MUST be 1 or 2 for the "integer/half-integer" constraint?
                # Actually, the constraint is on the *entries*. If we find a solution with denom 3, it's a valid tensor decomposition, 
                # but does it satisfy the "integer/half-integer entries" constraint? The prompt says "with integer/half-integer entries".
                # This implies the specific decomposition we return must have these properties.
                # If we fail this, we continue searching.
                pass

    # Fallback: Schoolbook Rank 24
    rank = n * m * p
    U_school = [[0] * rank for _ in range(n * m)]
    V_school = [[0] * rank for _ in range(m * p)]
    W_school = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    return {"U": U_school, "V": V_school, "W": W_school}
