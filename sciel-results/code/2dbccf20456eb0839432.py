from fractions import Fraction
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 19
    
    # Indices
    # U: i (0..n-1), j (0..m-1) -> index i*m + j
    # V: j (0..m-1), k (0..p-1) -> index j*p + k
    # W: k (0..p-1), i (0..n-1) -> index k*n + i

    U_size = n * m
    V_size = m * p
    W_size = p * n

    def solve_sparse_system(M, b):
        """
        Solves M x = b where M is K x rank, b is K x 1.
        Finds a sparse solution by selecting K independent columns via Gaussian elimination with column pivoting.
        Returns x as a list of Fractions of length rank.
        """
        K = len(b)
        r = len(M[0]) if K > 0 else 0
        
        if K == 0:
            return [Fraction(0)] * r

        # Augmented matrix
        M_aug = []
        for i in range(K):
            row = M[i][:]
            row.append(b[i])
            M_aug.append(row)
            
        pivot_cols = []
        row_idx = 0
        
        # We want to find K pivot columns.
        for col in range(r):
            if row_idx >= K:
                break
                
            # Find a row from row_idx to K-1 with non-zero entry in this column
            swap_idx = -1
            for i in range(row_idx, K):
                if M_aug[i][col] != 0:
                    swap_idx = i
                    break
                    
            if swap_idx == -1:
                continue
                
            # Swap rows to bring non-zero to current row
            if swap_idx != row_idx:
                M_aug[row_idx], M_aug[swap_idx] = M_aug[swap_idx], M_aug[row_idx]
                
            pivot_cols.append(col)
            
            # Normalize the pivot row
            pivot_val = M_aug[row_idx][col]
            if pivot_val != 0:
                M_aug[row_idx] = [val / pivot_val for val in M_aug[row_idx]]
                
            # Eliminate this column from all other rows
            for i in range(K):
                if i == row_idx:
                    continue
                factor = M_aug[i][col]
                if factor != 0:
                    M_aug[i] = [M_aug[i][j] - factor * M_aug[row_idx][j] for j in range(r + 1)]
                    
            row_idx += 1
            
        if row_idx < K:
            # Rank deficient
            return [Fraction(0)] * r

        # Solution:
        # The pivot columns correspond to the basic variables.
        # The non-pivot columns correspond to free variables, which we set to 0.
        sol = [Fraction(0)] * r
        for i, col in enumerate(pivot_cols):
            sol[col] = M_aug[i][r]
            
        return sol

    def random_init():
        U = [[Fraction(random.randint(-1, 1)) for _ in range(target_rank)] for _ in range(U_size)]
        V = [[Fraction(random.randint(-1, 1)) for _ in range(target_rank)] for _ in range(V_size)]
        W = [[Fraction(random.randint(-1, 1)) for _ in range(target_rank)] for _ in range(W_size)]
        return U, V, W

    def check_convergence(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    sum_val = Fraction(0)
                    ui_idx = i * m + j
                    vj_idx = j * p + k
                    wk_idx = k * n + i
                    for c in range(target_rank):
                        sum_val += U[ui_idx][c] * V[vj_idx][c] * W[wk_idx][c]
                    if sum_val != 1:
                        return False
        return True

    def is_valid_format(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if val.denominator not in [1, 2]:
                        return False
        return True

    max_iterations = 100
    attempts = 50
    
    # Try to find a rank-19 solution
    for attempt in range(attempts):
        random.seed(attempt * 1000 + 42)
        U, V, W = random_init()
        
        for iter_idx in range(max_iterations):
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
                        row = [V[vj_idx][c] * W[wk_idx][c] for c in range(target_rank)]
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
                        row = [U[ui_idx][c] * W[wk_idx][c] for c in range(target_rank)]
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
                        row = [U[ui_idx][c] * V[vj_idx][c] for c in range(target_rank)]
                        M.append(row)
                    sol = solve_sparse_system(M, b)
                    W[wk_idx] = sol

        if check_convergence(U, V, W):
            if is_valid_format(U, V, W):
                return {
                    "U": [[float(val) for val in row] for row in U],
                    "V": [[float(val) for val in row] for row in V],
                    "W": [[float(val) for val in row] for row in W]
                }

    # Fallback: Schoolbook rank 24
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
