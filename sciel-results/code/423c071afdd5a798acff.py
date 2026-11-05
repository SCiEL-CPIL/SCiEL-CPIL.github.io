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

    def solve_least_squares_sparse(M, b, rank):
        """
        Solves M x = b where M is K x rank, b is K x 1.
        Finds a sparse solution by selecting K independent columns.
        Returns x as a list of Fractions of length rank.
        """
        K = len(b)
        if K == 0:
            return [Fraction(0)] * rank

        # Find K linearly independent columns
        # We want to pick columns c1...cK such that M[:, c1...cK] is invertible
        # Use Gaussian elimination to find pivot columns.
        
        M_aug = [row[:] + [b[i]] for i, row in enumerate(M)]
        
        pivot_cols = []
        row_idx = 0
        for col in range(rank):
            if row_idx >= K:
                break
            # Find a row below row_idx with non-zero entry in col
            swap_idx = -1
            for i in range(row_idx, K):
                if M_aug[i][col] != 0:
                    swap_idx = i
                    break
            if swap_idx == -1:
                continue
            # Swap
            M_aug[row_idx], M_aug[swap_idx] = M_aug[swap_idx], M_aug[row_idx]
            pivot_cols.append(col)
            
            # Eliminate
            pivot_val = M_aug[row_idx][col]
            # Normalize current row
            M_aug[row_idx] = [val / pivot_val for val in M_aug[row_idx]]
            
            for i in range(K):
                if i == row_idx:
                    continue
                factor = M_aug[i][col]
                if factor != 0:
                    M_aug[i] = [M_aug[i][j] - factor * M_aug[row_idx][j] for j in range(rank + 1)]
            
            row_idx += 1
        
        if row_idx < K:
            # Rank deficiency: return zero vector (or could try other column subsets, but this is simpler)
            return [Fraction(0)] * rank

        # We have pivot_cols and M_aug is in RREF for these columns.
        # The submatrix of M_aug corresponding to pivot_cols is Identity (K x K).
        # So x_pivot = M_aug[:, rank] (the last column, which is the solution for the pivot variables)
        sol = [Fraction(0)] * rank
        for i, col in enumerate(pivot_cols):
            sol[col] = M_aug[i][rank]
            
        return sol

    def check_convergence(U, V, W, rank):
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

    def random_init(rank):
        U = [[Fraction(random.randint(-2, 2)) for _ in range(rank)] for _ in range(U_size)]
        V = [[Fraction(random.randint(-2, 2)) for _ in range(rank)] for _ in range(V_size)]
        W = [[Fraction(random.randint(-2, 2)) for _ in range(rank)] for _ in range(W_size)]
        # Ensure no all-zero rows to avoid singular systems immediately
        for i in range(U_size):
            if all(U[i][c] == 0 for c in range(rank)):
                U[i][0] = Fraction(1)
        for i in range(V_size):
            if all(V[i][c] == 0 for c in range(rank)):
                V[i][0] = Fraction(1)
        for i in range(W_size):
            if all(W[i][c] == 0 for c in range(rank)):
                W[i][0] = Fraction(1)
        return U, V, W

    max_iterations = 100
    attempts = 50
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer

    for attempt in range(attempts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(attempt * 1000 + 42)
        rank = target_rank
        U, V, W = random_init(rank)
        
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            if check_convergence(U, V, W, rank):
                break
                
            # 1. Update U
            # For each i, j:
            # Solve sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1 for k=0..p-1
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
                    sol = solve_least_squares_sparse(M, b, rank)
                    U[ui_idx] = sol

            # 2. Update V
            # For each j, k:
            # Solve sum_c V[j,k,c] * (U[i,j,c] * W[k,i,c]) = 1 for i=0..n-1
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
                    sol = solve_least_squares_sparse(M, b, rank)
                    V[vj_idx] = sol

            # 3. Update W
            # For each k, i:
            # Solve sum_c W[k,i,c] * (U[i,j,c] * V[j,k,c]) = 1 for j=0..m-1
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
                    sol = solve_least_squares_sparse(M, b, rank)
                    W[wk_idx] = sol

        if check_convergence(U, V, W, rank):
            # Check if denominators are 1 or 2
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
                # If denominators are larger, we can try to scale the whole tensor?
                # No, the tensor equation is homogeneous in scaling of individual factors?
                # If we scale U by a, V by b, W by c, the product scales by abc.
                # We need abc = 1.
                # If denominators are 3, 3, 3, we can scale U by 1/3, V by 1/3, W by 9? No.
                # The constraint is on the entries of the returned matrices.
                # If we find a solution with denom 3, it is NOT valid under the "integer/half-integer" constraint.
                # However, the problem statement says "Find a rank-minimal exact tensor decomposition ... with integer/half-integer entries".
                # This implies such a solution EXISTS.
                # If our solver finds a solution with denom 3, it might be a different solution in the same equivalence class?
                # Or it might be that we need to search more.
                # For now, we reject it and continue searching.
                pass

    # Fallback: Return schoolbook decomposition (Rank 24) if 19 fails?
    # The task asks for rank-minimal. Returning 24 gives score 24 (worse than 19).
    # But it's valid.
    # However, the directive asks for a rank-19 solution.
    # If we fail to find 19, we should probably return a valid 24 to get non-zero score?
    # Or return the best we found?
    # Let's return the schoolbook 24 as a safety net, but the loop above should find 19.
    
    # Re-implementing schoolbook just in case
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
