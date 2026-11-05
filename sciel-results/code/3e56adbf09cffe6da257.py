"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy: 
    1. Build rank-33 Strassen block decomposition (3 blocks of 2x2x2 Strassen + 2x1x2 standard)
    2. Try to find a rank-27 decomposition using a different block structure
    3. Use algebraic search to find lower rank decompositions
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    def try_rank_27_decomposition():
        """
        Try to construct a rank-27 decomposition using a different block structure.
        
        Idea: Split the 3x6 B matrix differently.
        - View m=3 as a 2+1 split (2 rows + 1 row)
        - For the 2x2 part of A (rows i=0,1; cols j=0,1) and 2x6 part of B (rows j=0,1; all 6 cols):
          This is a 2x2 by 2x6 multiplication, which can be viewed as 3 independent 2x2 multiplications
          (each with 6/2=3 columns), so we use Strassen 3 times = 21 multiplications
        - For the remaining 2x1 part of A (rows i=0,1; col j=2) and 1x6 part of B (row j=2; all 6 cols):
          This is a 2x1 by 1x6 multiplication = 2*1*6 = 12 multiplications schoolbook
        Total: 21 + 12 = 33. Same as before.
        
        Alternative: Try to share some multiplications between the Strassen parts.
        Or try a completely different approach.
        
        Let me try: Split p=6 into 3+3 instead of 2+2+2.
        - 2x3 by 3x3: rank 15 (schoolbook) or use some optimization
        - 2x3 by 3x3: rank 15
        Total: 30. Still not 27.
        
        Let me try to find if rank 27 is possible by checking lower bounds.
        """
        pass

    def algebraic_search_for_rank(target_rank, max_iters=5000):
        """
        Use an algebraic approach to search for a decomposition of given rank.
        We fix U and V, then solve for W, or use iterative methods.
        """
        if time.time() > deadline - 5:
            return None
        
        # Try structured search: fix U to have specific patterns
        for trial in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse U, V
            U = np.zeros((dim_a, target_rank), dtype=np.int64)
            V = np.zeros((dim_b, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                # Each column of U has 2-4 non-zero entries
                nz_count = np.random.randint(2, 5)
                rows = np.random.choice(dim_a, size=nz_count, replace=False)
                for r in rows:
                    U[r, c] = np.random.choice([-1, 1])
                
                # Each column of V has 2-4 non-zero entries
                nz_count = np.random.randint(2, 5)
                rows = np.random.choice(dim_b, size=nz_count, replace=False)
                for r in rows:
                    V[r, c] = np.random.choice([-1, 1])
            
            # For each c, we need W[:, c] such that:
            # sum_c U[:,c] V[:,c] W[:,c]^T = T
            # This is a linear system in W if we fix U and V.
            
            # Build the linear system: for each (a,b,c_idx) where T[a,b,c_idx] != 0,
            # we need sum_r U[a,r] V[b,r] W[c_idx,r] = T[a,b,c_idx]
            
            # Collect equations
            equations = []
            unknowns = []  # indices (c_idx, r)
            rhs = []
            
            unknown_idx = {}
            def get_unknown_idx(ci, r):
                key = (ci, r)
                if key not in unknown_idx:
                    unknown_idx[key] = len(unknown_idx)
                return unknown_idx[key]
            
            for a in range(dim_a):
                for b in range(dim_b):
                    for ci in range(dim_c):
                        if T[a, b, ci] != 0:
                            # Equation: sum_r U[a,r] V[b,r] W[ci,r] = T[a,b,ci]
                            eq = np.zeros(target_rank, dtype=np.float64)
                            for r in range(target_rank):
                                coeff = float(U[a, r] * V[b, r])
                                if coeff != 0:
                                    idx = get_unknown_idx(ci, r)
                                    eq[idx] = coeff
                            equations.append(eq)
                            rhs.append(float(T[a, b, ci]))
            
            if len(equations) == 0:
                continue
            
            A_mat = np.array(equations, dtype=np.float64)
            b_vec = np.array(rhs, dtype=np.float64)
            
            # Check if system is consistent and has integer solution
            # Use least squares and check if residual is zero
            try:
                sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    continue
                
                # Check if solution is close to integer
                sol_rounded = np.round(sol)
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    continue
                
                # Verify
                W = np.zeros((dim_c, target_rank), dtype=np.int64)
                for ci in range(dim_c):
                    for r in range(target_rank):
                        W[ci, r] = int(sol_rounded[get_unknown_idx(ci, r)])
                
                if verify_decomposition(U, V, W):
                    return U, V, W
                    
            except:
                continue
        
        return None

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks
    found_improvement = False
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try algebraic search
        result = algebraic_search_for_rank(r, max_iters=2000)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            found_improvement = True
            break
        
        # Also try random search with small integers
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_a, r))
            V = np.random.randint(-1, 2, size=(dim_b, r))
            W = np.random.randint(-1, 2, size=(dim_c, r))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break
        
        if found_improvement:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
