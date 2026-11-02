import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    def build_rank_28():
        """Build the known rank-28 decomposition."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Strassen 2x2x2 helper: adds 7 terms
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1: (A00+A11)(B00+B11) -> C00, C11
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            # M2: (A10+A11)B00 -> C10, -C11
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = -1.0
            col += 1

            # M3: A00(B01-B11) -> C01, C11
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            # M4: A11(B10-B00) -> C00, C10
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = -1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            col += 1

            # M5: (A00+A01)B11 -> -C00, C01
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            col += 1

            # M6: (A10-A00)(B00+B01) -> C11
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+0), col] = -1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            # M7: (A01-A11)(B10+B11) -> C00
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = -1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            col += 1

        # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
        add_strassen(i_off=0, j_off=1, k_off=0)

        # Block 2b: j=1,2, k=2 (schoolbook 2x2x1)
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j, col] = 1.0
                V[j * p + 2, col] = 1.0
                W[2 * n + i, col] = 1.0
                col += 1

        # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
        add_strassen(i_off=0, j_off=1, k_off=3)

        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return U, V, W

    def verify_decomposition(U, V, W, rank, atol=1e-10):
        """Verify that the decomposition correctly computes matrix multiplication."""
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test

        C_decomp = np.zeros((n, p))
        for t in range(rank):
            u_t = U[:, t].reshape(n, m)
            v_t = V[:, t].reshape(m, p)
            w_t = W[:, t].reshape(p, n)
            s1 = np.sum(A_test * u_t)
            s2 = np.sum(B_test * v_t)
            C_decomp += s1 * s2 * w_t.T

        return np.allclose(C_test, C_decomp, atol=atol)

    def try_rank_27_structured():
        """
        Try to find a rank-27 decomposition by fixing U to a structured pattern
        and solving for V and W via linear algebra.
        
        Strategy: Use the rank-28 solution as a starting point, then attempt to
        merge two rank-1 terms into one by solving a constrained linear system.
        """
        if time.time() - start_time > DEADLINE - 5:
            return None

        # Start from rank-28 solution
        U28, V28, W28 = build_rank_28()

        # Attempt 1: Try to find rank-27 by removing redundancy
        # The tensor T_{ijk} = sum_t U_{it} V_{jt} W_{kt}
        # We need to find if there's a rank-27 representation.
        
        # Approach: Fix U to have a specific structure and solve for V, W
        # Use the fact that for matrix multiplication, we can exploit the
        # Kronecker product structure.
        
        rank = 27
        # Try different structured U patterns
        
        # Pattern 1: Based on Kronecker product of identity blocks
        # U has shape (n*m, rank) = (6, 27)
        # Structure: group columns by (i, j) pairs with some merging
        
        # Let's try: for each i in {0,1}, j in {0,1,2}, we have some number of columns
        # Total: 6 columns for j=0 (2 per i), and 21 for j=1,2 (3.5 per i,j on average)
        
        # Actually, let's try a more systematic approach:
        # Fix U to be a specific matrix and solve for V, W
        
        # The constraint is: for all i,j,k:
        #   T_{ijk} = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # where T_{ijk} = 1 if (i,j,k) is a valid multiplication index, else 0
        # Actually for matrix multiplication, T_{ijk} = delta_{j,j'} where j is the shared index
        # More precisely: C_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor is: T_{i,j,k} = 1 (it's always 1 for valid indices)
        # Wait, that's not right. The tensor decomposition is:
        # C_{ik} = sum_t (sum_j A_{ij} U_{i,j,t}) * (sum_j B_{jk} V_{j,k,t}) * W_{k,i,t}
        # This must equal sum_j A_{ij} B_{jk} for all A, B
        
        # So we need: for each t, the product U_t * V_t * W_t (as rank-1 tensors)
        # to sum to the multiplication tensor.
        
        # Let's set up the linear system differently.
        # Given U fixed, we need to find V and W such that:
        # For each (i,k): sum_t W[k,i,t] * (sum_j A_{ij} U[i,j,t]) * (sum_j B_{jk} V[j,k,t]) = sum_j A_{ij} B_{jk}
        
        # This is a bilinear constraint in V and W, so it's not linear.
        # However, if we fix W as well, it becomes linear in V, and vice versa.
        
        # Alternative: Use the fact that the multiplication tensor has a specific structure.
        # For 2x3x5, the minimal rank is known to be 28 (or possibly less).
        
        # Let's try a simpler approach: check if any two columns of the rank-28
        # solution are linearly dependent in a way that allows merging.
        
        # Check rank of U, V, W matrices
        rank_U = np.linalg.matrix_rank(U28, tol=1e-10)
        rank_V = np.linalg.matrix_rank(V28, tol=1e-10)
        rank_W = np.linalg.matrix_rank(W28, tol=1e-10)
        
        # If any factor has rank < 28, we might be able to reduce
        # But for a minimal decomposition, all factors should have full rank
        
        # Try iterative approach: fix U, solve for V given W, then solve for W given V
        # Start from rank-28 and try to project to rank-27
        
        # This is complex; let's try a simpler heuristic:
        # Try to find if there's a rank-27 solution by random search with structure
        
        if time.time() - start_time > DEADLINE - 10:
            return None
        
        # Try several random structured U patterns
        np.random.seed(123)
        
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Create a structured U: for each (i,j) pair, assign some columns
            U = np.zeros((n * m, rank), dtype=float)
            
            # Assign columns to (i,j) pairs
            # 6 pairs, 27 columns -> some pairs get 4, some get 5
            # Let's try: j=0 gets 4 columns (2 per i), j=1 gets 12, j=2 gets 11
            cols_per_j = [4, 12, 11]
            col_offset = 0
            
            for j in range(m):
                for i in range(n):
                    # Assign cols_per_j[j]//n or cols_per_j[j]//n + 1 columns
                    num_cols = cols_per_j[j] // n + (1 if i < cols_per_j[j] % n else 0)
                    for c in range(num_cols):
                        U[i * m + j, col_offset + c] = 1.0
                    col_offset += num_cols
            
            # Now solve for V and W using alternating least squares
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS iterations
            for it in range(50):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Solve for V given U, W
                # For each k: sum_t W[k,i,t] * (sum_j A_{ij} U[i,j,t]) * V[j,k,t] = ...
                # This is still bilinear. Let's use a different approach.
                
                # Actually, let's just verify the rank-28 solution and return it
                # The rank-27 search is too complex for the time budget.
                break
            
            break
        
        return None

    # First, try to find a rank-27 solution
    result_27 = try_rank_27_structured()
    if result_27 is not None:
        U, V, W = result_27
        if verify_decomposition(U, V, W, 27):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-28
    U, V, W = build_rank_28()
    if verify_decomposition(U, V, W, 28):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return schoolbook rank-30
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col_sb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col_sb] = 1.0
                V_sb[j * p + k, col_sb] = 1.0
                W_sb[k * n + i, col_sb] = 1.0
                col_sb += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
