"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank
    
    def try_rank_with_fixed_W(W_pattern, rank):
        """
        Given a fixed W pattern (list of lists of shape (p*n, rank)),
        solve for U and V such that the decomposition holds.
        For each (i,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j.
        If W[k*n+i,c] is nonzero, we can write:
          sum_j U[i*m+j,c] * V[j*p+k,c] = 1/W[k*n+i,c] for each c where W is nonzero.
        This is still bilinear. Instead, use iterative approach:
        Alternate between fixing U and solving for V (linear), and vice versa.
        """
        W = [row[:] for row in W_pattern]
        
        # Initialize U and V randomly
        np.random.seed(12345)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            # Fix U, solve for V: For each (j,k,c) where W[k*n+i,c]!=0 for some i
            # Actually, the equations are: for each (i,j,k):
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed U and W, this is linear in V.
            # But the same V[j*p+k,c] appears in equations for different i.
            # Let's set up the linear system for V.
            
            # Unknowns: V[j*p+k, c] for j in range(m), k in range(p), c in range(rank)
            # Equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Build linear system A_v @ v_vec = b_v
            # v_vec is flattened V
            num_v_unknowns = m * p * rank
            A_v = np.zeros((n*m*p, num_v_unknowns))
            b_v = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            w_val = W[k*n+i][c]
                            if abs(w_val) > 1e-12:
                                v_idx = (j*p + k) * rank + c
                                A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                        eq_idx += 1
            
            # Solve least squares
            try:
                V_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
            except:
                return None
            V = V_sol.reshape(m*p, rank)
            
            # Fix V, solve for U: similar
            num_u_unknowns = n * m * rank
            A_u = np.zeros((n*m*p, num_u_unknowns))
            b_u = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            w_val = W[k*n+i][c]
                            if abs(w_val) > 1e-12:
                                u_idx = (i*m + j) * rank + c
                                A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                        eq_idx += 1
            
            try:
                U_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
            except:
                return None
            U = U_sol.reshape(n*m, rank)
            
            # Check convergence
            if iteration % 50 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-10:
                    # Snap to nearest half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    # W is already fixed (integer/half-integer by construction)
                    if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W, rank):
                        return U_snap.tolist(), V_snap.tolist(), W
        
        return None
    
    # Try structured W patterns
    # Pattern 1: Schoolbook W (each column has one 1)
    # Pattern 2: W with 2x2 blocks
    
    # First, try to find a good decomposition using alternating least squares
    # with various W initializations
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try different W patterns
    W_patterns = []
    
    # Pattern 0: Schoolbook
    W0 = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for k in range(p):
            W0[k*n+i][col] = 1
            col += 1
    W_patterns.append(('schoolbook', W0, schoolbook_rank))
    
    # Try rank 24 with random W
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        np.random.seed(100 + trial)
        W_rand = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, 24))
        # Ensure each (i,k) has at least one nonzero in W
        for i in range(n):
            for k in range(p):
                if all(W_rand[k*n+i] == 0):
                    W_rand[k*n+i][trial % 24] = 1.0
        W_patterns.append((f'rand_{trial}', W_rand.tolist(), 24))
    
    # Try rank 20
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        np.random.seed(200 + trial)
        W_rand = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, 20))
        for i in range(n):
            for k in range(p):
                if all(W_rand[k*n+i] == 0):
                    W_rand[k*n+i][trial % 20] = 1.0
        W_patterns.append((f'rand20_{trial}', W_rand.tolist(), 20))
    
    # Try rank 18
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        np.random.seed(300 + trial)
        W_rand = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, 18))
        for i in range(n):
            for k in range(p):
                if all(W_rand[k*n+i] == 0):
                    W_rand[k*n+i][trial % 18] = 1.0
        W_patterns.append((f'rand18_{trial}', W_rand.tolist(), 18))
    
    # Try each pattern
    for name, W_pat, rank in sorted(W_patterns, key=lambda x: x[2]):
        if time.time() > deadline - 5:
            break
        result = try_rank_with_fixed_W(W_pat, rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
