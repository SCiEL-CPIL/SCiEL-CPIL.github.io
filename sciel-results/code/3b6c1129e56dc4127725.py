import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
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
    
    def try_kronecker_structure(rank):
        """
        Try to construct a decomposition using Kronecker product structure.
        View 6 = 2*3, so output columns k = (k0, k1) where k0 in {0,1}, k1 in {0,1,2}
        This gives a (2,3,2,3) structure.
        
        Ansatz: Use rank = 2 * 3 * 2 = 12 with structure:
        For each (k0, k1) where k0 in {0,1} (pair index) and k1 in {0,1,2} (within pair):
          Actually k = k0*3 + k1, so pairs are (0,1,2) and (3,4,5)
          
        Better: view k as (k_a, k_b) with k_a in {0,1}, k_b in {0,1,2}, k = k_a*3 + k_b
        Then we have a 2x3x2x3 tensor structure.
        
        Try rank 12: U[i,j,c], V[j,k,c], W[k,i,c] with c indexing (c_a, c_b)
        """
        if time.time() > deadline - 10:
            return None
        
        # Try rank 12 with structured Kronecker approach
        # k = k_a * 3 + k_b, where k_a in {0,1}, k_b in {0,1,2}
        # Output index: (i, k_a, k_b)
        # Input A: (i, j), Input B: (j, k_a, k_b)
        
        # For each (k_a, k_b), we need:
        # sum_j A[i,j] * B[j, k_a, k_b] = output[i, k_a, k_b]
        
        # This is like 2 independent 2x3 x 3x3 problems (for each k_a)
        # Each 2x3 x 3x3 has schoolbook rank 18, total 36
        # But we can share structure across k_a
        
        # Try: for each k_b (3 values), solve a 2x3 x 3x2 problem
        # 2x3 x 3x2: schoolbook rank 12, total 3*12 = 36
        
        # Let me try a different grouping: k = k_a + 3*k_b where k_a in {0,1,2}, k_b in {0,1}
        # So pairs are (0,3), (1,4), (2,5)
        
        # For each pair (k_a, k_a+3), we have a 2x3 x 3x2 subproblem
        # 3 such subproblems, each rank 12 schoolbook = 36 total
        
        # To beat schoolbook, we need to share rank terms across subproblems.
        
        # Try rank 24: use 8 terms per subproblem instead of 12
        # This requires solving 2x3 x 3x2 with rank 8
        
        # Let's try to find rank 8 decomposition for 2x3 x 3x2
        # n2=2, m2=3, p2=2, schoolbook=12
        # Try rank 8, 9, 10, 11
        
        for sub_rank in [8, 9, 10, 11, 12]:
            if time.time() > deadline - 10:
                break
            
            # Try to find rank sub_rank decomposition for 2x3 x 3x2
            # Then use 3 copies for the 3 subproblems, total rank = 3 * sub_rank
            total_rank = 3 * sub_rank
            
            if total_rank >= schoolbook_rank:
                continue
            
            result = _find_2x3x3x2_decomposition(sub_rank, deadline)
            if result is not None:
                U_sub, V_sub, W_sub = result
                # Build full decomposition
                U_full = [[0] * total_rank for _ in range(n * m)]
                V_full = [[0] * total_rank for _ in range(m * p)]
                W_full = [[0] * total_rank for _ in range(p * n)]
                
                for sub_idx in range(3):
                    offset = sub_idx * sub_rank
                    # Subproblem: A is 2x3, B_sub is 3x2 (columns k=2*sub_idx and k=2*sub_idx+1)
                    # Wait, let me reconsider the grouping
                    # k = 0,1,2,3,4,5
                    # Group 1: k=0,1 (sub_idx=0)
                    # Group 2: k=2,3 (sub_idx=1)  
                    # Group 3: k=4,5 (sub_idx=2)
                    
                    for i in range(n):
                        for j in range(m):
                            for c in range(sub_rank):
                                U_full[i*m+j][offset+c] = U_sub[i*m+j][c]
                    
                    for j in range(m):
                        for kb in range(2):
                            k = sub_idx * 2 + kb
                            for c in range(sub_rank):
                                V_full[j*p+k][offset+c] = V_sub[j*2+kb][c]
                    
                    for kb in range(2):
                        k = sub_idx * 2 + kb
                        for i in range(n):
                            for c in range(sub_rank):
                                W_full[k*n+i][offset+c] = W_sub[kb*n+i][c]
                
                if verify_decomposition(U_full, V_full, W_full, total_rank):
                    return U_full, V_full, W_full, total_rank
        
        return None
    
    def _find_2x3x3x2_decomposition(rank, deadline):
        """Find rank decomposition for 2x3 x 3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        schoolbook = n2 * m2 * p2  # 12
        
        def verify_sub(U, V, W, r):
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m2+j][c] * V[j*p2+k][c] * W[k*n2+i][c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        # Try ALS approach for the subproblem
        for trial in range(5):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(1000 + trial * 11)
            
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p2*n2, rank))
            for i in range(p2*n2):
                if all(W[i] == 0):
                    W[i][trial % rank] = 1.0
            
            U = np.random.randn(n2*m2, rank) * 0.1
            V = np.random.randn(m2*p2, rank) * 0.1
            
            for iteration in range(150):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for V
                A_v = np.zeros((n2*m2*p2, m2*p2*rank))
                b_v = np.ones(n2*m2*p2)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for c in range(rank):
                                w_val = W[k*n2+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p2 + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m2+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m2*p2, rank)
                
                # Solve for U
                A_u = np.zeros((n2*m2*p2, n2*m2*rank))
                b_u = np.ones(n2*m2*p2)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for c in range(rank):
                                w_val = W[k*n2+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m2 + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p2+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n2*m2, rank)
                
                if iteration % 5 == 0 or iteration == 149:
                    max_err = 0.0
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m2+j][c] * V[j*p2+k][c] * W[k*n2+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_sub(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_sub(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    def try_direct_als(rank, max_iters=200, trials=3):
        """Direct ALS for the full 2x3x3x6 problem."""
        for trial in range(trials):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(3000 + trial * 7)
            
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, rank))
            for i in range(p*n):
                if all(W[i] == 0):
                    W[i][trial % rank] = 1.0
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            
            prev_error = float('inf')
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for V
                A_v = np.zeros((n*m*p, m*p*rank))
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
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m*p, rank)
                
                # Solve for U
                A_u = np.zeros((n*m*p, n*m*rank))
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
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n*m, rank)
                
                if iteration % 5 == 0 or iteration == max_iters - 1:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    
                    if max_err > prev_error * 1.01 and iteration > 50:
                        break
                    prev_error = max_err
                    
        return None
    
    # First try the Kronecker structure approach
    kronecker_result = try_kronecker_structure(24)
    if kronecker_result is not None:
        U, V, W, rank = kronecker_result
        return {"U": U, "V": V, "W": W}
    
    # Try direct ALS for various ranks
    for rank in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        res = try_direct_als(rank, max_iters=150, trials=2)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
