import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][r] = 1
                V_school[j * p + k][r] = 1
                W_school[k * n + i][r] = 1
                r += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True
    
    def verify_numpy(U, V, W, rank):
        """Verify using numpy for speed."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a rank-25 decomposition using structured approach
    # Strategy: use a block structure based on the dimensions
    
    # Approach: Try to use a decomposition where we group the j-index
    # For each (i,k), we need sum_j A[i,j]*B[j,k] = 1
    # This suggests we need at least 5 terms per (i,k) pair in the worst case
    
    # Let's try a systematic search for rank 25-30
    
    np.random.seed(42)
    
    for target_rank in range(25, 35):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random rational matrices with small numerators/denominators
            # Use a mix of 0, 1, -1, 1/2, -1/2
            values = [0, 1, -1, Fraction(1, 2), Fraction(-1, 2)]
            
            U_rand = [[Fraction(np.random.choice(values)) for _ in range(target_rank)] for _ in range(n * m)]
            W_rand = [[Fraction(np.random.choice(values)) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Solve for V: for each (i,k,j), sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
            # This is a system of n*m*p equations in m*p*target_rank unknowns
            
            num_eqs = n * m * p
            num_unknowns = m * p * target_rank
            
            # Build the system
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(target_rank):
                            col_idx = (j * p + k) * target_rank + r
                            A_mat[eq_idx, col_idx] = float(U_rand[i * m + j][r] * W_rand[k * n + i][r])
                        eq_idx += 1
            
            # Check if the system is consistent
            try:
                rank_A = np.linalg.matrix_rank(A_mat)
                # Only proceed if the system is likely solvable
                if rank_A < num_unknowns:
                    V_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    
                    # Check if the solution is close to rational with small denominator
                    V_rand = V_flat.reshape(m * p, target_rank)
                    
                    # Try to snap to fractions with denominator 2
                    V_snap = np.rint(V_rand * 2) / 2.0
                    
                    # Verify
                    if verify_numpy(U_rand, V_snap.tolist(), W_rand, target_rank):
                        best_rank = target_rank
                        best_result = {
                            "U": U_rand,
                            "V": V_snap.tolist(),
                            "W": W_rand
                        }
                        break
                    else:
                        # Try snapping to integers
                        V_snap_int = np.rint(V_rand)
                        if verify_numpy(U_rand, V_snap_int.tolist(), W_rand, target_rank):
                            best_rank = target_rank
                            best_result = {
                                "U": U_rand,
                                "V": V_snap_int.tolist(),
                                "W": W_rand
                            }
                            break
            except:
                continue
        
        if best_rank == target_rank:
            break
    
    return best_result
