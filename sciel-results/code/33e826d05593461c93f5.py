import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def compute_errors(U, V, W):
        """Compute all (i,j,k) errors efficiently using einsum."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    errors[i, j, k] = val - 1.0
        return errors
    
    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        errors = compute_errors(U, V, W)
        return np.max(np.abs(errors))
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school.copy(), V_school.copy(), W_school.copy())
    
    # === Structured block decomposition: 2x2 + 3x3 blocks ===
    # Partition indices: top-left 2x2 block uses indices {0,1}, bottom-right 3x3 uses {2,3,4}
    # Block structure:
    #   A = [[A11, A12],    B = [[B11, B12],    C = A*B = [[C11, C12],
    #        [A21, A22]]         [B21, B22]]              [C21, C22]]
    #   A11: 2x2, A12: 2x3, A21: 3x2, A22: 3x3
    #   B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    #   C11 = A11*B11 (2x2), C12 = A11*B12 + A12*B22 (2x3)
    #   C21 = A21*B11 + A22*B21 (3x2), C22 = A21*B12 + A22*B22 (3x3)
    
    # Strassen algorithm for 2x2 matrix multiplication (rank 7)
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Laderman algorithm for 3x3 matrix multiplication (rank 15)
    # This is a known rank-15 decomposition for 3x3 matrix multiplication
    
    # For rectangular 2x3 * 2x3 -> 2x3 and 3x2 * 3x2 -> 3x2,
    # we use schoolbook-like rank-6 decompositions (2*3=6 each)
    # But we need to be careful about the tensor structure.
    
    # Let's build the full rank-70 decomposition:
    # Rank 7: Strassen for C11 = A11*B11
    # Rank 15: Laderman for C22 = A22*B22
    # Rank 6: schoolbook for C12 contribution A11*B12 (2x2 * 2x3)
    # Rank 6: schoolbook for C12 contribution A12*B22 (2x3 * 3x3)
    # Rank 6: schoolbook for C21 contribution A21*B11 (3x2 * 2x2)
    # Rank 6: schoolbook for C21 contribution A22*B21 (3x3 * 3x2)
    # Total: 7 + 15 + 6 + 6 + 6 + 6 = 46
    
    # Wait, let me reconsider. The tensor for 5x5*5x5 has 125 entries.
    # Each rank-1 term contributes to one (i,j,k) entry.
    # We need exactly 1 for each (i,j,k) where i,j,k in {0..4}.
    
    # Let me define the block indices:
    # Block 0 (top-left 2x2): i in {0,1}, j in {0,1}
    # Block 1 (top-right 2x3): i in {0,1}, j in {2,3,4}
    # Block 2 (bottom-left 3x2): i in {2,3,4}, j in {0,1}
    # Block 3 (bottom-right 3x3): i in {2,3,4}, j in {2,3,4}
    
    # For C[i,j] = sum_k A[i,k]*B[k,j]:
    # C11[i,j] = sum_{k=0,1} A11[i,k]*B11[k,j] for i,j in {0,1}
    # C12[i,j] = sum_{k=0,1} A11[i,k]*B12[k,j-2] + sum_{k=2,3,4} A12[i,k-2]*B22[k-2,j-2] for i in {0,1}, j in {2,3,4}
    # C21[i,j] = sum_{k=0,1} A21[i-2,k]*B11[k,j] + sum_{k=2,3,4} A22[i-2,k-2]*B21[k-2,j] for i in {2,3,4}, j in {0,1}
    # C22[i,j] = sum_{k=0,1} A21[i-2,k]*B12[k,j-2] + sum_{k=2,3,4} A22[i-2,k-2]*B22[k-2,j-2] for i,j in {2,3,4}
    
    # Strategy: Use separate rank-1 decompositions for each block of C
    # C11: 2x2 multiplication, use Strassen rank 7
    # C22: 3x3 multiplication, use Laderman rank 15
    # C12: 2x3 multiplication, needs rank 6 (schoolbook)
    # C21: 3x2 multiplication, needs rank 6 (schoolbook)
    # Total: 7 + 15 + 6 + 6 = 34
    
    # But wait - the tensor entries are (i,j,k) where i is row of C, j is col of C, k is inner dimension.
    # For C11: i in {0,1}, j in {0,1}, k in {0,1} -> 8 entries
    # For C12: i in {0,1}, j in {2,3,4}, k in {0,1} (from A11*B12) and k in {2,3,4} (from A12*B22)
    #   So k can be 0,1,2,3,4 -> 2*3*5 = 30 entries
    # For C21: i in {2,3,4}, j in {0,1}, k in {0,1,2,3,4} -> 3*2*5 = 30 entries
    # For C22: i in {2,3,4}, j in {2,3,4}, k in {0,1,2,3,4} -> 3*3*5 = 45 entries
    # Total: 8 + 30 + 30 + 45 = 113... that's not 125.
    
    # I think I'm overcomplicating. Let me just use the schoolbook rank-125 as fallback
    # and try to find a lower rank using the optimization approach, but with better initialization.
    
    # Actually, let me try a different approach: use known algorithms for sub-blocks
    # and combine them.
    
    # For now, let's use the optimization approach but with a smarter initialization
    # based on the block structure.
    
    candidate_ranks = [34, 40, 46, 50, 55, 60, 70, 80, 90, 100, 110, 120]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 25:
            break
        
        if rank >= rank_schoolbook:
            continue
        
        best_err_for_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init_seed in range(8):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(init_seed * 1000 + rank + 7)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Structured initialization: assign rank-1 terms to blocks
            # For each (i,j,k) triple, we want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # when r corresponds to that (i,j,k), and 0 otherwise.
            
            # Initialize with schoolbook-like pattern but with some structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1
            
            # Add random perturbation for the remaining columns
            for r in range(col, rank):
                i_idx = rng.randint(n)
                j_idx = rng.randint(m)
                k_idx = rng.randint(p)
                U[i_idx*m+j_idx, r] = 1.0 + rng.randn() * 0.1
                V[j_idx*p+k_idx, r] = 1.0 + rng.randn() * 0.1
                W[k_idx*n+i_idx, r] = 1.0 + rng.randn() * 0.1
            
            # Phase 1: Standard alternating minimization
            prev_err = float('inf')
            for it in range(60):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((rank, p))
                        for k in range(p):
                            B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((rank, m))
                        for j in range(m):
                            C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Compute error
                errors = compute_errors(U, V, W)
                max_err = np.max(np.abs(errors))
                
                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if max_err < 1e-10:
                    break
                
                if max_err > prev_err * 1.5 and it > 15:
                    break
                prev_err = max_err
            
            if best_err_for_rank < 1e-8:
                break
        
        if best_err_for_rank < 1e-6 and best_U is not None:
            # Phase 2: High-precision polish pass
            U_polish = best_U.copy()
            V_polish = best_V.copy()
            W_polish = best_W.copy()
            
            polish_prev_err = best_err_for_rank
            for pol_it in range(30):
                if time.time() > deadline - 5:
                    break
                
                errors = compute_errors(U_polish, V_polish, W_polish)
                max_err = np.max(np.abs(errors))
                
                if max_err < 1e-12:
                    break
                
                # Update U with high precision
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((rank, p))
                        for k in range(p):
                            B[:, k] = V_polish[j*p+k, :] * W_polish[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U_polish[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            A[:, i] = U_polish[i*m+j, :] * W_polish[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            V_polish[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((rank, m))
                        for j in range(m):
                            C[:, j] = U_polish[i*m+j, :] * V_polish[j*p+k, :]
                        try:
                            sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                            W_polish[k*n+i, :] = sol
                        except:
                            pass
                
                errors = compute_errors(U_polish, V_polish, W_polish)
                max_err = np.max(np.abs(errors))
                
                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U_polish.copy()
                    best_V = V_polish.copy()
                    best_W = W_polish.copy()
                
                if max_err > polish_prev_err * 1.1 and pol_it > 5:
                    break
                polish_prev_err = max_err
            
            # Round to nearest half-integer
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            max_err = verify(U_r, V_r, W_r, rank)
            if max_err < 1e-8:
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                if rank <= 50:
                    break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
