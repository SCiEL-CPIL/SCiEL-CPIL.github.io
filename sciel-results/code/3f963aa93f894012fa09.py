import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = 3, 3, 6
    
    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = generate_schoolbook()
    best_rank = n * m * p
    
    def construct_3x3x2_explicit():
        """Construct a 3x3x2 decomposition using Strassen for 2x2x2 core + boundary terms."""
        R = 15
        U = np.zeros((9, R), dtype=int)
        V = np.zeros((6, R), dtype=int)
        W = np.zeros((6, R), dtype=int)
        
        # Strassen for 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k] for i,j in {0,1}, k in {0,1}
        # U idx: i*3+j, V idx: j*2+k, W idx: k*3+i
        # Standard Strassen:
        # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=(A00-A10)(B11+B11)
        # Let me use verified Strassen:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00  
        # M3=(A00-A10)(B11+B11) -- wrong
        # 
        # Correct Strassen:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=(A00-A10)(B11+B11) -- still wrong
        #
        # Let me use: C00=M1-M2+M3-M4, C01=M2+M4, C10=M1-M3+M5, C11=M1-M2+M4+M5
        # where M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=(A00-A10)(B11+B11)
        #
        # I'll just use a known correct set:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=(A00-A10)(B11+B11) -- I keep making the same error
        #
        # OK definitive Strassen:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=(A00-A10)(B11+B11)
        # M4=(A01-A10)(B00+B01)
        # M5=(A01+A11)B11
        # M6=(A00+A01)(B10)
        # M7=(A10-A11)(B01)
        # C00=M1-M4+M6-M7
        # C01=M4+M5
        # C10=M2-M6+M7
        # C11=M1-M2+M3+M5
        #
        # Let me verify C00: M1-M4+M6-M7
        # = (A00+A11)(B00+B11) - (A01-A10)(B00+B01) + (A00+A01)B10 - (A10-A11)B01
        # = A00B00+A00B11+A11B00+A11B11 - A01B00-A01B01+A10B00+A10B01 + A00B10+A01B10 - A10B01+A11B01
        # = A00B00+A00B11+A11B00+A11B11 - A01B00-A01B01+A10B00+A10B01 + A00B10+A01B10 - A10B01+A11B01
        # C00 should be A00B00+A01B10
        # Extra terms: A00B11+A11B00+A11B11-A01B00-A01B01+A10B00+A10B01+A00B10+A01B10-A10B01+A11B01
        # This doesn't simplify to just A00B00+A01B10. So this is wrong.
        #
        # Let me use the actual standard Strassen formulas:
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = (a11 - a21)(b22 + b22) -- I'll stop trying to recall and just verify numerically
        pass
        
        # I'll use a simple verified Strassen by checking all 4 outputs
        # Let me define Strassen terms and verify:
        # Term 0: U: A00,A11; V: B00,B11; W: C00,C01,C10,C11 (coefficients to work out)
        # This is getting too complex to do by hand. Let me use a different approach.
        
        # Direct construction: use rank-1 terms for all 18 products, then try to merge
        # Actually, let me just return None and use the random search with better initialization
        return None
    
    def find_3x3x2_decomposition(R, max_attempts=200):
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Structured initialization: start with schoolbook-like and perturb
            U_r = np.zeros((9, R), dtype=int)
            V_r = np.zeros((6, R), dtype=int)
            W_r = np.zeros((6, R), dtype=int)
            
            # Initialize with a partial structure
            if R >= 18:
                for i in range(3):
                    for j in range(3):
                        for k in range(2):
                            col = i * 6 + j * 2 + k
                            if col < R:
                                U_r[i * 3 + j, col] = 1
                                V_r[j * 2 + k, col] = 1
                                W_r[k * 3 + i, col] = 1
                # Now try to merge some terms
                for _ in range(50):
                    if time.time() > deadline - 10:
                        return None
                    # Randomly zero out some entries and see if we can adjust
                    r1 = np.random.randint(0, R)
                    r2 = np.random.randint(0, R)
                    if r1 == r2:
                        continue
                    # Try to merge columns r1 and r2
                    U_r[:, r2] = U_r[:, r1] + U_r[:, r2]
                    V_r[:, r2] = V_r[:, r1] + V_r[:, r2]
                    W_r[:, r2] = W_r[:, r1] + W_r[:, r2]
                    U_r[:, r1] = 0
                    V_r[:, r1] = 0
                    W_r[:, r1] = 0
                    # Check if still valid
                    valid = True
                    for i in range(3):
                        for j in range(3):
                            for k in range(2):
                                for ip in range(3):
                                    for jp in range(3):
                                        for kp in range(2):
                                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                            actual = 0.0
                                            for r in range(R):
                                                actual += U_r[i*3+j, r] * V_r[jp*2+kp, r] * W_r[k*3+i, r]
                                            if abs(actual - expected) > 1e-9:
                                                valid = False
                                                break
                                        if not valid: break
                                    if not valid: break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid:
                        # Revert
                        U_r[:, r1] = U_r[:, r2]
                        V_r[:, r1] = V_r[:, r2]
                        W_r[:, r1] = W_r[:, r2]
                        U_r[:, r2] = 0
                        V_r[:, r2] = 0
                        W_r[:, r2] = 0
            else:
                # Random init for lower ranks
                U_r = np.random.randint(-2, 3, size=(9, R))
                V_r = np.random.randint(-2, 3, size=(6, R))
                W_r = np.zeros((6, R))
                
                valid = True
                for i in range(3):
                    for k in range(2):
                        M = np.zeros((9, R))
                        for j in range(3):
                            for jp in range(3):
                                M[j * 3 + jp, :] = U_r[i * 3 + j, :] * V_r[jp * 2 + k, :]
                        
                        rhs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
                        
                        try:
                            w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                        except:
                            valid = False
                            break
                        
                        w_doubled = w_vec * 2
                        closest_ints = np.round(w_doubled)
                        error = np.max(np.abs(w_doubled - closest_ints))
                        
                        if error < 1e-6:
                            W_r[k * 3 + i, :] = w_vec
                        else:
                            valid = False
                            break
                
                if not valid:
                    continue
                W_r = np.round(W_r * 2) / 2  # Snap to half-integers
            
            # Final verification
            correct = True
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        for ip in range(3):
                            for jp in range(3):
                                for kp in range(2):
                                    expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    actual = 0.0
                                    for r in range(R):
                                        actual += U_r[i*3+j, r] * V_r[jp*2+kp, r] * W_r[k*3+i, r]
                                    if abs(actual - expected) > 1e-9:
                                        correct = False
                                        break
                                if not correct: break
                            if not correct: break
                        if not correct: break
                    if not correct: break
                if not correct: break
            if correct:
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    for R in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 20:
            break
        decomp = find_3x3x2_decomposition(R)
        if decomp is not None:
            U_core, V_core, W_core = decomp
            total_rank = 3 * R
            
            if total_rank < best_rank:
                U_tot = [[0] * total_rank for _ in range(n * m)]
                V_tot = [[0] * total_rank for _ in range(m * p)]
                W_tot = [[0] * total_rank for _ in range(p * n)]
                
                for block in range(3):
                    offset = block * R
                    for i in range(3):
                        for j in range(3):
                            for r in range(R):
                                U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
                    
                    for j in range(3):
                        for k in range(2):
                            for r in range(R):
                                V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
                    
                    for i in range(3):
                        for k in range(2):
                            for r in range(R):
                                W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
                
                # Verify full decomposition
                correct = True
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            for ip in range(3):
                                for jp in range(3):
                                    for kp in range(6):
                                        expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                        actual = 0.0
                                        for r in range(total_rank):
                                            actual += U_tot[i*3+j][r] * V_tot[jp*6+kp][r] * W_tot[k*3+i][r]
                                        if abs(actual - expected) > 1e-9:
                                            correct = False
                                            break
                                    if not correct: break
                                if not correct: break
                            if not correct: break
                        if not correct: break
                    if not correct: break
                if correct:
                    best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                    best_rank = total_rank
                    break
    
    return best_result
