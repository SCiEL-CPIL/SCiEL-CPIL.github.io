import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook

    # Strategy: Structured Block Strassen Construction
    # We decompose the 4x4 middle dimension (m=4) into 2x2 blocks of 2x2 matrices.
    # Let A be 3x4, B be 4x6.
    # Split A into A1 (3x2), A2 (3x2).
    # Split B into B1 (2x6), B2 (2x6).
    # Then C = A1*B1 + A2*B2.
    # This is just standard multiplication of two 3x2 and 2x6 matrices.
    # Rank of 3x2x6 multiplication is at most 3*2*6 = 36.
    # Wait, the tensor is (3x4) * (4x6).
    # If we view the 4 dimension as two 2-dimensions, we have:
    # C = A_left * B_top + A_right * B_bottom
    # Where A_left is 3x2, B_top is 2x6.
    # A_right is 3x2, B_bottom is 2x6.
    # The multiplication of two 3x2 and 2x6 matrices can be done with rank 3*2*6 = 36?
    # No, the rank of the tensor M_{3,2,6} is what matters.
    # Actually, we can use Strassen on the 2x2 block structure if we pad or split further.
    
    # Let's try a different structural split.
    # Split the 4x4 space into four 2x2 blocks? No, m=4 is a single dimension.
    # Let's split the output C (3x6) into blocks?
    
    # Alternative Structural Strategy:
    # Use the fact that rank(3x4x6) <= rank(3x4x4) + rank(3x4x2) ? No.
    
    # Let's use the known bound: rank(3x4x6) <= 46?
    # Actually, let's try to construct a rank 48 decomposition explicitly.
    # We can view the 4x6 matrix as two 4x3 matrices.
    # C = A * B1 + A * B2 where B1, B2 are 4x3.
    # This is 2 * rank(3x4x3).
    # Rank(3x4x3) is at most 3*4*3 = 36.
    # So 2 * 36 = 72. Not better.
    
    # Let's try splitting the input A (3x4) into two 3x2 matrices A1, A2.
    # And B (4x6) into two 2x6 matrices B1, B2.
    # C = A1*B1 + A2*B2.
    # This is the sum of two 3x2x6 tensors.
    # The rank of a 3x2x6 tensor is at most 18 (schoolbook).
    # So total rank <= 36.
    # Wait, is the rank of M_{3,2,6} actually 18?
    # Yes, for rectangular matrices, the schoolbook is often optimal or close to it.
    # But we can do better by combining terms?
    # No, A1*B1 and A2*B2 are independent operations on disjoint parts of A and B.
    # So rank is sum of ranks.
    # Rank(M_{3,2,6}) = 18?
    # If so, total rank = 36.
    # Let's verify if we can achieve rank 36.
    
    # Construction for rank 36:
    # U, V, W will have 36 columns.
    # First 18 columns correspond to A1*B1.
    # Last 18 columns correspond to A2*B2.
    
    U_36 = [[0] * 36 for _ in range(n * m)]
    V_36 = [[0] * 36 for _ in range(m * p)]
    W_36 = [[0] * 36 for _ in range(p * n)]
    
    # Part 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B)
    # A1 is 3x2. B1 is 2x6.
    # For each i in 0..2, j1 in 0..1, k in 0..5:
    # Term: A[i, j1] * B[j1, k]
    # This contributes to C[i, k].
    # U[i*m + j1, col] = 1
    # V[j1*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    col = 0
    for i in range(n):
        for j1 in range(2):
            for k in range(p):
                # j1 is the column index in A (0,1)
                # j1 is the row index in B (0,1)
                U_36[i * m + j1][col] = 1
                V_36[j1 * p + k][col] = 1
                W_36[k * n + i][col] = 1
                col += 1
                
    # Part 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B)
    # A2 is 3x2. B2 is 2x6.
    # For each i in 0..2, j2 in 0..1, k in 0..5:
    # Term: A[i, j2+2] * B[j2+2, k]
    for i in range(n):
        for j2 in range(2):
            for k in range(p):
                j_idx = j2 + 2
                U_36[i * m + j_idx][col] = 1
                V_36[j_idx * p + k][col] = 1
                W_36[k * n + i][col] = 1
                col += 1
                
    if verify(U_36, V_36, W_36, n, m, p):
        if 36 < best_rank:
            best = {"U": U_36, "V": V_36, "W": W_36}
            best_rank = 36

    # Can we do better than 36?
    # Rank(M_{3,2,6}) might be less than 18?
    # Lower bound for M_{3,2,6}: max(3*2, 2*6, 3*6) = 18? No, LB is higher.
    # Actually, for M_{m,n,p}, LB is max(mn, np, pm) is not correct.
    # LB is max(m, n, p) * something?
    # For 3x2x6, schoolbook is 18.
    # Is it possible to go lower?
    # If we use Strassen on 3x2x6?
    # Pad 3 to 4? Rank(M_{4,2,6})?
    # This is getting complicated.
    
    # Let's try a different split: 3x4x6 -> 3x4x3 + 3x4x3?
    # C = A * [B1 B2] = A*B1 + A*B2.
    # Rank(M_{3,4,3}) is at most 36.
    # Total 72. Worse.
    
    # What if we split 3 into 1+2?
    # C1 (1x6) = A1 (1x4) * B.
    # C2 (2x6) = A2 (2x4) * B.
    # Rank(M_{1,4,6}) = 6? (Each output element is a dot product of length 4, so 4 mults per element, 6 elements = 24? No, rank is number of mults. 1x4 * 4x6. Schoolbook is 1*4*6 = 24.
    # Rank(M_{2,4,6}) = 2*4*6 = 48.
    # Total 72.
    
    # The 3x2x6 split giving 36 seems the best structural simple one.
    # Can we optimize the 3x2x6 part?
    # M_{3,2,6}.
    # Let's try to find a rank < 18 for M_{3,2,6} using ILP on the smaller tensor.
    
    n2, m2, p2 = 3, 2, 6
    target_rank_2x = 17
    found_better_2x = False
    U_2x_best = None
    V_2x_best = None
    W_2x_best = None
    
    random.seed(42)
    # Try to find rank 17 for 3x2x6
    for target_r in range(17, 15, -1):
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random basis for 3x2x6
            U_b = []
            V_b = []
            W_b = []
            for c in range(target_r):
                u = np.array([random.choice([-1, 0, 1]) for _ in range(n2 * m2)])
                v = np.array([random.choice([-1, 0, 1]) for _ in range(m2 * p2)])
                w = np.array([random.choice([-1, 0, 1]) for _ in range(p2 * n2)])
                if np.all(u==0) or np.all(v==0) or np.all(w==0):
                    u[0]=1; v[0]=1; w[0]=1
                U_b.append(u)
                V_b.append(v)
                W_b.append(w)
                
            A_mat = np.zeros((n2*m2*p2, target_r))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        idx = i*m2*p2 + j*p2 + k
                        for c in range(target_r):
                            A_mat[idx, c] = U_b[c][i*m2+j] * V_b[c][j*p2+k] * W_b[c][k*n2+i]
                            
            b_vec = np.ones(n2*m2*p2)
            try:
                res = milp(c=np.zeros(target_r), constraints=[LinearConstraint(A_mat, b_vec, b_vec)], 
                           integrality=np.ones(target_r), bounds=Bounds(lb=-5, ub=5), 
                           options={"time_limit": 2, "gap_abs": 0})
                if res.status == 0:
                    x = np.round(res.x).astype(int)
                    U_s = []
                    V_s = []
                    W_s = []
                    for row in range(n2*m2):
                        U_s.append([x[c]*U_b[c][row] for c in range(target_r)])
                    for row in range(m2*p2):
                        V_s.append([x[c]*V_b[c][row] for c in range(target_r)])
                    for row in range(p2*n2):
                        W_s.append([x[c]*W_b[c][row] for c in range(target_r)])
                        
                    # Verify 3x2x6
                    valid = True
                    U_arr = np.array(U_s)
                    V_arr = np.array(V_s)
                    W_arr = np.array(W_s)
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = 0
                                for c in range(target_r):
                                    s += U_arr[i*m2+j, c] * V_arr[j*p2+k, c] * W_arr[k*n2+i, c]
                                if abs(s-1) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        U_2x_best = U_s
                        V_2x_best = V_s
                        W_2x_best = W_s
                        found_better_2x = True
                        break
            except:
                pass
        if found_better_2x:
            break
            
    if found_better_2x:
        # Combine two copies of the optimized 3x2x6 tensor
        # One for A1*B1, one for A2*B2
        rank_new = len(U_2x_best[0]) * 2
        U_new = [[0] * rank_new for _ in range(n*m)]
        V_new = [[0] * rank_new for _ in range(m*p)]
        W_new = [[0] * rank_new for _ in range(p*n)]
        
        # Fill first half (A1*B1)
        for i in range(n):
            for j1 in range(2):
                for c in range(len(U_2x_best[0])):
                    U_new[i*m + j1][c] = U_2x_best[i*m2 + j1][c]
                    # V and W for B1 (rows 0,1 of B)
                    # V_2x is for m2=2, p2=6.
                    # V_2x[j1*p2 + k][c]
                    # We need to map this to V_new[j*p + k] where j is 0 or 1.
                    # In the 3x2x6 tensor, V index is j*p2 + k.
                    # In the full tensor, V index is j*p + k.
                    # Since p2=6 and p=6, and j is 0,1, the indices match directly for the first block.
                    for k in range(p):
                         V_new[j1*p + k][c] = V_2x_best[j1*p2 + k][c]
                         W_new[k*n + i][c] = W_2x_best[k*n2 + i][c]
                         
        # Fill second half (A2*B2)
        offset = len(U_2x_best[0])
        for i in range(n):
            for j2 in range(2):
                j_idx = j2 + 2
                for c in range(len(U_2x_best[0])):
                    U_new[i*m + j_idx][offset + c] = U_2x_best[i*m2 + j2][c]
                    # V index in 2x tensor is j2*p2 + k.
                    # V index in full tensor is j_idx*p + k.
                    for k in range(p):
                         V_new[j_idx*p + k][offset + c] = V_2x_best[j2*p2 + k][c]
                         W_new[k*n + i][offset + c] = W_2x_best[k*n2 + i][c]
                         
        if verify(U_new, V_new, W_new, n, m, p):
             if rank_new < best_rank:
                 best = {"U": U_new, "V": V_new, "W": W_new}
                 best_rank = rank_new

    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook

    # Strategy: Structured Block Strassen Construction
    # Split m=4 into two 2s.
    # C = A1*B1 + A2*B2
    # Rank <= 2 * Rank(M_{3,2,6})
    
    # First, try to find a good rank for M_{3,2,6}
    n2, m2, p2 = 3, 2, 6
    rank_2x = 18 # Schoolbook
    U_2x_best = None
    V_2x_best = None
    W_2x_best = None
    
    # Initialize with schoolbook for 3x2x6
    U_2x_best = [[0] * rank_2x for _ in range(n2 * m2)]
    V_2x_best = [[0] * rank_2x for _ in range(m2 * p2)]
    W_2x_best = [[0] * rank_2x for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U_2x_best[i * m2 + j][col] = 1
                V_2x_best[j * p2 + k][col] = 1
                W_2x_best[k * n2 + i][col] = 1
                col += 1
                
    # Try to improve M_{3,2,6} rank using ILP
    random.seed(42)
    for target_r in range(17, 15, -1):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            U_b = []
            V_b = []
            W_b = []
            for c in range(target_r):
                u = np.array([random.choice([-1, 0, 1]) for _ in range(n2 * m2)])
                v = np.array([random.choice([-1, 0, 1]) for _ in range(m2 * p2)])
                w = np.array([random.choice([-1, 0, 1]) for _ in range(p2 * n2)])
                if np.all(u==0) or np.all(v==0) or np.all(w==0):
                    u[0]=1; v[0]=1; w[0]=1
                U_b.append(u)
                V_b.append(v)
                W_b.append(w)
                
            A_mat = np.zeros((n2*m2*p2, target_r))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        idx = i*m2*p2 + j*p2 + k
                        for c in range(target_r):
                            A_mat[idx, c] = U_b[c][i*m2+j] * V_b[c][j*p2+k] * W_b[c][k*n2+i]
                            
            b_vec = np.ones(n2*m2*p2)
            try:
                res = milp(c=np.zeros(target_r), constraints=[LinearConstraint(A_mat, b_vec, b_vec)], 
                           integrality=np.ones(target_r), bounds=Bounds(lb=-5, ub=5), 
                           options={"time_limit": 2, "gap_abs": 0})
                if res.status == 0:
                    x = np.round(res.x).astype(int)
                    U_s = []
                    V_s = []
                    W_s = []
                    for row in range(n2*m2):
                        U_s.append([x[c]*U_b[c][row] for c in range(target_r)])
                    for row in range(m2*p2):
                        V_s.append([x[c]*V_b[c][row] for c in range(target_r)])
                    for row in range(p2*n2):
                        W_s.append([x[c]*W_b[c][row] for c in range(target_r)])
                        
                    # Verify 3x2x6
                    valid = True
                    U_arr = np.array(U_s)
                    V_arr = np.array(V_s)
                    W_arr = np.array(W_s)
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = 0
                                for c in range(target_r):
                                    s += U_arr[i*m2+j, c] * V_arr[j*p2+k, c] * W_arr[k*n2+i, c]
                                if abs(s-1) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        U_2x_best = U_s
                        V_2x_best = V_s
                        W_2x_best = W_s
                        rank_2x = target_r
                        found = True
                        break
            except:
                pass
        if found:
            break
            
    # Combine two copies
    rank_new = rank_2x * 2
    U_new = [[0] * rank_new for _ in range(n*m)]
    V_new = [[0] * rank_new for _ in range(m*p)]
    W_new = [[0] * rank_new for _ in range(p*n)]
    
    r_2x = rank_2x
    # Fill first half (A1*B1)
    for i in range(n):
        for j1 in range(2):
            for c in range(r_2x):
                U_new[i*m + j1][c] = U_2x_best[i*m2 + j1][c]
                for k in range(p):
                     V_new[j1*p + k][c] = V_2x_best[j1*p2 + k][c]
                     W_new[k*n + i][c] = W_2x_best[k*n2 + i][c]
                     
    # Fill second half (A2*B2)
    offset = r_2x
    for i in range(n):
        for j2 in range(2):
            j_idx = j2 + 2
            for c in range(r_2x):
                U_new[i*m + j_idx][offset + c] = U_2x_best[i*m2 + j2][c]
                for k in range(p):
                     V_new[j_idx*p + k][offset + c] = V_2x_best[j2*p2 + k][c]
                     W_new[k*n + i][offset + c] = W_2x_best[k*n2 + i][c]
                     
    if verify(U_new, V_new, W_new, n, m, p):
         if rank_new < best_rank:
             best = {"U": U_new, "V": V_new, "W": W_new}
             best_rank = rank_new

    return best
# EVOLVE-BLOCK-END
