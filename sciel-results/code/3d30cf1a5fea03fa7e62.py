"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7).
    Indexing convention:
    A: [[a, b], [c, d]] -> U rows 0:a, 1:b, 2:c, 3:d
    B: [[e, f], [g, h]] -> V rows 0:e, 1:f, 2:g, 3:h
    C: [[i, j], [k, l]] -> W rows 0:i, 1:j, 2:k, 3:l
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Vectorized construction
    # U is the same for all blocks
    for t in range(num_pairs):
        if _time_remaining() < 1.0:
            # If we run out of time, return what we have (though this shouldn't happen for 8 iterations)
            break
        U[:, t*7:(t+1)*7] = U_core
        
        # V indices: j*p + k
        # For block t, k is 2t and 2t+1
        # j=0: indices 2t, 2t+1
        # j=1: indices p+2t, p+2t+1
        V[2*t, t*7:(t+1)*7] = V_core[0, :]
        V[2*t+1, t*7:(t+1)*7] = V_core[1, :]
        V[p+2*t, t*7:(t+1)*7] = V_core[2, :]
        V[p+2*t+1, t*7:(t+1)*7] = V_core[3, :]
        
        # W indices: k*n + i
        # For block t, k is 2t and 2t+1
        # k=2t: indices 4t, 4t+1
        # k=2t+1: indices 4t+2, 4t+3
        # W_core rows: 0:(i=0, k=0), 1:(i=1, k=0), 2:(i=0, k=1), 3:(i=1, k=1)
        # Wait, let's re-verify W_core row mapping from _strassen_2x2
        # W[0,0]=1, W[3,0]=1 -> M1 contributes to C[0,0] and C[1,1]. So W row 0 is C[0,0], W row 3 is C[1,1]?
        # Let's check standard Strassen W matrix definition in the helper.
        # In _strassen_2x2:
        # M1: W[0,0]=1, W[3,0]=1.
        # M2: W[2,1]=1, W[3,1]=-1.
        # M3: W[1,2]=1, W[3,2]=1.
        # M4: W[0,3]=1, W[2,3]=1.
        # M5: W[0,4]=-1, W[1,4]=1.
        # M6: W[3,5]=1.
        # M7: W[0,6]=1.
        
        # Let's trace C[0,0] (index 0 in W rows if W is flattened C):
        # C[0,0] = a*e + b*g = M1 - M5 + M6 + M7?
        # M1 = (a+d)(e+h) = ae + ah + de + dh
        # M5 = (a+b)h = ah + bh
        # M6 = (c-a)(e+f) = ce + cf - ae - af
        # M7 = (b-d)(g+h) = bg + bh - dg - dh
        # M1 - M5 + M6 + M7 = (ae + ah + de + dh) - (ah + bh) + (ce + cf - ae - af) + (bg + bh - dg - dh)
        # = ae - bh + ce + cf - ae - af + bg + bh - dg
        # = ce + cf + bg - dg - af ... this is not C[0,0].
        
        # The standard Strassen decomposition is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M5 + M6 + M7
        # C12 = M3 + M5
        # C21 = M2 - M4 - M7
        # C22 = M1 - M2 + M4 + M3
        
        # Let's check my W assignments in _strassen_2x2 against this.
        # My W[0] corresponds to C11?
        # W[0, 0]=1 (M1)
        # W[0, 4]=-1 (M5)
        # W[0, 6]=1 (M7)
        # Missing M6? W[0, 5] is 0.
        # Ah, my _strassen_2x2 W matrix construction in the previous code was:
        # M1: W[0,0]=1, W[3,0]=1
        # M2: W[2,1]=1, W[3,1]=-1
        # M3: W[1,2]=1, W[3,2]=1
        # M4: W[0,3]=1, W[2,3]=1
        # M5: W[0,4]=-1, W[1,4]=1
        # M6: W[3,5]=1  <-- Only W[3,5] set. But C11 needs M6.
        # M7: W[0,6]=1
        
        # It seems the W matrix in the helper might be incorrect or I am misinterpreting the rows.
        # Let's re-verify the W matrix construction in _strassen_2x2.
        # If W rows are C[0,0], C[0,1], C[1,0], C[1,1]:
        # C[0,0] needs M1, -M5, M6, M7.
        # My helper:
        # M1: W[0,0]=1. Correct.
        # M5: W[0,4]=-1. Correct.
        # M7: W[0,6]=1. Correct.
        # M6: W[3,5]=1. This puts M6 in C[1,1].
        # But C[1,1] = M1 - M2 + M4 + M3.
        # My helper for C[1,1] (row 3):
        # M1: W[3,0]=1. Correct.
        # M2: W[3,1]=-1. Correct.
        # M3: W[3,2]=1. Correct.
        # M4: W[3,3] is 0 in helper? No, M4: W[0,3]=1, W[2,3]=1. So W[3,3] is 0.
        # Wait, M4 should be in C[1,1].
        
        # The helper _strassen_2x2 has bugs in the W matrix.
        # I need to fix the W matrix in _strassen_2x2.
        
        pass # Placeholder, will rewrite _strassen_2x2 correctly below.
        
    # The loop above was just analysis. I need to rewrite the function with the correct W matrix.
    # I will implement the correct Strassen W matrix here.
    
    # Correct Strassen W matrix (rows C11, C12, C21, C22; cols M1..M7)
    # C11 = M1 - M5 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 - M7
    # C22 = M1 - M2 + M4 + M3
    
    W_core_correct = np.array([
        [1, 0, 0, 0, -1, 1, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, -1, 0, 0, -1], # C21
        [1, -1, 1, 1, 0, 0, 0]  # C22
    ], dtype=np.int64)
    
    # Re-run construction with correct W
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        U[:, t*7:(t+1)*7] = U_core
        V[2*t, t*7:(t+1)*7] = V_core[0, :]
        V[2*t+1, t*7:(t+1)*7] = V_core[1, :]
        V[p+2*t, t*7:(t+1)*7] = V_core[2, :]
        V[p+2*t+1, t*7:(t+1)*7] = V_core[3, :]
        
        # W mapping:
        # Block t corresponds to columns 2t, 2t+1 of B.
        # Output C has 2 rows (n=2) and 16 columns (p=16).
        # C row i, col k. Index in W is k*n + i.
        # For col 2t: indices 4t (i=0), 4t+1 (i=1).
        # For col 2t+1: indices 4t+2 (i=0), 4t+3 (i=1).
        
        # W_core rows:
        # 0: C11 (i=0, local col 0)
        # 1: C12 (i=1, local col 0) -- Wait.
        # Standard 2x2 multiplication:
        # C = A B
        # A is 2x2, B is 2x2.
        # C[0,0] = M1 - M5 + M6 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 - M4 - M7
        # C[1,1] = M1 - M2 + M4 + M3
        
        # So W_core rows are:
        # Row 0: C[0,0]
        # Row 1: C[0,1]
        # Row 2: C[1,0]
        # Row 3: C[1,1]
        
        # In the 2x16 problem, the "2x2" block for B columns 2t, 2t+1 produces C rows 0,1 for columns 2t, 2t+1.
        # So:
        # C[0, 2t] corresponds to C[0,0] of the block -> W_core row 0.
        # C[0, 2t+1] corresponds to C[0,1] of the block -> W_core row 1.
        # C[1, 2t] corresponds to C[1,0] of the block -> W_core row 2.
        # C[1, 2t+1] corresponds to C[1,1] of the block -> W_core row 3.
        
        # W index for C[i, k] is k*n + i = 2*k + i.
        # k=2t, i=0 -> 4t.
        # k=2t+1, i=0 -> 4t+2.
        # k=2t, i=1 -> 4t+1.
        # k=2t+1, i=1 -> 4t+3.
        
        W[4*t, t*7:(t+1)*7] = W_core_correct[0, :]
        W[4*t+1, t*7:(t+1)*7] = W_core_correct[2, :]
        W[4*t+2, t*7:(t+1)*7] = W_core_correct[1, :]
        W[4*t+3, t*7:(t+1)*7] = W_core_correct[3, :]
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
