import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Top-level Rank: 7 * 7 = 49
    R = 49
    U = np.zeros((16, R), dtype=float)
    V = np.zeros((24, R), dtype=float)
    W = np.zeros((24, R), dtype=float)
    
    # Top-level Strassen coefficients for 4x4 * 4x6
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11 (each 2x3)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Inner decomposition for 2x2 * 2x3
    # We need to construct U_inner, V_inner, W_inner for this block
    # U_inner: (4, 7)
    # V_inner: (6, 7)
    # W_inner: (6, 7)
    
    # Standard Strassen U vectors (coeffs for A00, A01, A10, A11)
    U_std = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    # Standard Strassen V vectors (coeffs for B00, B01, B10, B11)
    # Note: B is 2x3. Indices 0,1,2 for cols.
    # V rows: 0,1 for col 0; 2,3 for col 1; 4,5 for col 2.
    # Standard Strassen uses cols 0 and 1.
    V_std = np.zeros((6, 7), dtype=float)
    # Term 1: B00 + B11 -> V[0] += 1, V[3] += 1
    V_std[0, 0] = 1; V_std[3, 0] = 1
    # Term 2: B00 -> V[0] += 1
    V_std[0, 1] = 1
    # Term 3: B01 - B11 -> V[2] += 1, V[3] += -1
    V_std[2, 2] = 1; V_std[3, 2] = -1
    # Term 4: B10 - B00 -> V[1] += 1, V[0] += -1
    V_std[1, 3] = 1; V_std[0, 3] = -1
    # Term 5: B11 -> V[3] += 1
    V_std[3, 4] = 1
    # Term 6: B00 + B01 -> V[0] += 1, V[2] += 1
    V_std[0, 5] = 1; V_std[2, 5] = 1
    # Term 7: B10 + B11 -> V[1] += 1, V[3] += 1
    V_std[1, 6] = 1; V_std[3, 6] = 1
    
    # Standard Strassen W vectors (coeffs for C00, C01, C10, C11)
    # W rows: 0,1 for col 0; 2,3 for col 1; 4,5 for col 2.
    W_std = np.zeros((6, 7), dtype=float)
    # C00 = M1 + M4 - M5 - M7
    W_std[0, 0] = 1; W_std[0, 3] = 1; W_std[0, 4] = -1; W_std[0, 6] = -1
    # C01 = M3 + M5
    W_std[1, 2] = 1; W_std[1, 4] = 1
    # C10 = M2 + M7
    W_std[2, 1] = 1; W_std[2, 6] = 1
    # C11 = M1 - M2 + M3 + M6
    W_std[3, 0] = 1; W_std[3, 1] = -1; W_std[3, 2] = 1; W_std[3, 5] = 1
    
    # Now solve for V_ext (6x7) and W_ext (6x7) for col 2
    # Constraints:
    # sum_t W_std[0,t] * (U_std[t] . V_ext[t]) = 0
    # sum_t W_std[1,t] * (U_std[t] . V_ext[t]) = 0
    # sum_t W_std[2,t] * (U_std[t] . V_ext[t]) = 0
    # sum_t W_std[3,t] * (U_std[t] . V_ext[t]) = 0
    # sum_t W_ext[4,t] * (U_std[t] . V_ext[t]) = A00*B02 + A01*B12
    # sum_t W_ext[5,t] * (U_std[t] . V_ext[t]) = A10*B02 + A11*B12
    
    # This is bilinear. We can fix W_ext and solve for V_ext?
    # Or fix V_ext and solve for W_ext?
    # Let's try to set V_ext to be simple.
    # Suppose V_ext is non-zero only for t=0, 1.
    # V_ext[4, 0] = v00, V_ext[5, 0] = v01
    # V_ext[4, 1] = v10, V_ext[5, 1] = v11
    # Others 0.
    
    # Then P0_ext = U_std[0] . [0,0,0,0,v00,v01] = 1*v00 + 1*v01 = v00+v01
    # P1_ext = U_std[1] . [0,0,0,0,v10,v11] = 1*v10 + 1*v11 = v10+v11
    # P2_ext ... P6_ext = 0
    
    # Constraints:
    # W_std[0,0] P0_ext + W_std[0,1] P1_ext = 0
    # W_std[1,0] P0_ext + W_std[1,1] P1_ext = 0
    # W_std[2,0] P0_ext + W_std[2,1] P1_ext = 0
    # W_std[3,0] P0_ext + W_std[3,1] P1_ext = 0
    
    # W_std[0,0]=1, W_std[0,1]=0 -> P0_ext = 0
    # W_std[1,0]=0, W_std[1,1]=0 -> 0=0
    # W_std[2,0]=0, W_std[2,1]=1 -> P1_ext = 0
    # W_std[3,0]=1, W_std[3,1]=-1 -> P0_ext - P1_ext = 0
    
    # So P0_ext = 0 and P1_ext = 0.
    # Then C02 = W_ext[4,0] P0_ext + W_ext[4,1] P1_ext = 0.
    # But we need C02 != 0.
    # So this choice of V_ext is too restrictive.
    
    # We need more degrees of freedom.
    # Let's use all 7 terms.
    # This is a system of 6 equations for 28 unknowns.
    # We can use a simple ansatz.
    # Let V_ext[t] = [1, 0] for all t?
    # Then P_t_ext = U_std[t, 0] + U_std[t, 2] * 0? No.
    # V_ext[t] = [v_t0, v_t1].
    # P_t_ext = U_std[t,0]v_t0 + U_std[t,1]v_t1 + U_std[t,2]v_t0 + U_std[t,3]v_t1?
    # No, U_std[t] is [A00, A01, A10, A11].
    # V_ext[t] is [B02, B12].
    # P_t_ext = A00 U_t0 B02_v + A01 U_t1 B12_v + ...
    # Actually, P_t_ext = sum_i U_std[t,i] V_ext[t,i].
    
    # Let's just solve it numerically.
    # But we can't iterate in construct() easily if it's complex.
    # However, we can hardcode a solution.
    
    # Alternative: Use the fact that we can add NEW terms.
    # If we can't reuse, we add 4 terms. Rank 11.
    # If we CAN reuse, Rank 7.
    
    # Let's assume we can find a solution.
    # For now, let's try to implement the rank 49 solution.
    
    # To make it robust, we will just output the rank 77 solution if we can't easily solve the system.
    # But the prompt asks to improve.
    
    # Let's try to solve the system symbolically or with a fixed pattern.
    
    pass
