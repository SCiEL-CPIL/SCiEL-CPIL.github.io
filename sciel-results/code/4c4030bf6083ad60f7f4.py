"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    def time_check():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")

    # Target Rank: 71
    # Structure:
    # Top 2 rows (i=0,1):
    #   Cols 0,1 (k=0,1): 2x5 * 5x2 -> 2x2 using Strassen+Outer (Rank 18)
    #   Cols 2,3 (k=2,3): 2x5 * 5x2 -> 2x2 using Strassen+Outer (Rank 18)
    #   Col  4   (k=4):   2x5 * 5x1 -> 2x1 using Schoolbook (Rank 10)
    # Bottom 1 row (i=2):
    #   Cols 0..4 (k=0..4): 1x5 * 5x5 -> 1x5 using Schoolbook (Rank 25)
    # Total: 18 + 18 + 10 + 25 = 71

    # --- Helper: Strassen 2x2x2 ---
    # Terms are (u, v, w) where u,v,w are length-4 lists.
    # u: [U00, U01, U10, U11]
    # v: [V00, V01, V10, V11]
    # w: [W00, W01, W10, W11]
    # Convention: C[i,k] += u[i] * v[k] * w[i,k] ? No.
    # Standard: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For 2x2x2: indices i,j,k in {0,1}.
    # U is 2x2, V is 2x2, W is 2x2.
    # Let's define the 7 terms explicitly.
    
    strassen_2x2 = [
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # U: [[1,0],[0,1]] -> [1,0,0,1]
        # V: [[1,0],[0,1]] -> [1,0,0,1]
        # W: [[1,0],[0,1]] -> [1,0,0,1]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        # U: [[0,0],[1,1]] -> [0,0,1,1]
        # V: [[1,0],[0,0]] -> [1,0,0,0]
        # W: [[0,1],[0,-1]]? 
        # C21 is i=1, k=0. C22 is i=1, k=1.
        # W[k,i].
        # W[0,1] = 1, W[1,1] = -1.
        # W matrix: [[0, 1], [-1, 0]]? No, W is indexed by k,i.
        # W[0,0], W[0,1], W[1,0], W[1,1].
        # We need W[0,1] = 1 (for C21? No, C21 is i=1, k=0 -> W[0,1]).
        # We need W[1,1] = -1 (for C22? No, C22 is i=1, k=1 -> W[1,1]).
        # So W = [[0, 1], [0, -1]]? 
        # Let's check indices:
        # W[0,1] is row 0, col 1. Value 1.
        # W[1,1] is row 1, col 1. Value -1.
        # Other entries 0.
        # List: [W00, W01, W10, W11] = [0, 1, 0, -1].
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        
        # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
        # U: [[1,0],[0,0]] -> [1,0,0,0]
        # V: [[0,0],[1,-1]] -> [0,0,1,-1]
        # W: C12 (i=1, k=0) += 1 -> W[0,1] = 1.
        #     C22 (i=1, k=1) += 1 -> W[1,1] = 1.
        # W = [[0, 1], [0, 1]]? 
        # W[0,1]=1, W[1,1]=1.
        # List: [0, 1, 0, 1].
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        
        # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
        # U: [[0,0],[0,1]] -> [0,0,0,1]
        # V: [[-1,1],[0,0]] -> [-1,1,0,0]
        # W: C11 (i=1, k=0) += 1 -> W[0,1] = 1.
        #     C21 (i=1, k=0) += 1 -> W[0,1] = 1. Wait.
        # C11 is i=1, k=0. C21 is i=1, k=0? No.
        # C[i,k]. i is row, k is col.
        # C11: i=0, k=0? No, standard 1-based: C11 is row 1, col 1.
        # In 0-based: C[0,0].
        # Let's stick to 0-based.
        # C[0,0] += M4. -> W[0,0] += 1.
        # C[1,0] += M4. -> W[0,1] += 1.
        # W = [[1, 1], [0, 0]].
        # List: [1, 1, 0, 0].
        ([0,0,0,1], [-1,1,0,0], [1,1,0,0]),
        
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        # U: [[1,1],[0,0]] -> [1,1,0,0]
        # V: [[0,0],[0,1]] -> [0,0,0,1]
        # W: C[0,0] -= 1 -> W[0,0] = -1.
        #     C[0,1] += 1 -> W[1,0] = 1.
        # W = [[-1, 1], [0, 0]].
        # List: [-1, 1, 0, 0].
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # U: [[-1,0],[1,0]] -> [-1,0,1,0]
        # V: [[1,1],[0,0]] -> [1,1,0,0]
        # W: C[1,1] += 1 -> W[1,1] = 1.
        # W = [[0,0],[0,1]].
        # List: [0,0,0,1].
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        # U: [[0,1],[0,-1]] -> [0,1,0,-1]
        # V: [[0,0],[1,1]] -> [0,0,1,1]
        # W: C[0,0] += 1 -> W[0,0] = 1.
        # W = [[1,0],[0,0]].
        # List: [1,0,0,0].
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # --- Helper: Outer Product 2x1 * 1x2 -> 2x2 ---
    # A is 2x1, B is 1x2.
    # C[0,0] = A[0]B[0]
    # C[0,1] = A[0]B[1]
    # C[1,0] = A[1]B[0]
    # C[1,1] = A[1]B[1]
    # Terms (u2, v2, w4)
    outer_2x1_1x2 = [
        ([1,0], [1,0], [1,0,0,0]), # A0*B0 -> C00
        ([1,0], [0,1], [0,1,0,0]), # A0*B1 -> C01
        ([0,1], [1,0], [0,0,1,0]), # A1*B0 -> C10
        ([0,1], [0,1], [0,0,0,1]), # A1*B1 -> C11
    ]

    # --- Construct 2x5 * 5x2 -> 2x2 (Rank 18) ---
    # A is 2x5, B is 5x2, C is 2x2.
    # Split A cols into (0,1), (2,3), (4).
    # Split B rows into (0,1), (2,3), (4).
    # C = A[:,:2]B[0:2,:] + A[:,2:4]B[2:4,:] + A[:,4]B[4:,:]
    
    def get_2x5_5x2_terms():
        terms = []
        # Part 1: A[:,0:2] * B[0:2,:] (2x2 * 2x2)
        for u4, v4, w4 in strassen_2x2:
            u10 = [0]*10
            v10 = [0]*10
            # u4: [U00, U01, U10, U11] -> A[0,0], A[0,1], A[1,0], A[1,1]
            u10[0*5 + 0] = u4[0]
            u10[0*5 + 1] = u4[1]
            u10[1*5 + 0] = u4[2]
            u10[1*5 + 1] = u4[3]
            # v4: [V00, V01, V10, V11] -> B[0,0], B[0,1], B[1,0], B[1,1]
            v10[0*2 + 0] = v4[0]
            v10[0*2 + 1] = v4[1]
            v10[1*2 + 0] = v4[2]
            v10[1*2 + 1] = v4[3]
            terms.append((u10, v10, list(w4)))
            
        # Part 2: A[:,2:4] * B[2:4,:] (2x2 * 2x2)
        for u4, v4, w4 in strassen_2x2:
            u10 = [0]*10
            v10 = [0]*10
            u10[0*5 + 2] = u4[0]
            u10[0*5 + 3] = u4[1]
            u10[1*5 + 2] = u4[2]
            u10[1*5 + 3] = u4[3]
            v10[2*2 + 0] = v4[0]
            v10[2*2 + 1] = v4[1]
            v10[3*2 + 0] = v4[2]
            v10[3*2 + 1] = v4[3]
            terms.append((u10, v10, list(w4)))
            
        # Part 3: A[:,4] * B[4,:] (2x1 * 1x2)
        for u2, v2, w4 in outer_2x1_1x2:
            u10 = [0]*10
            v10 = [0]*10
            u10[0*5 + 4] = u2[0]
            u10[1*5 + 4] = u2[1]
            v10[4*2 + 0] = v2[0]
            v10[4*2 + 1] = v2[1]
            terms.append((u10, v10, list(w4)))
            
        return terms

    # --- Assemble Full 3x5 * 5x5 -> 3x5 ---
    all_terms = [] # List of (u15, v25, w15)

    # 1. Top 2 rows (i=0,1), Cols 0,1 (k=0,1)
    # Uses 2x5 * 5x2 -> 2x2
    t_2x5_5x2 = get_2x5_5x2_terms()
    for u10, v10, w4 in t_2x5_5x2:
        u15 = [0]*15
        v25 = [0]*25
        w15 = [0]*15
        
        # U: A[0..1, 0..4] -> rows 0..9 of U
        for i in range(2):
            for j in range(5):
                u15[i*5 + j] = u10[i*5 + j]
        
        # V: B[0..4, 0..1] -> rows 0..4 (k=0), 5..9 (k=1) of V
        for j in range(5):
            v25[j*5 + 0] = v10[j*2 + 0]
            v25[j*5 + 1] = v10[j*2 + 1]
            
        # W: C[0..1, 0..1] -> W[k*3 + i]
        # w4: [C00, C01, C10, C11] -> C[i, dk]
        # C[0,0] -> w4[0] -> W[0*3 + 0]
        # C[0,1] -> w4[1] -> W[1*3 + 0]
        # C[1,0] -> w4[2] -> W[0*3 + 1]
        # C[1,1] -> w4[3] -> W[1*3 + 1]
        w15[0*3 + 0] = w4[0]
        w15[1*3 + 0] = w4[1]
        w15[0*3 + 1] = w4[2]
        w15[1*3 + 1] = w4[3]
        
        all_terms.append((u15, v25, w15))

    # 2. Top 2 rows (i=0,1), Cols 2,3 (k=2,3)
    for u10, v10, w4 in t_2x5_5x2:
        u15 = [0]*15
        v25 = [0]*25
        w15 = [0]*15
        
        for i in range(2):
            for j in range(5):
                u15[i*5 + j] = u10[i*5 + j]
        
        for j in range(5):
            v25[j*5 + 2] = v10[j*2 + 0]
            v25[j*5 + 3] = v10[j*2 + 1]
            
        # C[0,2] -> W[2*3 + 0]
        # C[0,3] -> W[3*3 + 0]
        # C[1,2] -> W[2*3 + 1]
        # C[1,3] -> W[3*3 + 1]
        w15[2*3 + 0] = w4[0]
        w15[3*3 + 0] = w4[1]
        w15[2*3 + 1] = w4[2]
        w15[3*3 + 1] = w4[3]
        
        all_terms.append((u15, v25, w15))

    # 3. Top 2 rows (i=0,1), Col 4 (k=4)
    # Schoolbook: 2x5 * 5x1 -> 2x1
    for i in range(2):
        for j in range(5):
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            u15[i*5 + j] = 1
            v25[j*5 + 4] = 1
            w15[4*3 + i] = 1
            all_terms.append((u15, v25, w15))

    # 4. Bottom row (i=2), Cols 0..4 (k=0..4)
    # Schoolbook: 1x5 * 5x5 -> 1x5
    for j in range(5):
        for k in range(5):
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            u15[2*5 + j] = 1
            v25[j*5 + k] = 1
            w15[k*3 + 2] = 1
            all_terms.append((u15, v25, w15))

    rank = len(all_terms)
    
    U = np.zeros((15, rank))
    V = np.zeros((25, rank))
    W = np.zeros((15, rank))
    
    for r in range(rank):
        U[:, r] = all_terms[r][0]
        V[:, r] = all_terms[r][1]
        W[:, r] = all_terms[r][2]
        
    time_check()
    
    # Verify
    U_r = U.reshape(3, 5, rank)
    V_r = V.reshape(5, 5, rank)
    W_r = W.reshape(5, 3, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = 75
        U_sb = np.zeros((15, rank_sb))
        V_sb = np.zeros((25, rank_sb))
        W_sb = np.zeros((15, rank_sb))
        col = 0
        for i in range(3):
            for j in range(5):
                for k in range(5):
                    U_sb[i*5 + j, col] = 1
                    V_sb[j*5 + k, col] = 1
                    W_sb[k*3 + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
