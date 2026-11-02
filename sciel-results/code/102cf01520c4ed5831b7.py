import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the 2x2x2 block at indices (0,1)x(0,1)x(0,1)
    # and schoolbook for everything else.
    
    # Total rank will be 105 - 1 = 104.
    # 7 for Strassen + (105 - 8) for schoolbook.
    
    U = []
    V = []
    W = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, u, v, w):
        # u, v, w are the matrices to append to
        u.append([0] * len(U) if U else 0) # This logic is flawed, let's build lists of rows first.
        
    # Let's build the columns directly.
    rank = 0
    U_rows = {}
    V_rows = {}
    W_rows = {}
    
    # Initialize
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen terms for i in [0,1], j in [0,1], k in [0,1]
    # M1
    U_list.append({(0,0): 1, (1,1): 1})
    V_list.append({(0,0): 1, (1,1): 1})
    W_list.append({(0,0): 1, (1,1): 1})
    rank += 1
    
    # M2
    U_list.append({(1,0): 1, (1,1): 1})
    V_list.append({(0,0): 1})
    W_list.append({(1,0): 1, (1,1): 1}) # Wait, C21 = M2 + M4. C22 = M1 - M2 + ...
    # Let's map W correctly.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Reset
    U_list = []
    V_list = []
    W_list = []
    
    # M1: (A00+A11)(B00+B11)
    U_list.append({(0,0): 1, (1,1): 1})
    V_list.append({(0,0): 1, (1,1): 1})
    W_list.append({(0,0): 1, (1,1): -1}) # C00 += M1, C11 += M1. Wait.
    # C00 gets M1. C11 gets M1.
    # So W for M1 should have 1 at (0,0) and 1 at (1,1).
    W_list[-1] = {(0,0): 1, (1,1): 1}
    
    # M2: (A10+A11)B00
    U_list.append({(1,0): 1, (1,1): 1})
    V_list.append({(0,0): 1})
    W_list.append({(1,0): 1, (1,1): -1}) # C10 += M2, C11 -= M2.
    
    # M3: A00(B10-B11)
    U_list.append({(0,0): 1})
    V_list.append({(1,0): 1, (1,1): -1})
    W_list.append({(0,1): 1, (1,1): 1}) # C01 += M3, C11 += M3.
    
    # M4: A11(B01-B00)
    U_list.append({(1,1): 1})
    V_list.append({(0,1): 1, (0,0): -1})
    W_list.append({(0,0): 1, (1,0): 1}) # C00 += M4, C10 += M4.
    
    # M5: (A00+A01)B11
    U_list.append({(0,0): 1, (0,1): 1})
    V_list.append({(1,1): 1})
    W_list.append({(0,0): -1, (0,1): 1}) # C00 -= M5, C01 += M5.
    
    # M6: (A10-A00)(B00+B01)
    U_list.append({(1,0): 1, (0,0): -1})
    V_list.append({(0,0): 1, (0,1): 1})
    W_list.append({(1,1): 1}) # C11 += M6.
    
    # M7: (A01-A11)(B10+B11)
    U_list.append({(0,1): 1, (1,1): -1})
    V_list.append({(1,0): 1, (1,1): 1})
    W_list.append({(0,0): 1}) # C00 += M7.
    
    # Now add schoolbook terms for the rest.
    # Indices:
    # A: 0..2, 0..4
    # B: 0..4, 0..6
    # C: 0..2, 0..6
    
    # Strassen covered A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is covered by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                # Add schoolbook term
                u_dict = {(i, j): 1}
                v_dict = {(j, k): 1}
                w_dict = {(i, k): 1}
                U_list.append(u_dict)
                V_list.append(v_dict)
                W_list.append(w_dict)
                
    # Convert dicts to lists
    final_U = []
    final_V = []
    final_W = []
    
    # We need to flatten the dictionaries into vectors of size n*m, m*p, p*n
    # But the interface expects matrices of shape (n*m, rank), etc.
    # So we build the columns.
    
    rank_final = len(U_list)
    
    U_mat = [[0] * rank_final for _ in range(n * m)]
    V_mat = [[0] * rank_final for _ in range(m * p)]
    W_mat = [[0] * rank_final for _ in range(p * n)]
    
    for r in range(rank_final):
        for (i, j), val in U_list[r].items():
            U_mat[i * m + j][r] = val
        for (j, k), val in V_list[r].items():
            V_mat[j * p + k][r] = val
        for (i, k), val in W_list[r].items():
            W_mat[k * n + i][r] = val
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
