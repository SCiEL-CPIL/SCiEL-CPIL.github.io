import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for W
    # P1: (1, 0, 0, 1) -> C00+=1, C01+=0, C10+=0, C11+=1
    # P2: (0, 0, 1, -1) -> C00+=0, C01+=0, C10+=1, C11+=-1
    # P3: (0, 1, 0, 1) -> C00+=0, C01+=1, C10+=0, C11+=1
    # P4: (1, 0, 1, 0) -> C00+=1, C01+=0, C10+=1, C11+=0
    # P5: (-1, 1, 0, 0) -> C00+=-1, C01+=1, C10+=0, C11+=0
    # P6: (0, 0, 0, 1) -> C00+=0, C01+=0, C10+=0, C11+=1
    # P7: (1, 0, 0, 0) -> C00+=1, C01+=0, C10+=0, C11+=0
    
    # Wait, let's re-verify the Strassen formula mapping to C indices.
    # C = [ C00, C01 ]
    #     [ C10, C11 ]
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Coeffs for P1: C00:+1, C01:0, C10:0, C11:+1
    # Coeffs for P2: C00:0, C01:0, C10:+1, C11:-1
    # Coeffs for P3: C00:0, C01:+1, C10:0, C11:+1
    # Coeffs for P4: C00:+1, C01:0, C10:+1, C11:0
    # Coeffs for P5: C00:-1, C01:+1, C10:0, C11:0
    # Coeffs for P6: C00:0, C01:0, C10:0, C11:+1
    # Coeffs for P7: C00:+1, C01:0, C10:0, C11:0
    
    W_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # U and V factors for each P_s
    # Each is a list of (row_index, value)
    # U row index: i*m + j
    # V row index: j*p + k
    
    # P1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # P2: (A10+A11)B00
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # P3: A00(B01-B11)
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # P4: A11(B10-B00)
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # P5: (A00+A01)B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # P6: (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # P7: (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    
    # Note: The k indices in V are relative to the block start.
    # In the code, we will add k_start to the k part of the index.
    
    U_factors = [
        [(0, 1), (5, 1)], # P1: i=0,j=0; i=1,j=1 -> 0*4+0=0, 1*4+1=5
        [(4, 1), (5, 1)], # P2: i=1,j=0; i=1,j=1 -> 1*4+0=4, 1*4+1=5
        [(0, 1)],         # P3: i=0,j=0 -> 0
        [(5, 1)],         # P4: i=1,j=1 -> 5
        [(0, 1), (1, 1)], # P5: i=0,j=0; i=0,j=1 -> 0, 1
        [(4, 1), (0, -1)],# P6: i=1,j=0; i=0,j=0 -> 4, 0
        [(1, 1), (5, -1)] # P7: i=0,j=1; i=1,j=1 -> 1, 5
    ]
    
    V_factors = [
        [(0, 1), (7, 1)], # P1: j=0,k=0; j=1,k=1 -> 0*6+0=0, 1*6+1=7
        [(0, 1)],         # P2: j=0,k=0 -> 0
        [(1, 1), (7, -1)],# P3: j=0,k=1; j=1,k=1 -> 1, 7
        [(6, 1), (0, -1)],# P4: j=1,k=0; j=0,k=0 -> 6, 0
        [(7, 1)],         # P5: j=1,k=1 -> 7
        [(0, 1), (1, 1)], # P6: j=0,k=0; j=0,k=1 -> 0, 1
        [(6, 1), (7, 1)]  # P7: j=1,k=0; j=1,k=1 -> 6, 7
    ]
    
    for g in range(2):
        j_offset = 2 * g
        for l in range(3):
            k_start = 2 * l
            for s in range(7):
                r = g * 21 + l * 7 + s
                
                # Fill U
                for idx, val in U_factors[s]:
                    # idx is the row index for j=0,1 case.
                    # We need to adjust j by j_offset.
                    # The idx was calculated as i*m + j.
                    # We need i*m + (j + j_offset).
                    # idx = i*4 + j.
                    # new_idx = i*4 + j + j_offset = idx + j_offset.
                    U[idx + j_offset, r] += val
                
                # Fill V
                for idx, val in V_factors[s]:
                    # idx is j*p + k.
                    # We need (j + j_offset)*p + k.
                    # idx = j*6 + k.
                    # new_idx = (j+jo)*6 + k = j*6 + k + jo*6 = idx + j_offset*6.
                    V[idx + j_offset * 6, r] += val
                
                # Fill W
                # W row index is k*n + i = 2k + i.
                # k is k_start or k_start+1.
                # i is 0 or 1.
                # The coefficients are for C00 (k=0, i=0), C01 (k=1, i=0), C10 (k=0, i=1), C11 (k=1, i=1).
                # Wait, the mapping of coeffs to k,i:
                # C00: k=0, i=0 -> W[0*2+0] = W[0]
                # C01: k=1, i=0 -> W[1*2+0] = W[2]
                # C10: k=0, i=1 -> W[0*2+1] = W[1]
                # C11: k=1, i=1 -> W[1*2+1] = W[3]
                
                # But k is shifted by k_start.
                # So C00 corresponds to k=k_start, i=0.
                # C01 corresponds to k=k_start+1, i=0.
                # C10 corresponds to k=k_start, i=1.
                # C11 corresponds to k=k_start+1, i=1.
                
                c = W_coeffs[s]
                # c[0] is C00, c[1] is C01, c[2] is C10, c[3] is C11
                
                W[k_start * 2 + 0, r] += c[0] # k_start, i=0
                W[(k_start + 1) * 2 + 0, r] += c[1] # k_start+1, i=0
                W[k_start * 2 + 1, r] += c[2] # k_start, i=1
                W[(k_start + 1) * 2 + 1, r] += c[3] # k_start+1, i=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
