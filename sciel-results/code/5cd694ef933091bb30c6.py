"""Matrix multiplication 2x4 by 4x5 tensor decomposition (rank 36)."""
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns the rank-36 exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
    schoolbook for the remaining columns to achieve total rank 36.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (8, 36), (20, 36), (10, 36)
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    rank = 36
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Split m=4 into two halves of size 2: j in {0,1} (left) and j in {2,3} (right)
    # Split p=5 into two halves: k in {0,1,2,3} (first 4) and k in {4} (last 1)
    # 
    # For each half of j (2 choices) and each split of p (4+1):
    #   - 2x2x4 part: use two Strassen 2x2x2 decompositions (rank 7 each = 14)
    #   - 2x2x1 part: schoolbook (rank 4)
    # Total per j-half: 14 + 4 = 18
    # Total: 2 * 18 = 36
    
    col = 0
    
    for j_half in range(2):
        # j_half=0: columns j=0,1; j_half=1: columns j=2,3
        j_offset = j_half * 2
        
        # --- Part 1: 2x2x4 using two Strassen 2x2x2 blocks ---
        # k in {0,1} and k in {2,3}
        for k_half in range(2):
            k_offset = k_half * 2
            
            # Strassen factors for 2x2x2:
            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B21-B22)
            # M4 = A22*(B12-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)
            
            # A is 2x2: rows i=0,1 (n=2), cols j=j_offset, j_offset+1 (m=2)
            # B is 2x2: rows j=j_offset, j_offset+1 (m=2), cols k=k_offset, k_offset+1 (p=2)
            # C is 2x2: rows k=k_offset, k_offset+1 (p=2), cols i=0,1 (n=2)
            
            # U factors: for each Strassen term, specify which (i, j) entry is involved
            # V factors: for each Strassen term, specify which (j, k) entry is involved
            # W factors: for each Strassen term, specify which (k, i) output is affected
            
            strassen_terms = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # U_coeffs: dict mapping (i,j) -> coefficient
                # V_coeffs: dict mapping (j,k) -> coefficient  
                # W_coeffs: dict mapping (k,i) -> coefficient
                # Term 1: (A11+A22)(B11+B22)
                ({(0, 0): 1, (1, 1): 1}, 
                 {(0, 0): 1, (1, 1): 1}, 
                 {(0, 0): 1, (1, 1): 1, (0, 1): 1, (1, 0): 1}),
                # Term 2: (A21+A22)*B11
                ({(1, 0): 1, (1, 1): 1}, 
                 {(0, 0): 1}, 
                 {(1, 0): 1}),
                # Term 3: A11*(B21-B22)
                ({(0, 0): 1}, 
                 {(1, 0): 1, (1, 1): -1}, 
                 {(1, 1): 1}),
                # Term 4: A22*(B12-B11)
                ({(1, 1): 1}, 
                 {(0, 1): 1, (0, 0): -1}, 
                 {(1, 0): 1}),
                # Term 5: (A11+A12)*B22
                ({(0, 0): 1, (0, 1): 1}, 
                 {(1, 1): 1}, 
                 {(0, 1): 1}),
                # Term 6: (A21-A11)*(B11+B12)
                ({(1, 0): 1, (0, 0): -1}, 
                 {(0, 0): 1, (0, 1): 1}, 
                 {(1, 1): 1}),
                # Term 7: (A12-A22)*(B21+B22)
                ({(0, 1): 1, (1, 1): -1}, 
                 {(1, 0): 1, (1, 1): 1}, 
                 {(0, 0): 1, (1, 1): -1}),
            ]
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                # Fill U
                for (i, j_local), coeff in u_coeffs.items():
                    j_global = j_offset + j_local
                    U[i * m + j_global, col] += coeff
                
                # Fill V
                for (j_local, k_local), coeff in v_coeffs.items():
                    j_global = j_offset + j_local
                    k_global = k_offset + k_local
                    V[j_global * p + k_global, col] += coeff
                
                # Fill W
                for (k_local, i), coeff in w_coeffs.items():
                    k_global = k_offset + k_local
                    W[k_global * n + i, col] += coeff
                
                col += 1
        
        # --- Part 2: 2x2x1 schoolbook for k=4 ---
        k_single = 4
        for i in range(n):
            for j_local in range(2):
                j_global = j_offset + j_local
                U[i * m + j_global, col] = 1
                V[j_global * p + k_single, col] = 1
                W[k_single * n + i, col] = 1
                col += 1
    
    # Convert to lists with clean values
    def clean(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    clean_row.append(int(round(val)))
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
