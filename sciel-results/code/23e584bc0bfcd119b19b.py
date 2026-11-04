import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    def add_strassen(i_base, j_base, k_base, signs):
        nonlocal col
        # i_base, j_base, k_base are offsets for the 2x2 block
        # signs is a 4x4 matrix of coefficients for C
        # Actually, we just need the 7 products
        
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # We map local indices (0,1) to global
        # A local (u,v) -> global (i_base+u, j_base+v)
        # B local (u,v) -> global (j_base+u, k_base+v)
        # C local (u,v) -> global (i_base+u, k_base+v)
        
        prods = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: 2x2, V: 2x2, W: 2x2
            # P1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00+=1, C01+=0, C10+=0, C11+=1? No, W is 2x2
            # Wait, W in the interface is (p*n, rank).
            # For a 2x2 block, W local is 2x2.
            # We need to define W for the 4 output cells.
            
            # Let's define the 7 terms explicitly with U, V, W matrices of size 2x2 (flattened)
        ]
        
        # Term 1: P1
        U1 = [1, 0, 0, 1] # A00+A11
        V1 = [1, 0, 0, 1] # B00+B11
        W1 = [1, 0, 0, 1] # C00+=1, C11+=1? No.
        # C = P1 + P4 - P5 + P7
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We add 7 columns.
        
        # Col 0: P1
        # U: A00+A11 -> [1,0,0,1]
        # V: B00+B11 -> [1,0,0,1]
        # W: C00+=1, C11+=1 -> [1,0,0,1]
        # Note: W index is k*n+i. Local k,u. Global k_base+u, i_base+v?
        # W global index: (k_base+u)*n + (i_base+v)
        
        U_flat = [1, 0, 0, 1]
        V_flat = [1, 0, 0, 1]
        W_flat = [1, 0, 0, 1] # C00, C11 get 1
        
        for u in range(2):
            for v in range(2):
                if U_flat[u*2+v] != 0:
                    U[(i_base+u)*m + (j_base+v)][col] = U_flat[u*2+v]
                if V_flat[u*2+v] != 0:
                    V[(j_base+u)*p + (k_base+v)][col] = V_flat[u*2+v]
                if W_flat[u*2+v] != 0:
                    W[(k_base+v)*n + (i_base+u)][col] = W_flat[u*2+v]
        col += 1
        
        # ... and so on for P2..P7
        
