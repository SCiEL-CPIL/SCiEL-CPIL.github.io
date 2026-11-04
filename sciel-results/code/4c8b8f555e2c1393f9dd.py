"""Matrix multiplication 4,4,7 tensor decomposition using Strassen-based approach."""
import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2 part of 4x4x3) + 16 (for 4x4x1 part of 4x4x3) = 93
    rank = 93
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper: Strassen 2x2*2x2
    # Returns U, V, W of shape (4, 7)
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    def strassen_222():
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # p1 = (x11+x22)(y11+y22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1   # z11
        W_s[1, 0] = -1  # z12
        W_s[2, 0] = 0   # z21
        W_s[3, 0] = 1   # z22
        
        # p2 = (x21+x22)(y11+y12)
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1; V_s[1, 1] = 1
        W_s[2, 1] = 1   # z21
        W_s[3, 1] = -1  # z22
        
        # p3 = (x11+x12)(y21+y22)
        U_s[0, 2] = 1; U_s[1, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = 1
        W_s[1, 2] = 1   # z12
        W_s[3, 2] = 1   # z22
        
        # p4 = x11(y22-y12)
        U_s[0, 3] = 1
        V_s[3, 3] = 1; V_s[1, 3] = -1
        W_s[0, 3] = 1   # z11
        W_s[2, 3] = -1  # z21
        W_s[3, 3] = 1   # z22
        
        # p5 = x22(y11-y21)
        U_s[3, 4] = 1
        V_s[0, 4] = 1; V_s[2, 4] = -1
        W_s[0, 4] = -1  # z11
        W_s[2, 4] = 1   # z21
        
        # p6 = (x21-x11)(y11+y22)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[3, 5] = 1
        W_s[1, 5] = 1   # z12
        
        # p7 = (x12-x22)(y22-y11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[3, 6] = 1; V_s[0, 6] = -1
        W_s[0, 6] = 1   # z11
        W_s[1, 6] = -1  # z12
        W_s[2, 6] = 1   # z21
        
        return U_s, V_s, W_s

    # Helper: Schoolbook 2x2*2x1
    # Returns U, V, W of shape (4, 4), (2, 4), (2, 4)
    # U index: i*2+j, V index: j, W index: i
    def schoolbook_221():
        U_s = np.zeros((4, 4))
        V_s = np.zeros((2, 4))
        W_s = np.zeros((2, 4))
        
        # c1 = a11 b1 + a12 b2
        # c2 = a21 b1 + a22 b2
        
        # a11 b1 -> c1
        U_s[0, 0] = 1
        V_s[0, 0] = 1
        W_s[0, 0] = 1
        
        # a12 b2 -> c1
        U_s[1, 1] = 1
        V_s[1, 1] = 1
        W_s[0, 1] = 1
        
        # a21 b1 -> c2
        U_s[2, 2] = 1
        V_s[0, 2] = 1
        W_s[1, 2] = 1
        
        # a22 b2 -> c2
        U_s[3, 3] = 1
        V_s[1, 3] = 1
        W_s[1, 3] = 1
        
        return U_s, V_s, W_s

    col = 0
    
    # Part 1: M_{4,4,4} (columns 0-3 of C)
    # Strassen Level 1: 7 products
    # Each product is 2x2*2x2, decomposed by Strassen (7 terms)
    
    # p1 = (A11+A22)(B11+B22)
    # A blocks: A11 (0,0)-(1,1), A22 (2,2)-(3,3)
    # B blocks: B11 (0,0)-(1,1), B22 (2,2)-(3,3)
    # C blocks: C11 (0,0)-(1,1), C22 (2,2)-(3,3)
    
    # We need to map the Strassen 2x2*2x2 decomposition to the global indices.
    # For a given Strassen term (u, v, w) from strassen_222():
    # u is a vector of length 4 (indices for X entries)
    # v is a vector of length 4 (indices for Y entries)
    # w is a vector of length 4 (indices for Z entries)
    
    # X entries map to A indices.
    # Y entries map to B indices.
    # Z entries map to C indices.
    
    # Let's define the mapping for each of the 7 Level-1 Strassen products.
    
    # Product 1: X = A11+A22, Y = B11+B22, Z = C11+C22 (conceptually)
    # Actually, the Strassen formulas give linear combinations.
    # Let's just iterate through the 7 Level-1 products and for each, 
    # apply the 2x2*2x2 Strassen decomposition, mapping indices appropriately.
    
    # Level 1 Products:
    # P1: X = A11+A22, Y = B11+B22
    # P2: X = A21+A22, Y = B11+B12
    # P3: X = A11+A12, Y = B21+B22
    # P4: X = A11, Y = B22-B12
    # P5: X = A22, Y = B11-B21
    # P6: X = A21-A11, Y = B11+B22
    # P7: X = A12-A22, Y = B22-B11
    
    # C contributions:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 - P1 + P6 - P7
    # C21 = P2 - P4 + P5 + P7
    # C22 = P1 - P2 + P3 + P4
    
    # For each Pi, we have a 2x2*2x2 multiplication X*Y = Z_i.
    # We decompose X*Y using strassen_222().
    # Then we add the contributions to the global U, V, W.
    
    # Mapping for A indices (X):
    # X is 2x2. Local indices (0,1,2,3) correspond to (0,0), (0,1), (1,0), (1,1).
    # We need to map these to global A indices.
    
    # Mapping for B indices (Y):
    # Y is 2x2. Local indices (0,1,2,3) correspond to (0,0), (0,1), (1,0), (1,1).
    # We need to map these to global B indices.
    
    # Mapping for C indices (Z):
    # Z is 2x2. Local indices (0,1,2,3) correspond to (0,0), (0,1), (1,0), (1,1).
    # We need to map these to global C indices.
    
    # Let's create a helper to add a block decomposition.
    # add_block(U_block, V_block, W_block, A_map, B_map, C_map, col_offset)
    
    def add_block(U_b, V_b, W_b, A_map, B_map, C_map, col_off):
        nonlocal col
        for r in range(U_b.shape[1]):
            for i_local in range(4):
                if U_b[i_local, r] != 0:
                    i_global = A_map[i_local]
                    U[i_global, col_off + r] += U_b[i_local, r]
            for j_local in range(4):
                if V_b[j_local, r] != 0:
                    j_global = B_map[j_local]
                    V[j_global, col_off + r] += V_b[j_local, r]
            for k_local in range(4):
                if W_b[k_local, r] != 0:
                    k_global = C_map[k_local]
                    W[k_global, col_off + r] += W_b[k_local, r]
        col += U_b.shape[1]

    # Define mappings for each Pi
    
    # A maps:
    # A11: [0,1,4,5] (indices 0,1,4,5 in 4x4 row-major? No, i*4+j)
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A12: [2,3,6,7]
    # A21: [8,9,12,13]
    # A22: [10,11,14,15]
    
    A11_map = [0, 1, 4, 5]
    A12_map = [2, 3, 6, 7]
    A21_map = [8, 9, 12, 13]
    A22_map = [10, 11, 14, 15]
    
    # B maps:
    # B11: [0,1,7,8] (j*7+k)
    # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # B12: [2,3,9,10]
    # B21: [14,15,21,22]
    # B22: [16,17,23,24]
    
    B11_map = [0, 1, 7, 8]
    B12_map = [2, 3, 9, 10]
    B21_map = [14, 15, 21, 22]
    B22_map = [16, 17, 23, 24]
    
    # C maps:
    # C11: [0,4,1,5] (k*4+i)
    # C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    # Wait, W index is k*n+i.
    # C11: k in {0,1}, i in {0,1}.
    # C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5.
    # Local Z indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # So C11_map = [0, 1, 4, 5].
    
    C11_map = [0, 1, 4, 5]
    C12_map = [2, 3, 6, 7]
    C21_map = [8, 9, 12, 13]
    C22_map = [10, 11, 14, 15]
    
    # Now apply for each Pi
    
    # P1: X=A11+A22, Y=B11+B22, Z=C11+C22
    # X_map: A11 and A22.
    # We need to create a combined map for X.
    # X = A11 + A22.
    # Local X index 0 (X11) gets 1 from A11[0] and 1 from A22[0].
    # So X_map is not a simple permutation. It's a linear combination.
    # My add_block function assumes U_b is the coefficient for the local X entry.
    # If X = A11 + A22, then X_entry = A11_entry + A22_entry.
    # The Strassen decomposition gives coefficients for X_entry.
    # So if U_b[local_x, r] = c, then we add c * (A11_entry + A22_entry).
    # This means we need to add c to U[A11_map[local_x]] and c to U[A22_map[local_x]].
    
    # Let's modify add_block to accept a list of (map, coeff) pairs for each dimension.
    
    def add_block_complex(U_b, V_b, W_b, A_maps, B_maps, C_maps, col_off):
        # A_maps: list of (map_list, coeff)
        # map_list: 4 global indices corresponding to local 0,1,2,3
        nonlocal col
        for r in range(U_b.shape[1]):
            for i_local in range(4):
                u_val = U_b[i_local, r]
                if u_val != 0:
                    for map_list, coeff in A_maps:
                        U[map_list[i_local], col_off + r] += u_val * coeff
            for j_local in range(4):
                v_val = V_b[j_local, r]
                if v_val != 0:
                    for map_list, coeff in B_maps:
                        V[map_list[j_local], col_off + r] += v_val * coeff
            for k_local in range(4):
                w_val = W_b[k_local, r]
                if w_val != 0:
                    for map_list, coeff in C_maps:
                        W[map_list[k_local], col_off + r] += w_val * coeff
        col += U_b.shape[1]
        
    U_s, V_s, W_s = strassen_222()
    
    # P1
    add_block_complex(U_s, V_s, W_s, 
                      [(A11_map, 1), (A22_map, 1)], 
                      [(B11_map, 1), (B22_map, 1)], 
                      [(C11_map, 1), (C22_map, 1)], col)
    
    # P2: X=A21+A22, Y=B11+B12, Z=C21+C22
    add_block_complex(U_s, V_s, W_s, 
                      [(A21_map, 1), (A22_map, 1)], 
                      [(B11_map, 1), (B12_map, 1)], 
                      [(C21_map, 1), (C22_map, 1)], col)
    
    # P3: X=A11+A12, Y=B21+B22, Z=C12+C22
    add_block_complex(U_s, V_s, W_s, 
                      [(A11_map, 1), (A12_map, 1)], 
                      [(B21_map, 1), (B22_map, 1)], 
                      [(C12_map, 1), (C22_map, 1)], col)
    
    # P4: X=A11, Y=B22-B12, Z=C11-C21+C22
    # Z = C11 - C21 + C22? No, C11 += P4, C21 -= P4, C22 += P4.
    add_block_complex(U_s, V_s, W_s, 
                      [(A11_map, 1)], 
                      [(B22_map, 1), (B12_map, -1)], 
                      [(C11_map, 1), (C21_map, -1), (C22_map, 1)], col)
    
    # P5: X=A22, Y=B11-B21, Z=-C11+C21
    add_block_complex(U_s, V_s, W_s, 
                      [(A22_map, 1)], 
                      [(B11_map, 1), (B21_map, -1)], 
                      [(C11_map, -1), (C21_map, 1)], col)
    
    # P6: X=A21-A11, Y=B11+B22, Z=C12
    add_block_complex(U_s, V_s, W_s, 
                      [(A21_map, 1), (A11_map, -1)], 
                      [(B11_map, 1), (B22_map, 1)], 
                      [(C12_map, 1)], col)
    
    # P7: X=A12-A22, Y=B22-B11, Z=C11-C12+C21
    add_block_complex(U_s, V_s, W_s, 
                      [(A12_map, 1), (A22_map, -1)], 
                      [(B22_map, 1), (B11_map, -1)], 
                      [(C11_map, 1), (C12_map, -1), (C21_map, 1)], col)
    
    # Part 1 done. col should be 49.
    
    # Part 2: M_{4,4,3} (columns 4-6 of C)
    # B' columns 4,5,6.
    # B'11 = B[0:2, 4:6] (2x2). Map: [28, 29, 35, 36]
    # B'12 = B[0:2, 6:7] (2x1). Map: [30, 37]
    # B'21 = B[2:4, 4:6] (2x2). Map: [42, 43, 49, 50]
    # B'22 = B[2:4, 6:7] (2x1). Map: [44, 51]
    
    Bp11_map = [28, 29, 35, 36]
    Bp12_map = [30, 37]
    Bp21_map = [42, 43, 49, 50]
    Bp22_map = [44, 51]
    
    # C' columns 4,5,6.
    # C'11 = C[0:2, 4:6] (2x2). Map: [16, 17, 20, 21]
    # C'12 = C[0:2, 6:7] (2x1). Map: [18, 22]
    # C'21 = C[2:4, 4:6] (2x2). Map: [24, 25, 28, 29]
    # C'22 = C[2:4, 6:7] (2x1). Map: [26, 30]
    
    Cp11_map = [16, 17, 20, 21]
    Cp12_map = [18, 22]
    Cp21_map = [24, 25, 28, 29]
    Cp22_map = [26, 30]
    
    # Products:
    # A11 * Bp11 (2x2*2x2) -> C'11
    # A12 * Bp21 (2x2*2x2) -> C'11
    # A21 * Bp11 (2x2*2x2) -> C'21
    # A22 * Bp21 (2x2*2x2) -> C'21
    # A11 * Bp12 (2x2*2x1) -> C'12
    # A12 * Bp22 (2x2*2x1) -> C'12
    # A21 * Bp12 (2x2*2x1) -> C'22
    # A22 * Bp22 (2x2*2x1) -> C'22
    
    # Use Strassen for 2x2*2x2
    # Use Schoolbook for 2x2*2x1
    
    # A11 * Bp11
    add_block_complex(U_s, V_s, W_s, 
                      [(A11_map, 1)], 
                      [(Bp11_map, 1)], 
                      [(Cp11_map, 1)], col)
                      
    # A12 * Bp21
    add_block_complex(U_s, V_s, W_s, 
                      [(A12_map, 1)], 
                      [(Bp21_map, 1)], 
                      [(Cp11_map, 1)], col)
                      
    # A21 * Bp11
    add_block_complex(U_s, V_s, W_s, 
                      [(A21_map, 1)], 
                      [(Bp11_map, 1)], 
                      [(Cp21_map, 1)], col)
                      
    # A22 * Bp21
    add_block_complex(U_s, V_s, W_s, 
                      [(A22_map, 1)], 
                      [(Bp21_map, 1)], 
                      [(Cp21_map, 1)], col)
                      
    # A11 * Bp12 (2x2*2x1)
    U_sb, V_sb, W_sb = schoolbook_221()
    # Bp12 is 2x1. Local indices 0,1.
    # V_sb is (2, 4).
    # We need to map V_sb local indices to global Bp12 indices.
    # Bp12_map = [30, 37].
    # C'12 is 2x1. Local indices 0,1.
    # Cp12_map = [18, 22].
    
    # Modify add_block_complex to handle 2x1?
    # Or just write a specific one.
    
    def add_block_221(U_b, V_b, W_b, A_map, B_map, C_map, col_off):
        nonlocal col
        for r in range(U_b.shape[1]):
            for i_local in range(4):
                if U_b[i_local, r] != 0:
                    U[A_map[i_local], col_off + r] += U_b[i_local, r]
            for j_local in range(2):
                if V_b[j_local, r] != 0:
                    V[B_map[j_local], col_off + r] += V_b[j_local, r]
            for k_local in range(2):
                if W_b[k_local, r] != 0:
                    W[C_map[k_local], col_off + r] += W_b[k_local, r]
        col += U_b.shape[1]
        
    add_block_221(U_sb, V_sb, W_sb, A11_map, Bp12_map, Cp12_map, col)
    add_block_221(U_sb, V_sb, W_sb, A12_map, Bp22_map, Cp12_map, col)
    add_block_221(U_sb, V_sb, W_sb, A21_map, Bp12_map, Cp22_map, col)
    add_block_221(U_sb, V_sb, W_sb, A22_map, Bp22_map, Cp22_map, col)
    
    # Total rank should be 49 + 28 + 16 = 93.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
