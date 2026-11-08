import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1 = (x00 + x11) * (y00 + y11)
    # P2 = (x10 + x11) * y00
    # P3 = x00 * (y01 - y11)
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y11
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    
    # X indices: x00, x01, x10, x11
    # Y indices: y00, y01, y10, y11
    # Z indices: z00, z01, z10, z11
    
    strassen_products = [
        # (x_coeffs, y_coeffs, z_coeffs)
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 1}), # z00 += P1, z11 += P1
        # Wait, Z reconstruction:
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # Let's define contributions to Z for each Pk
        # Pk is a scalar.
        # We add coeff * Pk to Z_ij.
        # So for each Pk, we have a vector of 4 coeffs for Z00, Z01, Z10, Z11.
        
        # P1: Z00 += 1, Z11 += 1
        # P2: Z10 += 1, Z11 -= 1
        # P3: Z01 += 1, Z11 += 1
        # P4: Z00 += 1, Z10 += 1
        # P5: Z00 -= 1, Z01 += 1
        # P6: Z11 += 1
        # P7: Z00 += 1
        
    ]
    
    # Redefining clearly:
    # P1: (x00+x11)(y00+y11) -> x:{0:1, 3:1}, y:{0:1, 3:1}, z:{0:1, 3:1}
    # P2: (x10+x11)y00 -> x:{2:1, 3:1}, y:{0:1}, z:{2:1, 3:-1}
    # P3: x00(y01-y11) -> x:{0:1}, y:{1:1, 3:-1}, z:{1:1, 3:1}
    # P4: x11(y10-y00) -> x:{3:1}, y:{2:1, 0:-1}, z:{0:1, 2:1}
    # P5: (x00+x01)y11 -> x:{0:1, 1:1}, y:{3:1}, z:{0:-1, 1:1}
    # P6: (x10-x00)(y00+y01) -> x:{2:1, 0:-1}, y:{0:1, 1:1}, z:{3:1}
    # P7: (x01-x11)(y10+y11) -> x:{1:1, 3:-1}, y:{2:1, 3:1}, z:{0:1}
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # X indices in U:
    # x00 -> A[0, 2s] -> U[0*m + 2s] = U[2s]
    # x01 -> A[0, 2s+1] -> U[0*m + 2s+1] = U[2s+1]
    # x10 -> A[1, 2s] -> U[1*m + 2s] = U[4+2s]
    # x11 -> A[1, 2s+1] -> U[1*m + 2s+1] = U[5+2s]
    
    # Y indices in V:
    # y00 -> B[2s, 2t] -> V[2s*p + 2t] = V[16s + 2t]
    # y01 -> B[2s, 2t+1] -> V[2s*p + 2t+1] = V[16s + 2t+1]
    # y10 -> B[2s+1, 2t] -> V[(2s+1)*p + 2t] = V[16s + 8 + 2t]
    # y11 -> B[2s+1, 2t+1] -> V[(2s+1)*p + 2t+1] = V[16s + 8 + 2t+1]
    
    # Z indices in W:
    # z00 -> C[0, 2t] -> W[2t*n + 0] = W[4t]
    # z01 -> C[0, 2t+1] -> W[(2t+1)*n + 0] = W[4t+2]
    # z10 -> C[1, 2t] -> W[2t*n + 1] = W[4t+1]
    # z11 -> C[1, 2t+1] -> W[(2t+1)*n + 1] = W[4t+3]
    
    # Iterate over 4 output blocks t and 2 input blocks s
    # Total 8 blocks. Each uses 7 Strassen products.
    # Rank = 8 * 7 = 56.
    
    col_offset = 0
    for t in range(4): # Output block index 0..3
        for s in range(2): # Input block index 0..1
            for p_idx in range(7):
                r = col_offset
                x_coeffs, y_coeffs, z_coeffs = strassen_defs[p_idx]
                
                # Update U
                for x_idx, val in x_coeffs.items():
                    u_idx = [2*s, 2*s+1, 4+2*s, 5+2*s][x_idx]
                    U[u_idx][r] += val
                
                # Update V
                for y_idx, val in y_coeffs.items():
                    v_idx = [16*s + 2*t, 16*s + 2*t+1, 16*s + 8 + 2*t, 16*s + 8 + 2*t+1][y_idx]
                    V[v_idx][r] += val
                
                # Update W
                for z_idx, val in z_coeffs.items():
                    w_idx = [4*t, 4*t+2, 4*t+1, 4*t+3][z_idx]
                    W[w_idx][r] += val
                
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
