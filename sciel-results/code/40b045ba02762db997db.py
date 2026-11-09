"""
Program to find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
Uses the recursive Strassen algorithm which yields a rank of 49.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices for the tensor decomposition.
    The decomposition is based on the recursive Strassen algorithm for 4x4 matrices.
    Rank is 49.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit

    n, m, p = DIMENSIONS
    
    # Rank 49
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices for A (4x4)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    A_blocks = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # Indices for B (4x4)
    B_blocks = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # Indices for C (4x4)
    # W index is k*n + i.
    # C is 4x4.
    # C11 block: i in 0,1; k in 0,1.
    #   (i=0,k=0) -> 0*4+0 = 0
    #   (i=1,k=0) -> 0*4+1 = 1
    #   (i=0,k=1) -> 1*4+0 = 4
    #   (i=1,k=1) -> 1*4+1 = 5
    # C12 block: i in 0,1; k in 2,3.
    #   (i=0,k=2) -> 2*4+0 = 8
    #   (i=1,k=2) -> 2*4+1 = 9
    #   (i=0,k=3) -> 3*4+0 = 12
    #   (i=1,k=3) -> 3*4+1 = 13
    # C21 block: i in 2,3; k in 0,1.
    #   (i=2,k=0) -> 0*4+2 = 2
    #   (i=3,k=0) -> 0*4+3 = 3
    #   (i=2,k=1) -> 1*4+2 = 6
    #   (i=3,k=1) -> 1*4+3 = 7
    # C22 block: i in 2,3; k in 2,3.
    #   (i=2,k=2) -> 2*4+2 = 10
    #   (i=3,k=2) -> 2*4+3 = 11
    #   (i=2,k=3) -> 3*4+2 = 14
    #   (i=3,k=3) -> 3*4+3 = 15
    
    C_blocks = {
        (0,0): [0, 1, 4, 5],
        (0,1): [8, 9, 12, 13],
        (1,0): [2, 3, 6, 7],
        (1,1): [10, 11, 14, 15]
    }
    
    # Strassen steps for 2x2 blocks
    # M_r = X_r Y_r
    # X_r, Y_r are 2x2 matrices formed from blocks of A, B.
    # We define the coefficients for X_r[u,v] and Y_r[u,v] in terms of A_blocks and B_blocks.
    
    # Step 0: M1 = (A11+A22)(B11+B22)
    # Step 1: M2 = (A21+A22)B11
    # Step 2: M3 = A11(B12-B22)
    # Step 3: M4 = A22(B21-B11)
    # Step 4: M5 = (A11+A12)B22
    # Step 5: M6 = (A21-A11)(B11+B12)
    # Step 6: M7 = (A12-A22)(B21+B22)
    
    # For each step r, we define:
    # X_coeffs: dict mapping (u,v) to list of (block_idx, coeff)
    # Y_coeffs: dict mapping (u,v) to list of (block_idx, coeff)
    # C_coeffs: dict mapping (u,v) of M_r to coeff in C
    # Note: M_r is added to C.
    # C11 = M1 - M2 + M4 + M5
    # C12 = M3 + M2 - M5 - M6
    # C21 = M1 + M4
    # C22 = M3 - M1 + M5 + M6
    
    # We will iterate r from 0 to 6.
    # For each r, we decompose M_r = X_r Y_r into 7 scalar multiplications.
    # The 7 scalar multiplications for 2x2 matmul X Y:
    # P0 = (X00+X11)(Y00+Y11)
    # P1 = (X10+X11)Y00
    # P2 = X00(Y01-Y11)
    # P3 = X11(Y10-Y00)
    # P4 = (X00+X01)Y11
    # P5 = (X10-X00)(Y00+Y01)
    # P6 = (X01-X11)(Y10+Y11)
    
    # Z00 = P0 - P1 + P3 + P4
    # Z01 = P2 + P1 - P4 - P5
    # Z10 = P0 + P3
    # Z11 = P2 - P0 + P4 + P5
    
    # For each scalar product P_s, we have linear forms for X and Y elements.
    # We need to combine these with X_coeffs and Y_coeffs to get linear forms on A and B.
    
    strassen_steps = [
        { # M1
            'X': {
                (0,0): [((0,0), 1), ((1,1), 1)],
                (0,1): [((0,0), 1), ((1,1), 1)],
                (1,0): [((0,0), 1), ((1,1), 1)],
                (1,1): [((0,0), 1), ((1,1), 1)]
            },
            'Y': {
                (0,0): [((0,0), 1), ((1,1), 1)],
                (0,1): [((0,0), 1), ((1,1), 1)],
                (1,0): [((0,0), 1), ((1,1), 1)],
                (1,1): [((0,0), 1), ((1,1), 1)]
            },
            'C': {
                (0,0): 1, # C11 += M1
                (1,0): 1  # C21 += M1
            }
        },
        { # M2
            'X': {
                (0,0): [((1,0), 1), ((1,1), 1)],
                (0,1): [((1,0), 1), ((1,1), 1)],
                (1,0): [((1,0), 1), ((1,1), 1)],
                (1,1): [((1,0), 1), ((1,1), 1)]
            },
            'Y': {
                (0,0): [((0,0), 1)],
                (0,1): [((0,0), 1)],
                (1,0): [((0,0), 1)],
                (1,1): [((0,0), 1)]
            },
            'C': {
                (0,0): -1, # C11 -= M2
                (0,1): 1    # C12 += M2
            }
        },
        { # M3
            'X': {
                (0,0): [((0,0), 1)],
                (0,1): [((0,0), 1)],
                (1,0): [((0,0), 1)],
                (1,1): [((0,0), 1)]
            },
            'Y': {
                (0,0): [((0,1), 1), ((1,1), -1)],
                (0,1): [((0,1), 1), ((1,1), -1)],
                (1,0): [((0,1), 1), ((1,1), -1)],
                (1,1): [((0,1), 1), ((1,1), -1)]
            },
            'C': {
                (0,1): 1,  # C12 += M3
                (1,1): 1   # C22 += M3
            }
        },
        { # M4
            'X': {
                (0,0): [((1,1), 1)],
                (0,1): [((1,1), 1)],
                (1,0): [((1,1), 1)],
                (1,1): [((1,1), 1)]
            },
            'Y': {
                (0,0): [((1,0), 1), ((0,0), -1)],
                (0,1): [((1,0), 1), ((0,0), -1)],
                (1,0): [((1,0), 1), ((0,0), -1)],
                (1,1): [((1,0), 1), ((0,0), -1)]
            },
            'C': {
                (0,0): 1,  # C11 += M4
                (1,0): 1   # C21 += M4
            }
        },
        { # M5
            'X': {
                (0,0): [((0,0), 1), ((0,1), 1)],
                (0,1): [((0,0), 1), ((0,1), 1)],
                (1,0): [((0,0), 1), ((0,1), 1)],
                (1,1): [((0,0), 1), ((0,1), 1)]
            },
            'Y': {
                (0,0): [((1,1), 1)],
                (0,1): [((1,1), 1)],
                (1,0): [((1,1), 1)],
                (1,1): [((1,1), 1)]
            },
            'C': {
                (0,0): 1,   # C11 += M5
                (0,1): -1,  # C12 -= M5
                (1,1): 1    # C22 += M5
            }
        },
        { # M6
            'X': {
                (0,0): [((1,0), 1), ((0,0), -1)],
                (0,1): [((1,0), 1), ((0,0), -1)],
                (1,0): [((1,0), 1), ((0,0), -1)],
                (1,1): [((1,0), 1), ((0,0), -1)]
            },
            'Y': {
                (0,0): [((0,0), 1), ((0,1), 1)],
                (0,1): [((0,0), 1), ((0,1), 1)],
                (1,0): [((0,0), 1), ((0,1), 1)],
                (1,1): [((0,0), 1), ((0,1), 1)]
            },
            'C': {
                (0,1): -1   # C12 -= M6
            }
        },
        { # M7
            'X': {
                (0,0): [((0,1), 1), ((1,1), -1)],
                (0,1): [((0,1), 1), ((1,1), -1)],
                (1,0): [((0,1), 1), ((1,1), -1)],
                (1,1): [((0,1), 1), ((1,1), -1)]
            },
            'Y': {
                (0,0): [((1,0), 1), ((1,1), 1)],
                (0,1): [((1,0), 1), ((1,1), 1)],
                (1,0): [((1,0), 1), ((1,1), 1)],
                (1,1): [((1,0), 1), ((1,1), 1)]
            },
            'C': {} # M7 is not used in the standard Strassen output formulas?
                    # Wait, let me double check the M7 usage.
                    # C11 = M1 - M2 + M4 + M5
                    # C12 = M3 + M2 - M5 - M6
                    # C21 = M1 + M4
                    # C22 = M3 - M1 + M5 + M6
                    # M7 is indeed not in the sum.
                    # However, M7 is part of the 7 multiplications.
                    # If M7 is not used in the output, its contribution to W is 0.
                    # This is correct for Strassen.
        }
    ]
    
    # Scalar Strassen steps for 2x2
    # P0: X: 00+11, Y: 00+11
    # P1: X: 10+11, Y: 00
    # P2: X: 00, Y: 01-11
    # P3: X: 11, Y: 10-00
    # P4: X: 00+01, Y: 11
    # P5: X: 10-00, Y: 00+01
    # P6: X: 01-11, Y: 10+11
    
    scalar_steps = [
        { 'X': { (0,0): [(0,0, 1), (1,1, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Y': { (0,0): [(0,0, 1), (1,1, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Z': { (0,0): 1, (0,1): 0, (1,0): 1, (1,1): -1 } }, # Z00+=1, Z10+=1, Z11-=1
        { 'X': { (0,0): [], (0,1): [], (1,0): [(1,0, 1), (1,1, 1)], (1,1): [] },
          'Y': { (0,0): [(0,0, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Z': { (0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0 } }, # Z00-=1, Z01+=1
        { 'X': { (0,0): [(0,0, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Y': { (0,0): [], (0,1): [(0,1, 1), (1,1, -1)], (1,0): [], (1,1): [] },
          'Z': { (0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1 } }, # Z01+=1, Z11+=1
        { 'X': { (0,0): [], (0,1): [], (1,0): [], (1,1): [(1,1, 1)] },
          'Y': { (0,0): [(0,0, -1)], (0,1): [], (1,0): [(1,0, 1)], (1,1): [] },
          'Z': { (0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0 } }, # Z00+=1, Z10+=1
        { 'X': { (0,0): [(0,0, 1), (0,1, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Y': { (0,0): [], (0,1): [], (1,0): [], (1,1): [(1,1, 1)] },
          'Z': { (0,0): 1, (0,1): -1, (1,0): 0, (1,1): 1 } }, # Z00+=1, Z01-=1, Z11+=1
        { 'X': { (0,0): [(0,0, -1)], (0,1): [], (1,0): [(1,0, 1)], (1,1): [] },
          'Y': { (0,0): [(0,0, 1), (0,1, 1)], (0,1): [], (1,0): [], (1,1): [] },
          'Z': { (0,0): 0, (0,1): -1, (1,0): 0, (1,1): 0 } }, # Z01-=1
        { 'X': { (0,0): [], (0,1): [(0,1, 1), (1,1, -1)], (1,0): [], (1,1): [] },
          'Y': { (0,0): [], (0,1): [], (1,0): [(1,0, 1), (1,1, 1)], (1,1): [] },
          'Z': { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0 } }  # M7 is not used in Z
    ]
    
    # Wait, my Z coefficients for scalar steps must match:
    # Z00 = P0 - P1 + P3 + P4
    # Z01 = P2 + P1 - P4 - P5
    # Z10 = P0 + P3
    # Z11 = P2 - P0 + P4 + P5
    
    # Let's re-verify scalar step Z coeffs:
    # P0: Z00+1, Z10+1, Z11-1. (Correct: P0 in Z00, Z10, -P0 in Z11)
    # P1: Z00-1, Z01+1. (Correct: -P1 in Z00, +P1 in Z01)
    # P2: Z01+1, Z11+1. (Correct: +P2 in Z01, +P2 in Z11)
    # P3: Z00+1, Z10+1. (Correct: +P3 in Z00, +P3 in Z10)
    # P4: Z00+1, Z01-1, Z11+1. (Correct: +P4 in Z00, -P4 in Z01, +P4 in Z11)
    # P5: Z01-1. (Correct: -P5 in Z01)
    # P6: 0. (Correct: P6 not in formulas)
    
    # Now construct U, V, W
    for r in range(7):
        step_r = strassen_steps[r]
        for s in range(7):
            step_s = scalar_steps[s]
            idx = r * 7 + s
            
            # Construct U vector (size 16)
            U_vec = [0.0] * 16
            # For each element (u,v) of X_r, we have linear forms from step_s['X']
            # We need to combine with step_r['X']
            
            # Iterate over X elements involved in P_s
            # step_s['X'] maps (u,v) to list of (x_elem, coeff)
            # Actually, step_s['X'] maps (u,v) to list of (x_elem, coeff) where x_elem is (u,v) of X?
            # No, my scalar_steps definition is a bit mixed.
            # Let's re-define scalar_steps clearly.
            # P_s = sum_{u,v} c_{u,v} X_{u,v} * sum_{u',v'} d_{u',v'} Y_{u',v'}
            
            # My current scalar_steps structure:
            # 'X': { (u,v): [(x_elem, coeff)] } -- This is wrong.
            # It should be: For each (u,v) in X, what is the coefficient in P_s?
            # P0 = (X00+X11)(Y00+Y11).
            # Coeff of X00 is 1. Coeff of X11 is 1.
            
            # Let's use a simpler representation for scalar steps.
            # x_coeffs: dict (u,v) -> coeff
            # y_coeffs: dict (u,v) -> coeff
            
            # I will redefine the scalar steps loop to be cleaner.
            pass # Placeholder, will rewrite logic below
            
    # Rewrite logic for clarity and correctness
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define scalar steps with explicit coeffs for X and Y elements
    # X elements: 00, 01, 10, 11
    # Y elements: 00, 01, 10, 11
    
    scalar_defs = [
        # P0
        {'x': {(0,0): 1, (1,1): 1}, 'y': {(0,0): 1, (1,1): 1}, 'z': {(0,0): 1, (1,0): 1, (1,1): -1}},
        # P1
        {'x': {(1,0): 1, (1,1): 1}, 'y': {(0,0): 1}, 'z': {(0,0): -1, (0,1): 1}},
        # P2
        {'x': {(0,0): 1}, 'y': {(0,1): 1, (1,1): -1}, 'z': {(0,1): 1, (1,1): 1}},
        # P3
        {'x': {(1,1): 1}, 'y': {(1,0): 1, (0,0): -1}, 'z': {(0,0): 1, (1,0): 1}},
        # P4
        {'x': {(0,0): 1, (0,1): 1}, 'y': {(1,1): 1}, 'z': {(0,0): 1, (0,1): -1, (1,1): 1}},
        # P5
        {'x': {(1,0): 1, (0,0): -1}, 'y': {(0,0): 1, (0,1): 1}, 'z': {(0,1): -1}},
        # P6
        {'x': {(0,1): 1, (1,1): -1}, 'y': {(1,0): 1, (1,1): 1}, 'z': {}}
    ]
    
    for r in range(7):
        step_r = strassen_steps[r]
        for s in range(7):
            step_s = scalar_defs[s]
            idx = r * 7 + s
            
            # Build U (A coeffs)
            # For each (u,v) in X, get coeff from step_s['x']
            # For each (u,v) in X, get block coeffs from step_r['X']
            # Combine: U_vec[idx_A] += coeff_s * coeff_r
            
            for (u, v), coeff_s in step_s['x'].items():
                if coeff_s == 0: continue
                # Get block coeffs for X[u,v] from step_r
                if (u, v) not in step_r['X']: continue
                for block_idx, coeff_r in step_r['X'][(u, v)]:
                    # Get A indices for this block
                    for a_idx in A_blocks[block_idx]:
                        U[a_idx, idx] += coeff_s * coeff_r
                        
            # Build V (B coeffs)
            for (u, v), coeff_s in step_s['y'].items():
                if coeff_s == 0: continue
                if (u, v) not in step_r['Y']: continue
                for block_idx, coeff_r in step_r['Y'][(u, v)]:
                    for b_idx in B_blocks[block_idx]:
                        V[b_idx, idx] += coeff_s * coeff_r
                        
            # Build W (C coeffs)
            # The product P_s contributes to Z[u,v] with coeff from step_s['z']
            # Z[u,v] is an element of M_r.
            # M_r contributes to C with coeff from step_r['C']
            
            for (u, v), coeff_s in step_s['z'].items():
                if coeff_s == 0: continue
                # Get C block coeffs for M_r[u,v]
                if (u, v) not in step_r['C']: continue
                coeff_r = step_r['C'][(u, v)]
                # Get C indices for this block
                for c_idx in C_blocks[(u, v)]:
                    W[c_idx, idx] += coeff_s * coeff_r
                    
    if time.time() > deadline:
        # Fallback to schoolbook if it takes too long (unlikely)
        rank = 64
        U = np.zeros((16, 64))
        V = np.zeros((16, 64))
        W = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
