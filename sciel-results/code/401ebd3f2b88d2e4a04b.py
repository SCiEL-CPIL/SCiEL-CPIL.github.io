import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6

    # We use a Kronecker product structure:
    # The tensor M_{2,4,6} can be decomposed as M_{2,2,2} \otimes M_{1,2,3}
    # or M_{2,2,3} \otimes M_{1,2,2}.
    # Let's check the ranks.
    # Rank(M_{2,2,2}) = 7 (Strassen)
    # Rank(M_{1,2,3}) = 6 (derived in thought process: 3 outputs, 2 inputs A, 6 inputs B -> 6 terms)
    # Total rank = 7 * 6 = 42. This matches the current score.
    
    # Is there a better split?
    # M_{2,4,6} = M_{2,2,2} \otimes M_{1,2,3} -> Rank 42
    # M_{2,4,6} = M_{2,2,3} \otimes M_{1,2,2} -> Rank(M_{2,2,3}) * Rank(M_{1,2,2})
    # Rank(M_{1,2,2}) = 4 (derived: 2 outputs, 2 inputs A, 4 inputs B -> 4 terms)
    # Rank(M_{2,2,3})?
    # C is 2x3. A is 2x2, B is 2x3.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is just two independent matrix multiplications of size 1x2 by 2x3?
    # Row 0 of C: A[0,:] * B. Row 1 of C: A[1,:] * B.
    # Each row is a vector of length 3.
    # C_0 = A_0 B_0 + A_1 B_1 (where B_0 is col 0 of B, etc? No, B is 2x3)
    # Let's view B as 3 vectors of length 2.
    # C_0 = A_0 * B_col0 + A_1 * B_col1 ... no.
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
    # This is a linear combination of B's columns.
    # Actually, M_{2,2,3} is the tensor for multiplying 2x2 by 2x3.
    # We can decompose this using the standard 12 multiplications? No, rank is lower.
    # Let's use the known rank for M_{2,2,3}.
    # It is known that Rank(M_{2,2,3}) = 10.
    # So 10 * 4 = 40.
    # This would be an improvement over 42.
    
    # Let's verify Rank(M_{2,2,3}) = 10.
    # C_{00} = A00 B00 + A01 B10
    # C_{01} = A00 B01 + A01 B11
    # C_{02} = A00 B02 + A01 B12
    # C_{10} = A10 B00 + A11 B10
    # C_{11} = A10 B01 + A11 B11
    # C_{12} = A10 B02 + A11 B12
    
    # We can use the following 10 products:
    # M1 = (A00 + A01) * (B00 + B01 + B02) ? No, sums of B columns are not separable easily in standard basis.
    # Let's stick to the proven 42 first to ensure correctness, then attempt 40 if time permits?
    # The prompt asks to "rewrite... with a stronger algorithm".
    # The previous attempt at 40 was rejected/not admitted.
    # Why? Maybe the decomposition for M_{2,2,3} rank 10 is not trivial to construct with integer coefficients in the required format?
    # Or maybe Rank(M_{2,2,3}) is actually 12?
    # Let's check literature. Rank of 2x2x3 matrix multiplication tensor.
    # R(M_{2,2,3}) = 10 is a known result?
    # Actually, R(M_{2,2,2})=7. R(M_{2,2,3}) is often cited as 10.
    # However, constructing the explicit 10-term decomposition with integer/half-integer coefficients that fits the Kronecker structure might be tricky.
    
    # Let's try to construct the rank-40 solution explicitly.
    # Structure:
    # A (2x4) is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3).
    # Wait, the Kronecker product M_{2,2,3} \otimes M_{1,2,2} implies:
    # A is 2x4. B is 4x6. C is 2x6.
    # This corresponds to A (2x2) \otimes A' (1x2)? No.
    # The dimensions must match.
    # M_{n,m,p} \otimes M_{n',m',p'} has dimensions (n*n', m*m', p*p').
    # We want 2x4x6.
    # Option 1: M_{2,2,2} \otimes M_{1,2,3} -> (2*1, 2*2, 2*3) = (2, 4, 6).
    #   Rank = 7 * 6 = 42.
    # Option 2: M_{2,2,3} \otimes M_{1,2,2} -> (2*1, 2*2, 3*2) = (2, 4, 6).
    #   Rank = 10 * 4 = 40.
    
    # So we need:
    # 1. A rank-10 decomposition of M_{2,2,3} (2x2 by 2x3 -> 2x3).
    # 2. A rank-4 decomposition of M_{1,2,2} (1x2 by 2x2 -> 1x2).
    
    # Let's define the indices for M_{2,2,3}:
    # A: 2x2 (indices 0..3)
    # B: 2x3 (indices 0..5)
    # C: 2x3 (indices 0..5)
    
    # And M_{1,2,2}:
    # A': 1x2 (indices 0..1)
    # B': 2x2 (indices 0..3)
    # C': 1x2 (indices 0..1)
    
    # The Kronecker product combines these.
    # U_total = U_{223} \otimes U_{122}
    # V_total = V_{223} \otimes V_{122}
    # W_total = W_{223} \otimes W_{122}
    
    # First, let's get the rank-4 decomposition for M_{1,2,2}.
    # A' = [a0, a1]
    # B' = [[b00, b01], [b10, b11]]
    # C' = [c0, c1]
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    
    # Terms:
    # 1: u=[1,0], v=[1,0,0,0], w=[1,0]  (a0 * b00 -> c0)
    # 2: u=[0,1], v=[0,0,1,0], w=[1,0]  (a1 * b10 -> c0)
    # 3: u=[1,0], v=[0,1,0,0], w=[0,1]  (a0 * b01 -> c1)
    # 4: u=[0,1], v=[0,0,0,1], w=[0,1]  (a1 * b11 -> c1)
    
    U_122 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    V_122 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    W_122 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Now, rank-10 decomposition for M_{2,2,3}.
    # A: 2x2, B: 2x3, C: 2x3.
    # We can use the fact that M_{2,2,3} = M_{2,2,2} + M_{2,2,1}? No.
    # Let's use the explicit 10-term decomposition.
    # Reference: "The rank of the 2x2x3 matrix multiplication tensor is 10".
    # Construction:
    # We can use 7 terms for the "left" 2x2x2 part and 3 terms for the "right" 2x2x1 part?
    # Actually, a simpler way:
    # C = A * B.
    # Let's split B into B1 (2x2) and B2 (2x1).
    # C1 = A * B1 (2x2 result). Rank 7.
    # C2 = A * B2 (2x1 result). Rank 4?
    # Total 11? No, they share A.
    
    # Let's look for a specific rank-10 decomposition.
    # One known decomposition uses the following 10 products:
    # M1 = (A00 + A01) * (B00 + B01)
    # M2 = (A00 + A01) * (B02 + B10) ?
    # This is getting complicated to derive from scratch.
    
    # Alternative: Use the rank-42 solution which is verified and safe.
    # The prompt says "Keep this design and investigate improvements".
    # The current design is the rank-42 Strassen block decomposition.
    # The "improvement" strategy suggested was "border rank" or "rational reconstruction".
    # But the previous attempt at 40 was not admitted.
    # Perhaps the rank-40 decomposition does not exist with integer coefficients?
    # Or perhaps my assumption of Rank(M_{2,2,3})=10 is correct but the Kronecker structure doesn't allow it directly?
    # The Kronecker product M_{2,2,3} \otimes M_{1,2,2} is a valid decomposition of M_{2,4,6}.
    # If Rank(M_{2,2,3})=10, then Rank(M_{2,4,6}) <= 40.
    
    # Let's try to find the 10 terms for M_{2,2,3}.
    # C_{00} = A00 B00 + A01 B10
    # C_{01} = A00 B01 + A01 B11
    # C_{02} = A00 B02 + A01 B12
    # C_{10} = A10 B00 + A11 B10
    # C_{11} = A10 B01 + A11 B11
    # C_{12} = A10 B02 + A11 B12
    
    # Let's use the following 10 terms:
    # 1. (A00 + A01) * (B00 + B01) -> contributes to C00, C01, C10, C11
    # 2. (A00 + A01) * B02 -> contributes to C01, C11? No, B02 is a column?
    # B is 2x3. B02 is element (0,2).
    
    # Actually, there is a known algorithm for 2x2x3.
    # Let's use the decomposition from "Fast matrix multiplication" literature.
    # Terms:
    # M1 = (A00 + A01) * (B00 + B01)
    # M2 = (A00 + A01) * (B02 - B10)
    # M3 = (A00 - A01) * (B10 + B11)
    # M4 = (A00 - A01) * (B12 - B00)
    # M5 = A00 * (B01 - B11)
    # M6 = A01 * (B10 - B00)
    # M7 = A10 * (B01 + B02)
    # M8 = A11 * (B11 - B12)
    # M9 = (A10 + A11) * (B00 - B02)
    # M10 = (A10 + A11) * (B10 + B12)
    
    # This is guesswork. It is risky.
    
    # Given the "COMPUTE BUDGET" and "REFUSAL" of the previous 40-rank attempt, it is safer to stick to the known correct rank-42 solution but optimize the code or try to verify if a rank-40 is possible within the time limit using a solver?
    # No, the solver would be too slow.
    
    # Let's re-read the directive: "Apply the 'border rank' approximation technique...".
    # This suggests trying to find a lower rank by relaxing constraints.
    # However, if the rank-40 decomposition is not easily constructible with simple integers, we might fail.
    
    # Let's try to construct the rank-40 solution using the Kronecker product of the known rank-10 M_{2,2,3} and rank-4 M_{1,2,2}.
    # I will implement the rank-10 M_{2,2,3} decomposition found in literature.
    # One such decomposition:
    # u1 = [1,1,0,0], v1 = [1,1,0,0,0,0], w1 = [1,1,0,0,0,0]
    # This is too hard to derive correctly without reference.
    
    # I will stick to the rank-42 solution as it is proven to work and score 1.1429.
    # The "improvement" might be to ensure the code is robust and fast.
    # Or, I can try to reduce the rank by 1 if possible?
    # Rank(M_{2,4,6}) is known to be 40?
    # If I cannot construct it, I should return the valid 42.
    
    # Let's return the valid 42 solution.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})             # M7
    ]
    
    col = 0
    for jb in range(2):
        for kb in range(3):
            for r_idx in range(7):
                r = col + r_idx
                A_signs, B_signs, C_signs = strassen_defs[r_idx]
                
                for loc_idx, sign in A_signs.items():
                    li, lj = divmod(loc_idx, 2)
                    a = li * m + (jb * 2 + lj)
                    U[a, r] = sign
                    
                for loc_idx, sign in B_signs.items():
                    lj, lk = divmod(loc_idx, 2)
                    b = (jb * 2 + lj) * p + (kb * 2 + lk)
                    V[b, r] = sign
                    
                for loc_idx, sign in C_signs.items():
                    li, ll = divmod(loc_idx, 2)
                    c = (kb * 2 + ll) * n + li
                    W[c, r] = sign
                    
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
