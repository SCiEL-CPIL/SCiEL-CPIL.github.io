import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We decompose M_{4,4,6} = M_{4,4,3} + M_{4,4,3}
    # Each M_{4,4,3} uses a rank decomposition.
    
    # For M_{4,4,3}:
    # We can use a decomposition of rank 26 or so?
    # Actually, let's stick to the previous approach but optimize it.
    
    # The previous approach had rank 77.
    # Let's try to find a better decomposition for M_{4,4,6}.
    
    # One known result: Rank(M_{4,4,6}) <= 77 (from the current program).
    # Can we do better?
    # 
    # Let's think about M_{4,4,6}.
    # We can write this as M_{4,4,4} + M_{4,4,2}.
    # Rank(M_{4,4,4}) = 49 (Strassen).
    # Rank(M_{4,4,2}) <= 8 (schoolbook, but can it be better?).
    # For M_{4,4,2}, C_{ij} = sum_{k=1}^4 A_{ik} B_{kj}.
    # This is 4 * 2 = 8 independent products? No, they share A.
    # Actually, M_{n,m,p} has rank at most n*m*p.
    # For M_{4,4,2}, that's 32.
    # But we can do better.
    # Note that M_{4,4,2} is isomorphic to M_{2,4,4} by transposing B.
    # Rank(M_{2,4,4})?
    # 
    # Let's try a different approach.
    # Use the fact that 6 = 3 + 3.
    # M_{4,4,6} = M_{4,4,3} + M_{4,4,3}.
    # What is the rank of M_{4,4,3}?
    # 
    # Actually, let's just keep the rank 77 approach but ensure it's correct.
    # The current program already achieves 1.2468, which corresponds to rank 77 (since seed is 96, 96/77 ≈ 1.2468).
    # 
    # To improve, we need to find a rank < 77.
    # 
    # Let's try M_{4,4,6} = M_{4,4,4} + M_{4,4,2}.
    # Rank(M_{4,4,4}) = 49.
    # For M_{4,4,2}, we can use a decomposition of rank 10?
    # If we can get M_{4,4,2} with rank 10, total is 59.
    # 
    # Actually, Rank(M_{2,4,4}) can be found.
    # M_{2,4,4}: A is 2x4, B is 4x4, C is 2x4.
    # C_{i,j} = sum_{k=1}^4 A_{i,k} B_{k,j}.
    # This is equivalent to M_{4,4,2} by symmetry (rank is invariant under transposition of the three tensors).
    # 
    # Let's use a known result: Rank(M_{2,4,4}) <= 10.
    # If so, total rank = 49 + 10 = 59.
    # 
    # Let's implement M_{4,4,4} with Strassen (rank 49) and M_{4,4,2} with a rank-10 decomposition.
    
    # First, let's build M_{4,4,4} with Strassen.
    R1 = 49
    U1 = np.zeros((16, R1), dtype=int)
    V1 = np.zeros((24, R1), dtype=int)
    W1 = np.zeros((16, R1), dtype=int)  # Wait, W for M_{4,4,4} is (4*4, 49) = (16, 49)
    
    # Strassen coefficients for 2x2x2
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
    
    # Build M_{4,4,4} using two levels of Strassen.
    # Top level: 4 blocks of 2x2 in A, 4 blocks of 2x2 in B (but B is 4x4, so 4 blocks of 2x2).
    # Actually, for M_{4,4,4}, we can use a 2-level Strassen.
    
    col = 0
    for s1 in range(7):
        for s2 in range(7):
            for a in range(4):
                if alpha[s1][a] != 0 and alpha[s2][a] != 0:
                    r1, c1 = a // 2, a % 2
                    # This is getting complicated. Let's use the existing approach from the current program
                    # but adapted for M_{4,4,4}.
            
            # Actually, let's just use the current program's approach for M_{4,4,6} directly.
            # The current program achieves rank 77.
            # To improve, we need a better decomposition.
    
    # Let's try a different strategy: use the fact that we can decompose M_{4,4,6} into
    # M_{4,4,3} + M_{4,4,3} and find a good rank for M_{4,4,3}.
    
    # For now, let's just return the same as the current program to ensure correctness.
    # The task is to improve, but if we can't find a better one quickly, we should keep the working solution.
    
    # Let's implement a rank-59 solution if possible.
    # M_{4,4,4} with rank 49, M_{4,4,2} with rank 10.
    
    # For M_{4,4,2}:
    # We need to multiply A (4x4) by B (4x2).
    # C (4x2) = A * B.
    # C_{i,j} = sum_{k=1}^4 A_{i,k} B_{k,j} for j=0,1.
    
    # A rank-10 decomposition for M_{4,4,2} would be ideal.
    # This is equivalent to M_{2,4,4}.
    
    # Let's use a simple schoolbook for M_{4,4,2} with rank 8 for now, giving total 49+8=57.
    # That's better than 77.
    
    R_total = 57
    U = np.zeros((16, R_total), dtype=int)
    V = np.zeros((24, R_total), dtype=int)
    W = np.zeros((24, R_total), dtype=int)
    
    # Part 1: M_{4,4,4} for the first 4 columns of B, using Strassen (rank 49)
    # This computes C[:, 0:4] = A * B[:, 0:4]
    
    col = 0
    # Two-level Strassen for M_{4,4,4}
    for s1 in range(7):
        for s2 in range(7):
            # U part
            for a in range(4):
                if alpha[s1][a] != 0 and alpha[s2][a] != 0:
                    r1, c1 = a // 2, a % 2
                    # The index in A: (r1*2 + something, c1*2 + something)
                    # This is getting complex. Let's use a simpler approach.
            
            # Actually, let's just use the current program's logic but for the 4x4x4 part.
    
    # Given the complexity, let's just return the current solution to ensure it works.
    # The current solution has rank 77 and scores 1.2468.
    
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    
    # Top-level Strassen coefficients
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
    
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if inner_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
