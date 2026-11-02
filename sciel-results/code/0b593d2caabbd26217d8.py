import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy: Decompose 3x5x5 into 2x5x5 + 1x5x5
    # For 2x5x5: use 2 disjoint 2x2x2 Strassen blocks (saves 2 rank terms)
    #   2*5*5 = 50 schoolbook, with 2 Strassen blocks: 50 - 2 = 48
    # For 1x5x5: schoolbook rank 5
    # Total: 48 + 5 = 53
    
    total_rank = 2 * 7 + (2 * m * p - 2 * 8) + (1 * m * p)
    # = 14 + (50 - 16) + 25 = 14 + 34 + 25 = 73... wait that's wrong
    
    # Let me recount:
    # 2x5x5 has 2*5*5 = 50 terms
    # 2 Strassen blocks each replace 8 terms with 7, so save 1 each
    # 50 - 2 = 48 terms for the 2x5x5 part
    # 1x5x5 has 1*5*5 = 25 terms
    # Total = 48 + 25 = 73... that's worse than 71!
    
    # Wait, I made an error. Let me reconsider.
    # In the original 3x5x5, we had 4 Strassen blocks saving 4 terms: 75 - 4 = 71
    # Now with 2x5x5 + 1x5x5:
    # 2x5x5: can we fit more Strassen blocks? 
    #   i in {0,1}, so we can have blocks with j in {0,1} and {2,3}, k in {0,1} and {2,3}
    #   That's 4 blocks: (i={0,1}, j={0,1}, k={0,1}), (i={0,1}, j={2,3}, k={0,1}),
    #   (i={0,1}, j={0,1}, k={2,3}), (i={0,1}, j={2,3}, k={2,3})
    #   All 4 are disjoint! So 50 - 4 = 46 for the 2x5x5 part
    # 1x5x5: 25 terms (schoolbook)
    # Total: 46 + 25 = 71... same as before!
    
    # Hmm, so splitting doesn't help if we can already fit 4 blocks in the 3x5x5.
    # The issue is that the 3rd row (i=2) can't participate in any 2x2x2 Strassen block.
    
    # Let me think differently. Can I use a different decomposition for the 1x5x5 part?
    # A 1x5x5 tensor is just a vector of 25 ones. It needs rank 25 (each term is independent).
    # Unless... can I share factors with the 2x5x5 part?
    
    # Actually, the key insight from the directive is to use a rank-19 decomposition for 5x5.
    # But that's for 5x5x5, not for 2x5x5.
    
    # Let me just keep the current approach but verify it's correct.
    # The current program achieves rank 71 with 4 Strassen blocks.
    
    # Actually, let me try a different structural approach:
    # Use 2x2x2 Strassen for blocks where possible, and look for any additional savings.
    
    # The 4 blocks in the current solution are:
    # (0,1, 0,1, 0,1), (0,1, 2,3, 0,1), (0,1, 0,1, 2,3), (0,1, 2,3, 2,3)
    # These cover i=0,1; j=0,1,2,3; k=0,1,2,3
    # Remaining: i=2 (all j,k) and i=0,1 with j=4 or k=4
    
    # Can I add more Strassen-like savings? Not with 2x2x2 blocks since i=2 is single.
    
    # Let me just implement the current 71-rank solution cleanly and verify it works.
    
    total_rank = 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1 = (a00 + a11)(b00 + b11)
        c = col
        U[i0*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        c = col
        U[i1*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + i1][c] += 1
        W[k1*n + i1][c] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        c = col
        U[i1*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + i1][c] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += -1
        W[k1*n + i0][c] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c = col
        U[i1*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        c = col
        U[i0*m + j1][c] += 1
        U[i1*m + j1][c] += -1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1
        col += 1
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i0, i1, j0, j1, k0, k1) in blocks:
                    if i0 <= i <= i1 and j0 <= j <= j1 and k0 <= k <= k1:
                        in_block = True
                        break
                if in_block:
                    continue
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = 0.0
                for c in range(U.shape[1]):
                    T_reconstructed[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")
    
    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")
