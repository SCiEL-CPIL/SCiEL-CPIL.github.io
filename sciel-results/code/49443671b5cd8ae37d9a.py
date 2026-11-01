"""Matrix multiplication 2,2,7 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 7]

import time
import random

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            jp = b // p
            k = b % p
            if j != jp:
                for c in range(p * n):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
            else:
                for c in range(p * n):
                    kp = c // n
                    ip = c % n
                    exp = 1.0 if (k == kp and i == ip) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - exp) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140

    # Schoolbook: rank = n*m*p = 28
    sb_rank = n * m * p
    U_sb = [[0] * sb_rank for _ in range(n * m)]
    V_sb = [[0] * sb_rank for _ in range(m * p)]
    W_sb = [[0] * sb_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": sb_rank}

    # Try to find lower rank via structured construction
    # For 2x2 x 2x7: C = A * B
    # A = [[a,b],[c,d]], B = 2x7, C = 2x7
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    # Try rank 14: use terms (i, j, k) but merge some
    # Actually, let's try a direct construction.
    
    # The tensor T[a,b,c] = 1 iff j=j' and k=k' and i=i'
    # where a=i*2+j, b=j'*7+k', c=k'*2+i'
    
    # Non-zero entries: for each (i,j,k) in {0,1}x{0,1}x{0..6}:
    #   a = i*2+j, b = j*7+k, c = k*2+i
    # So there are 2*2*7 = 28 non-zero entries, each = 1.
    
    # Rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^4, v_r in R^14, w_r in R^14
    
    # Each non-zero entry T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c] = 1
    # Each zero entry = 0
    
    # This is a system of equations. Let's try to solve it with rank r.
    
    # For rank 14: we have 14 * (4+14+14) = 14*32 = 448 unknowns
    # Constraints: 28 equations = 1, 4*14*14 - 28 = 784-28 = 756 equations = 0
    # Total: 784 equations, 448 unknowns. Overdetermined!
    
    # So rank 14 might not work. Let's check rank 16, 18, 20, etc.
    
    # Actually, the number of constraints is 784 (all entries), and unknowns is r*32.
    # For feasibility, we need r*32 >= 784, so r >= 24.5, i.e., r >= 25.
    # But this is a very loose bound since the tensor is highly structured.
    
    # Let me try a smarter approach: exploit the structure.
    
    # The tensor is: T[i,j; j',k'; k'',i''] = δ(j,j')δ(k,k')δ(i,i')
    # with a = i*m+j, b = j'*p+k', c = k''*n+i''
    
    # For fixed (i,j), the non-zero entries are at (b=j*p+k, c=k*n+i) for all k.
    # So for each (i,j), we have p=7 non-zero entries.
    
    # Let's try to use the structure to find a low-rank decomposition.
    
    # Approach: For each (i,j), the "slice" T[i,j, :, :] is a 14x14 matrix with 7 non-zero entries.
    # Specifically, T[i,j, j*p+k, k*n+i] = 1 for k=0..6.
    
    # This slice can be written as sum over k of e_{j*p+k} ⊗ e_{k*n+i}
    # which has rank 7 (since the 7 vectors e_{j*p+k} are distinct and e_{k*n+i} are distinct).
    
    # So each (i,j) slice has rank 7, and there are 4 slices.
    # The total tensor rank is at least max over slices = 7, but could be higher.
    
    # Let's try to construct a decomposition with rank 28 (schoolbook) and see if we can reduce.
    
    # For now, return schoolbook as the safe answer.
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

# EVOLVE-BLOCK-END
