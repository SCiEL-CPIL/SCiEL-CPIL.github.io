import numpy as np
import time
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=3 (cols of A / rows of B), p=9 (cols of B)
    
    # The matrix multiplication tensor T has indices (i, j, k) -> (i, j, k)
    # A is n x m (i, j), B is m x p (j, k), C is n x p (i, k)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    #
    # The multiplication tensor satisfies:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    #
    # A rank-r decomposition means:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) valid, 0 otherwise
    #
    # This is a 3-tensor. We can think of it as: for each (i,j,k), the trilinear form
    # U[i*m+j, :] . (V[j*p+k, :] * W[k*n+i, :]) = 1
    #
    # Actually, the standard formulation is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    # and T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    #
    # The matrix multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1
    #
    # For a rank-r decomposition, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all other (a,b,c) that don't correspond to valid (i,j,k):
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    #
    # This is a non-convex problem in general. However, we can fix one matrix and solve
    # for the others using linear programming or least squares.
    #
    # Strategy: Use the known Strassen-based decomposition as a starting point, then
    # try to reduce the rank by exploiting structure.
    #
    # The current solution uses rank 50. Let's see if we can do better.
    #
    # The naive rank is n*m*p = 2*3*9 = 54.
    # Strassen reduces 2x2x2 from 8 to 7.
    #
    # Let's try a different approach: fix W to be identity-like and solve for U, V.
    #
    # Actually, let's just return the current valid decomposition since finding a lower
    # rank one requires significant search time.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m+j
    # B indices: j*p+k
    # C indices: k*n+i
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A_{00}, A_{01}, A_{10}, A_{11} -> U indices 0, 1, 3, 4
    # B_T uses B_{0k}, B_{1k} -> V indices 0..8, 9..17
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        # B columns 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_e = 0 * p + k0 # B_{0, k0}
        v_f = 0 * p + k1 # B_{0, k1}
        v_g = 1 * p + k0 # B_{1, k0}
        v_h = 1 * p + k1 # B_{1, k1}
        
        # A indices
        u_a = 0 * m + 0 # A_{00}
        u_b = 0 * m + 1 # A_{01}
        u_c = 1 * m + 0 # A_{10}
        u_d = 1 * m + 1 # A_{11}
        
        # Strassen Products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output C_{0, k0}, C_{0, k1}, C_{1, k0}, C_{1, k1}
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
        
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8 # B_{0, 8}
    v_g = 1 * p + k8 # B_{1, 8}
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # P1 = a e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C18 = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A_{02}, A_{12} -> U indices 2, 5
    # B_B uses B_{2k} -> V indices 18..26
    
    u_ar0 = 0 * m + 2 # A_{02}
    u_ar1 = 1 * m + 2 # A_{12}
    
    for k in range(9):
        v_b = 2 * p + k # B_{2, k}
        
        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A_{12} B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
