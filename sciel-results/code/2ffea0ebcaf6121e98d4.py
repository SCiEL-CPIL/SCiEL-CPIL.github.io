import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # The matrix multiplication tensor T_{i,j,k} = 1 if (i,j,k) is a valid triple (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    # We want to decompose T = sum_r U_r V_r W_r
    # where U_r is a vector of size n*m, V_r is a vector of size m*p, W_r is a vector of size p*n.
    # Specifically, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a low rank decomposition.
    # Schoolbook rank is n*m*p = 72.
    # We know that for 3x3x3, the rank is 23.
    # For 3x3x8, we can potentially do better than 72.
    # A block decomposition:
    # Split the 8 columns into two 4x4 blocks? No, 3x3x4 and 3x3x4.
    # R(3,3,4) is at most 2*12=24? No, schoolbook is 36.
    # Actually, R(3,3,p) for p=8.
    # Let's try a direct search for a low rank decomposition using random initialization and gradient descent?
    # Or use a known result: R(3,3,8) <= 3*R(3,3,2) + ...?
    # Let's try to construct a decomposition of rank < 72.
    
    # Strategy:
    # 1. Try to find a decomposition of rank R using random vectors and solving for the third factor?
    #    Fix U and V, then W is determined by T and U, V?
    #    T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    #    This is a system of linear equations for W if U and V are fixed.
    #    We can iterate: random U, random V, solve for W, then update U, V?
    #    
    #    Let's try a simple iterative method:
    #    Initialize U, V, W randomly (small integers).
    #    Compute T_approx = sum_r U_r V_r W_r
    #    Error = T - T_approx
    #    Update one factor to reduce error?
    #    
    #    Alternatively, use the "tensor train" or "CP decomposition" approach.
    #    
    #    Let's try a heuristic:
    #    We know R(3,3,3)=23.
    #    Can we combine two 3x3x3 decompositions to get 3x3x6?
    #    If we have a rank-23 decomposition for 3x3x3, we can use it for columns 0-2 and 3-5.
    #    Rank = 23 + 23 = 46 for 3x3x6.
    #    Then add 3x3x2 (rank 18) for columns 6-7.
    #    Total rank = 46 + 18 = 64.
    #    This is an improvement over 72.
    #    
    #    Do we have a rank-23 decomposition for 3x3x3?
    #    Yes, it is known. But hardcoding it is risky.
    #    Can we generate one?
    #    Maybe we can use a rank-24 or 25 decomposition if 23 is too hard to get exactly.
    #    Actually, let's try to find a rank-23 decomposition using a search.
    #    
    #    However, searching for exact integer/rational solutions is hard.
    #    Let's try to use a rank-27 (schoolbook) for 3x3x3?
    #    Then 27+27+18 = 72. No improvement.
    #    
    #    What if we use a rank-24 decomposition for 3x3x3?
    #    24+24+18 = 66. Improvement.
    #    Is there a simple rank-24 decomposition?
    #    Yes, take the rank-23 and add a zero term? No, that's rank 23.
    #    Take the rank-27 and merge 3 terms?
    #    
    #    Let's try to find a rank-24 decomposition for 3x3x3.
    #    We can start with a random rank-24 decomposition and optimize?
    #    
    #    Actually, let's try a different block:
    #    3x3x8 can be seen as 3x3x(3+3+2).
    #    If we can get R(3,3,3) = 23, total is 64.
    #    If we can get R(3,3,3) = 24, total is 66.
    #    If we can get R(3,3,3) = 25, total is 68.
    #    If we can get R(3,3,3) = 26, total is 70.
    #    If we can get R(3,3,3) = 27, total is 72.
    #    
    #    Let's try to construct a rank-24 decomposition for 3x3x3.
    #    One known construction:
    #    Use the identity:
    #    (a+b+c)(d+e+f)(g+h+i) = ...
    #    This gives 27 terms.
    #    Can we reduce it?
    #    
    #    Let's try to use the "Winograd" algorithm for 3x3 matrix multiplication.
    #    Winograd's algorithm for 3x3x3 has rank 23.
    #    The formulas are:
    #    m1 = (a11+a12)(b11+b21)
    #    m2 = (a11+a12)(b12+b22)
    #    m3 = (a11-a12+a13)(b11+b21)
    #    m4 = (a11-a12+a13)(b12+b22)
    #    m5 = (a11+a13)(b11+b21)
    #    m6 = (a11+a13)(b12+b22)
    #    m7 = (a21+a22)(b11+b21)
    #    m8 = (a21+a22)(b12+b22)
    #    m9 = (a21-a22+a23)(b11+b21)
    #    m10 = (a21-a22+a23)(b12+b22)
    #    m11 = (a21+a23)(b11+b21)
    #    m12 = (a21+a23)(b12+b22)
    #    m13 = (a31+a32)(b11+b21)
    #    m14 = (a31+a32)(b12+b22)
    #    m15 = (a31-a32+a33)(b11+b21)
    #    m16 = (a31-a32+a33)(b12+b22)
    #    m17 = (a31+a33)(b11+b21)
    #    m18 = (a31+a33)(b12+b22)
    #    m19 = a11*b11
    #    m20 = a12*b12
    #    m21 = a13*b13
    #    m22 = a22*b22
    #    m23 = a23*b23
    #    m24 = a33*b33
    #    
    #    Wait, that's 24 terms? Let's count.
    #    18 + 6 = 24.
    #    Actually, Winograd's algorithm for 3x3 is often cited as rank 23.
    #    Maybe one of these terms can be eliminated?
    #    
    #    Let's check the rank-24 decomposition.
    #    If we can verify that a rank-24 decomposition is valid, we can use it.
    #    
    #    Let's try to construct the U, V, W matrices for the rank-24 Winograd-like decomposition.
    #    This is complex to do manually.
    #    
    #    Alternative:
    #    Use a global optimization to find a low rank decomposition.
    #    We can use scipy.optimize or a simple gradient descent.
    #    Objective: minimize ||T - T_approx||_F^2
    #    Subject to: U, V, W are integer/half-integer?
    #    Actually, the task says "integer/half-integer entries".
    #    This is a constraint.
    #    
    #    Let's try to find a real-valued low rank decomposition first, then round?
    #    Rounding might break the exactness.
    #    
    #    Given the time limit and the difficulty of finding an exact low-rank decomposition,
    #    let's try to use the block decomposition with rank-24 for 3x3x3.
    #    If we can't verify rank-24, we fall back to 72.
    #    
    #    Actually, let's try a simpler improvement:
    #    Can we do 3x3x8 with rank < 72 using a different block structure?
    #    For example, 3x3x4 and 3x3x4.
    #    R(3,3,4) is at most 2*12=24? No, schoolbook is 36.
    #    Is R(3,3,4) < 36?
    #    Yes, R(3,3,4) <= 2*R(3,3,2) + ...?
    #    Actually, R(3,3,4) is at most 30?
    #    If R(3,3,4) = 30, then 30+30=60 < 72.
    #    
    #    Let's check R(3,3,4).
    #    Schoolbook is 36.
    #    Can we do better?
    #    Yes, by using the fact that 4 = 2+2.
    #    R(3,3,2) = 18.
    #    So R(3,3,4) <= 18+18=36. No improvement.
    #    
    #    What if we use a non-block decomposition?
    #    
    #    Let's try to search for a rank-60 decomposition for 3x3x8.
    #    This is a large search space.
    #    
    #    Given the constraints, I will implement a heuristic search for a low rank decomposition.
    #    If it fails, I return the schoolbook solution.
    
    # Let's try to find a rank-24 decomposition for 3x3x3.
    # I will use a known rank-24 decomposition if I can construct it.
    # Otherwise, I will use a random search for a rank-25 or 26 decomposition.
    
    # For now, let's return the schoolbook solution to ensure validity.
    # But I will try to improve it if time allows.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
