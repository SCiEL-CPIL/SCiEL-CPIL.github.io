"""Rewritten program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on a 2x2x2 block decomposition.
    
    We view the 2x3 matrix A as [A1, A2, A3] where each Ai is 2x1.
    We view the 3x9 matrix B as [B1; B2; B3] where each Bj is 1x9.
    
    The product C = A*B = A1*B1 + A2*B2 + A3*B3 (each term is 2x9).
    
    We can apply Strassen's trick: instead of 3 separate outer products (each of
    rank 2*9 = 18, total 54), we can combine them.
    
    Actually, let's think differently. The inner dimension is 3. We can split it
    as 3 = 2 + 1. So A = [A1, A2, A3] where A1 is 2x2, [A2, A3] is 2x1.
    B = [B1; B2] where B1 is 2x9, [B2; B3] is 1x9.
    
    C = A1*B1 + [A2, A3] * [B2; B3]
    
    The first term is a 2x2 by 2x9 multiplication. We can use Strassen for the
    2x2 part. Let's write A1 = [[a11, a12], [a21, a22]] and B1 = [B1_top; B1_bot]
    where B1_top is 1x9 and B1_bot is 1x9.
    
    C1 = A1 * B1 = [[a11*B1_top + a12*B1_bot], [a21*B1_top + a22*B1_bot]]
    
    Using Strassen for 2x2 matrices (where the "scalar" entries are actually
    1x9 vectors):
    M1 = (a11 + a22) * (B1_top + B1_bot)
    M2 = (a21 + a22) * B1_top
    M3 = a11 * (B1_top - B1_bot)
    M4 = a22 * (B1_bot - B1_top)  # or a22 * B1_bot - a22 * B1_top
    
    Wait, Strassen for C = A*B where A, B are 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B12 - B22)
    M4 = A22 (B21 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11) (B11 + B12)
    M7 = (A12 - A22) (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In our case, the "matrix entries" are actually scalars (for A) and 1x9 vectors (for B).
    So M1 = (a11 + a22) * (B1_top + B1_bot) -- this is a scalar times a 1x9 vector = 1x9 vector
    Similarly for others.
    
    C1_top = M1 + M4 - M5 + M7 (1x9 vector)
    C1_bot = M2 + M4 (1x9 vector)
    
    Wait, that's not right. Let me redo.
    
    A1 is 2x2, B1 is 2x9. The product A1*B1 is 2x9.
    Let's partition A1 = [[a, b], [c, d]] and B1 = [e; f] where e, f are 1x9.
    
    A1 * B1 = [[a*e + b*f], [c*e + d*f]]
    
    Using Strassen (7 multiplications of "scalars" where one side is scalar and other is 1x9):
    M1 = (a + d)(e + f)
    M2 = (c + d) e
    M3 = a (f - e)  # wait, this should be a*(e - f) or similar
    M4 = d (f - e)
    M5 = (a + b) f
    M6 = (c - a) e  # hmm
    M7 = (b - d) f  # hmm
    
    Actually let's use the standard Strassen formulas properly.
    
    For C = A * B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In our case, B is 2x9, so B = [[e1,...,e9],[f1,...,f9]] where e = B1_top (1x9), f = B1_bot (1x9).
    So g doesn't exist; instead, the "entries" of B are vectors.
    
    Let me reinterpret: B1 is 2x9. Think of it as B1 = [e; f] where e, f are 1x9.
    Then A1 * B1 where A1 = [[a,b],[c,d]] gives:
    Row 1: a*e + b*f (1x9)
    Row 2: c*e + d*f (1x9)
    
    Using Strassen with "scalar" a,b,c,d and "vector" e,f:
    M1 = (a+d)(e+f)  -- scalar * vector = vector
    M2 = (c+d)e
    M3 = a(f-e)  -- hmm, let me use the correct formula
    
    Actually, the issue is that standard Strassen assumes square matrices. Let me think of this differently.
    
    Alternative approach: The inner dimension is 3. We can use the fact that 3 < 2*2 = 4,
    and pad with zeros to make it a 4x4 inner product, then apply Strassen recursively.
    
    But a simpler approach: we know the rank of 2x3x9 is at most 54 (schoolbook).
    Can we do better?
    
    The rank of 2xm x p is known to be:
    - For 2x2x2: rank 7 (Strassen)
    - For 2x3x3: This is less standard.
    
    Let me try a different decomposition. View the 2x3 matrix as having 3 columns, each 2x1.
    The product is sum of 3 outer products of 2x1 and 1x9 vectors.
    
    Each outer product of a 2x1 and 1x9 has rank 2*9 = 18.
    Total: 3 * 18 = 54.
    
    Can we combine them? The key insight is that the 3 columns of A share structure.
    
    Let A = [a1, a2, a3] where ai is 2x1. Let B = [b1; b2; b3] where bj is 1x9.
    C = a1*b1 + a2*b2 + a3*b3.
    
    We can write:
    C = (a1 + a2)*b1 + a2*(b2 - b1) + a3*b3
    
    That's still 3 terms.
    
    What about:
    C = (a1 + a2 + a3)*b1 + a2*(b2 - b1) + a3*(b3 - b1)
    
    Still 3 terms, each of rank 18. Total 54.
    
    To reduce rank, we need to exploit that the "vectors" a_i are 2-dimensional.
    The tensor rank of sum of k outer products of u_i (n-dim) and v_i (p-dim) where the u_i
    span a d-dimensional space and v_i span an e-dimensional space...
    
    Actually, the rank of the matrix multiplication tensor M_{n,m,p} is known:
    - M_{2,2,2} has rank 7
    - M_{2,2,3} has rank ? 
    - M_{2,3,9} has rank ?
    
    Let me try to construct a decomposition with rank < 54 using the 2+1 split.
    
    Split inner dimension as 2 + 1:
    A = [A1, A2, A3] -> [A12, A3] where A12 is 2x2, A3 is 2x1
    B = [B1; B2; B3] -> [B12; B3] where B12 is 2x9, B3 is 1x9
    
    C = A12 * B12 + A3 * B3
    
    For A12 * B12 (2x2 * 2x9): Use Strassen on the 2x2 part.
    This requires 7 "multiplications" where each is scalar * 1x9 vector.
    Each such multiplication has rank 9 (since it's a scalar times a fixed vector pattern).
    Wait, no. The multiplication of a scalar by a 1x9 vector is itself rank 9 in the
    tensor sense? No, it's just a scaling.
    
    Actually, let me think about this more carefully. The tensor for 2x2 * 2x9 is:
    T(i, j, k) where i in {1,2} (row of A), j in {1,2} (col of A = row of B), k in {1,...,9} (col of B)
    T(i,j,k) = delta_{i,k} ... no wait.
    
    The product C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T[i,j,k] = delta_{i,j} * delta_{j,k} ... no.
    
    T[i,j,k] = 1 if the indices match in the right way. For matrix multiplication C = AB,
    T[i,j,k] = delta_{i,j} is not right.
    
    The matrix multiplication tensor M_{n,m,p} has T[i,j,k] = delta_{i,k} where i in [n], j in [m], k in [p]?
    No, T[i,j,k] = 1 if i is the row of A, j is the column of A / row of B, k is the column of B.
    The tensor entries are T[i,j,k] = delta_{i,j} * delta_{j,k}? That doesn't make sense dimensionally.
    
    Let me restart. The matrix multiplication tensor for C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_{j=1}^m A[i,j] * B[j,k]
    
    The tensor T is in R^{n x m x p} with T[i,j,k] = 1 if and only if... actually T[i,j,k] is just
    the coefficient, which is always 1 for the "matching" indices. More precisely, the tensor is:
    T = sum_{i,j,k} delta_{i,j} delta_{j,k} e_i (in R^n) (tensor) e_j (in R^m) (tensor) e_k (in R^p)
    
    Wait, that's not right either. The correct definition:
    T[i,j,k] = 1 if j is the shared index, i.e., T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    where the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    
    Actually, the tensor is simply: T[i,j,k] = 1 for all valid i, j, k. No, that's the all-ones tensor.
    
    Let me look at this differently. The rank of the matrix multiplication tensor M_{n,m,p} is the
    minimum number of rank-1 tensors that sum to M_{n,m,p}.
    
    A rank-1 tensor in R^{n x m x p} has the form u (tensor) v (tensor) w where u in R^n, v in R^m, w in R^p.
    
    For M_{2,2,2}, we need to write:
    T[i,j,k] = 1 for all i,j,k in {1,2}
    as a sum of rank-1 tensors.
    
    Wait, that's the all-ones tensor, which has rank 1! So the matrix multiplication tensor can't be the all-ones tensor.
    
    I think I'm confusing myself. Let me reconsider.
    
    The matrix multiplication map is: (A, B) -> C = A * B
    This is a bilinear map from R^{n x m} x R^{m x p} -> R^{n x p}
    The associated tensor T is in R^{n x m} (tensor) R^{m x p} (tensor) R^{n x p}
    with T[A,B,C] = <C, A*B> = sum_{i,k} C[i,k] (A*B)[i,k] = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    
    So the tensor is: T = sum_{i,j,k} e_i (in R^{n x p}, specifically E_{i,k}) (tensor) e_{i,j} (in R^{n x m}) (tensor) e_{j,k} (in R^{m x p})
    
    Hmm, this is getting complicated. Let me just use the known result.
    
    Known results:
    - rank(M_{2,2,2}) = 7
    - rank(M_{2,2,3}) >= 8, and it's known to be 8 or 9
    - rank(M_{2,3,3}) is not standard
    
    Given the time constraint, let me try a pragmatic approach: use the 2+1 split with Strassen for the 2x2 part.
    
    C = A12 * B12 + A3 * B3
    
    For A12 * B12: A12 is 2x2, B12 is 2x9.
    Write A12 = [[a, b], [c, d]], B12 = [e; f] where e, f are 1x9.
    A12 * B12 = [[a*e + b*f], [c*e + d*f]]
    
    Using 7 multiplications (Strassen-style, where "multiplication" is scalar * 1x9 vector):
    M1 = (a + d)(e + f)
    M2 = (c + d)e
    M3 = a(f - e)
    M4 = d(f - e)  -- wait, let me use the correct Strassen formulas
    
    Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In our case, B12 = [e; f] where e is row 1 (1x9), f is row 2 (1x9).
    So in the Strassen formula, we have B = [[e, ?], [f, ?]] but B is 2x9, not 2x2.
    
    I think the right way is: B12 is 2x9, so we can view it as B12 = [B12_row1; B12_row2] where each row is 1x9.
    The product A12 * B12 is 2x9.
    
    For each column k (1 to 9), we compute (A12 * B12)[:, k] = A12 * B12[:, k]
    where B12[:, k] is a 2x1 vector.
    
    So for each k, we do a 2x2 * 2x1 multiplication, which is just matrix-vector multiplication.
    A12 * [x; y] = [a*x + b*y; c*x + d*y]
    
    This requires 4 scalar multiplications per column, or we can use a 3-multiplication method:
    (a+d)(x+y) - d*x - a*y ... no, that's not simpler.
    
    Actually, 2x2 * 2x1 just requires 4 multiplications (2 in each row), and there's no known way to do it with fewer than 4 for a single column. But across 9 columns, we can batch.
    
    For the 2x2 * 2x9 product, the total number of scalar multiplications is 4*9 = 36 if done naively.
    With Strassen, we do 7 "operations" where each operation is a scalar times a 1x9 vector, which is 9 scalar multiplications each, so 7*9 = 63? That's worse!
    
    I think I'm overcomplicating this. Let me just go with the schoolbook decomposition of rank 54, since that's what the current program does, and finding a better decomposition for 2x3x9 is non-trivial.
    
    Actually, wait. The knowledge says: "Exploiting the block structure (2+1 split of inner dimension) and using Strassen's algorithm for the 2x2x2 sub-tensors reduces the rank from 54 to 50."
    
    So the idea is: split inner dimension as 2+1. For the 2x2 part (A12 * B12 where A12 is 2x2 and B12 is 2x9), we can think of B12 as 9 copies of a 2x1 vector, and apply Strassen to the 2x2 * 2x1 part... but that doesn't reduce the count.
    
    Let me think about it as: A12 * B12 where A12 is 2x2 and B12 is 2x9.
    We can view this as: for each of the 9 columns, do 2x2 * 2x1.
    2x2 * 2x1 requires 4 multiplications (or can it be done with 3?).
    
    2x2 * 2x1: [a,b; c,d] * [x;y] = [ax+by; cx+dy]
    This requires 4 multiplications: ax, by, cx, dy. Can we do it with 3?
    (a+b)(x+y) - bx - ay... that's still 3 multiplications: (a+b)(x+y), bx, ay, but then we need cx and dy too.
    
    I don't think 2x2 * 2x1 can be done with fewer than 4 multiplications in general.
    
    Let me try a completely different approach. Maybe the rank of 2x3x9 is indeed 54, and the best we can do is 54.
    
    But the knowledge says rank 50 was achieved. Let me try to understand how.
    
    "Exploiting the block structure (2+1 split of inner dimension) and using Strassen's algorithm for the 2x2x2 sub-tensors"
    
    So split inner dimension as 2 + 1:
    A = [A1, A2, A3] where A1 is 2x1, A2 is 2x1, A3 is 2x1
    B = [B1; B2; B3] where B1 is 1x9, B2 is 1x9, B3 is 1x9
    
    Group A1, A2 as A12 (2x2) and A3 as A3 (2x1)
    Group B1, B2 as B12 (2x9) and B3 as B3 (1x9)
    
    C = A12 * B12 + A3 * B3
    
    Now, A12 * B12 is 2x2 * 2x9. We can use Strassen on the 2x2 part.
    But as I noted, Strassen for 2x2 * 2x9 might not help directly.
    
    Alternative: view A12 * B12 as a 2x2 * 2x9 multiplication.
    The tensor rank of 2x2 * 2x9... 
    
    Actually, let me try yet another approach. What if we use the fact that 3 = 2 + 1 and apply
    a 3-term formula?
    
    C = a1*b1 + a2*b2 + a3*b3 where ai is 2x1 and bi is 1x9.
    
    = (a1 + a2)*b1 + a2*(b2 - b1) + a3*b3
    
    = (a1 + a2 + a3)*b1 + (a2 + a3)*(b2 - b1) + a3*(b3 - b2)
    
    Each term is an outer product of a 2x1 and 1x9, so rank 18 each. Total 54.
    
    To get below 54, we need to exploit that the a_i's are 2-dimensional.
    
    Let a_i = [ai1, ai2]^T. Then:
    C[1,k] = a11*b1[k] + a21*b2[k] + a31*b3[k]
    C[2,k] = a12*b1[k] + a22*b2[k] + a32*b3[k]
    
    This is like a 2x3 * 3x9 multiplication, which is what we started with.
    
    I'm going in circles. Let me just implement the schoolbook decomposition with rank 54, since that's safe and known to work. If the knowledge says rank 50 was achieved, I'll need to figure out the exact construction, but for now, let me ensure correctness.
    
    Actually, let me try the 2+1 split with a specific formula.
    
    C = A12 * B12 + A3 * B3
    
    For A12 * B12:
    A12 = [[a, b], [c, d]]
    B12 = [e; f] where e, f are 1x9
    
    A12 * B12 = [[a*e + b*f], [c*e + d*f]]
    
    I'll use 4 "multiplications" (scalar * 1x9 vector):
    P1 = a*e
    P2 = b*f
    P3 = c*e
    P4 = d*f
    
    A12 * B12 = [[P1 + P2], [P3 + P4]]
    
    For A3 * B3:
    A3 = [g; h] (2x1)
    B3 = 1x9
    
    A3 * B3 = [[g*B3], [h*B3]]
    
    P5 = g*B3
    P6 = h*B3
    
    Total: 6 multiplications of (scalar * 1x9 vector), each of rank 9.
    Total rank: 6 * 9 = 54.
    
    Same as before. To reduce, we need to use Strassen on the 2x2 part.
    
    Strassen for [[a,b],[c,d]] * [e; f]:
    This is a 2x2 * 2x1 multiplication.
    
    Using Strassen-like:
    M1 = (a + d)(e + f)
    M2 = (c + d)e
    M3 = a*f
    
    Then:
    a*e + b*f = M1 - d*e - a*f + b*f ... this is getting messy.
    
    Let me look up: can 2x2 * 2x1 be done with 3 multiplications?
    [a,b] * [x;y] = ax + by
    [c,d] * [x;y] = cx + dy
    
    We need 4 outputs, each a linear combination of ax, by, cx, dy.
    The rank of this bilinear map is 4 in general, so we need 4 multiplications.
    
    So Strassen doesn't help for 2x2 * 2x1.
    
    I think the rank 50 construction must use a different grouping. Let me try:
    
    Split inner dimension as 2 + 1, but group differently.
    A = [a1, a2, a3] (each 2x1)
    B = [b1; b2; b3] (each 1x9)
    
    C = a1*b1 + a2*b2 + a3*b3
    
    Group (a1, a3) and (b1, b3):
    Let A' = [a1, a3] (2x2), B' = [b1; b3] (2x9)
    And a2, b2 remain.
    
    C = A' * B' + a2 * b2
    
    A' * B' is 2x2 * 2x9, which as we saw requires 4*9 = 36 scalar multiplications (or 4 "vector" multiplications of rank 9 each = rank 36).
    a2 * b2 requires 2*9 = 18 scalar multiplications (or 2 "vector" multiplications of rank 9 each = rank 18).
    Total: 36 + 18 = 54.
    
    Still 54.
    
    I'm stuck. Let me just go with rank 54 and ensure the code is correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
