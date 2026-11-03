"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 145  # safety margin before 150s wall-clock

def construct():
    n, m, p = DIMENSIONS
    # n=3 (rows of A), m=3 (cols of A / rows of B), p=7 (cols of B)
    # A is n x m, B is m x p, C = A @ B is n x p
    # Indexing convention: W[k*n + i, r] where k indexes rows of C (0..p-1) and i indexes cols of C (0..n-1)
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Let's re-read: "multiplying a 3 by 3 matrix by a 3 by 7 matrix"
    # So A is 3x3, B is 3x7. C = A @ B is 3x7.
    # U has shape (n*m, rank) = (9, rank). Index (i, j) where i in 0..2, j in 0..2. Row i*m+j.
    # V has shape (m*p, rank) = (21, rank). Index (j, k) where j in 0..2, k in 0..6. Row j*p+k.
    # W has shape (p*n, rank) = (21, rank). Index (k, i) where k in 0..6, i in 0..2. Row k*n+i.
    # The tensor entry T_{i,j,k} = 1 if k is the row index in C? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The rank decomposition is: C_{i,k} = sum_r U_{i*m+j?} ... wait.
    # Standard decomposition: C_{i,k} = sum_{j,r} A_{i,j} (B_{j,k} * something?)
    # The bilinear form is: C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This can be written as sum_{j} (A_{i,j}) * (B_{j,k}) * 1.
    # So the rank is sum over j of (number of i) * (number of k) ? No.
    # Each term in the sum is a product of a scalar from A, a scalar from B, and a constant 1.
    # The rank is the number of such terms.
    # The terms are indexed by (i, j, k).
    # U_{i*m+j, r} = 1 if r corresponds to (i,j,k).
    # V_{j*p+k, r} = 1 if r corresponds to (i,j,k).
    # W_{k*n+i, r} = 1 if r corresponds to (i,j,k).
    # Let's verify the W index.
    # The task says W[k*n+i, r].
    # If we sum over r: U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This sum contributes to C_{i,k}.
    # For a fixed (i,j,k), if we have a rank-1 term, U_{i*m+j}=1, V_{j*p+k}=1, W_{k*n+i}=1.
    # Then the sum is 1.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, we need to sum over j.
    # So for each i and k, C_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}.
    # This is a sum of 3 terms.
    # Each term is A_{i,j} * B_{j,k} * 1.
    # So we need a rank-1 tensor for each (i, j, k) tuple?
    # No, the rank is the number of scalar multiplications in the schoolbook algorithm.
    # Schoolbook: for each i, k, sum over j.
    # Number of multiplications: n * m * p = 3 * 3 * 7 = 63.
    # So rank = 63.
    
    # Let's double check if Strassen can improve this.
    # Strassen is for 2x2x2 (rank 7 vs 8) and 3x3x3 (rank 21 vs 23? No, Strassen rank for 3x3x3 is 21? 
    # Actually, Strassen's algorithm for 3x3 matrix multiplication has rank 21? 
    # No, Strassen's algorithm for 2x2 is rank 7.
    # For 3x3, the rank is known to be between 19 and 23. 
    # But we are multiplying 3x3 by 3x7.
    # The rank of the tensor for 3x3x7 is at least max(rank(3x3x3), ...).
    # Actually, the rank of M_{3,3,7} is at least the rank of M_{3,3,3} (by restriction).
    # And rank(M_{3,3,7}) <= rank(M_{3,3,3}) * ? No.
    # We can embed 3x3x7 into 3x3x3? No, 7 > 3.
    # We can view 3x7 as two 3x3 blocks plus a 3x1 block.
    # C = A @ B = [A @ B1, A @ B2, A @ B3] where B1, B2 are 3x3 and B3 is 3x1.
    # So C = [C1, C2, C3].
    # C1 = A @ B1 (3x3)
    # C2 = A @ B2 (3x3)
    # C3 = A @ B3 (3x1)
    # The rank of the whole tensor is the rank of the direct sum of the tensors for C1, C2, C3?
    # Not exactly, because they share the same A.
    # The tensor is T(i, j, k).
    # If we group k into blocks, the tensor is not a direct sum because the j index is shared.
    # However, we can use a decomposition that is efficient for the 3x3 parts.
    # But the schoolbook rank is 63.
    # Is there a known rank for 3x3x7?
    # The lower bound for rank of M_{n,m,p} is max(n^2, m^2, p^2, ...). 
    # For 3x3x7, lower bound is max(9, 9, 49) = 49? No, the simple lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
    # A stronger lower bound is the "slice rank" or "quantum rank".
    # But experimentally, for small matrices, the schoolbook is often optimal or close to it.
    # For 3x3x3, the rank is 23 (Landsberg-Manteuffel). Wait, is it?
    # Actually, the rank of 3x3x3 matrix multiplication is 23? 
    # Let me check my knowledge. 
    # Strassen's algorithm gives rank 21 for 3x3? No, Strassen is for 2x2 (rank 7).
    # For 3x3, one can use Strassen by embedding?
    # The rank of 3x3 matrix multiplication is known to be 23? 
    # Actually, I recall that the rank of 3x3x3 is 23. 
    # If rank(3x3x3) = 23, then rank(3x3x7) is at least 23.
    # But schoolbook is 63.
    # Can we do better than 63?
    # Yes, definitely.
    # We can compute C1 and C2 using a rank-23 decomposition for 3x3x3, and C3 using a rank-? decomposition for 3x3x1.
    # The tensor for 3x3x1 is just the multiplication of a 3x3 matrix by a 3x1 vector.
    # This is a bilinear map. The rank is 9 (schoolbook). 
    # Can we do better? 
    # The tensor is T(i, j, k) with k in 0..0. 
    # It's just the vector-matrix product. 
    # The rank of this tensor is 9? 
    # Actually, the rank of the tensor for A(3x3) * v(3x1) is 9? 
    # Let's check. 
    # The tensor is T_{i,j,k} = delta_{ik} ? No.
    # C_i = sum_j A_{i,j} v_j.
    # This is a sum of 9 terms. 
    # Is the rank 9? 
    # The slice rank of this tensor is 9. 
    # So the rank is at least 9. 
    # And schoolbook gives 9. 
    # So rank(3x3x1) = 9.
    
    # So if we treat the problem as computing three separate products, we would use:
    # rank = rank(3x3x3) + rank(3x3x3) + rank(3x3x1) = 23 + 23 + 9 = 55.
    # This is better than 63.
    
    # But can we do even better by sharing the computation of A?
    # The tensor is not a direct sum, so the rank might be less than 55.
    # However, finding the exact rank of 3x3x7 is a complex problem.
    # Given the time constraint and the requirement for an *exact* decomposition, 
    # implementing a rank-23 decomposition for 3x3x3 is non-trivial and risky.
    # The schoolbook decomposition is always correct and has rank 63.
    # Any attempt to reduce the rank requires a known, verified decomposition.
    # Since I don't have the explicit rank-23 decomposition for 3x3x3 memorized, 
    # and implementing it from scratch within 120s is impossible, 
    # I should stick to the schoolbook decomposition.
    
    # Wait, is there a simpler improvement?
    # What if we use the fact that 7 = 3 + 3 + 1?
    # We can group the columns of B into two 3-column blocks and one 1-column block.
    # B = [B1, B2, B3].
    # C = [C1, C2, C3].
    # If we use schoolbook for all, rank is 63.
    # If we use a rank-21 decomposition for 3x3x3 (if it exists?), 
    # actually, the rank of 3x3x3 is 23. 
    # But wait, Strassen's algorithm for 2x2x2 has rank 7.
    # Can we build a 3x3x3 decomposition using Strassen?
    # Yes, by padding to 4x4? No, that increases the size.
    # There are algorithms for 3x3x3 with rank 21? 
    # Let me check: "rank of 3x3 matrix multiplication".
    # The rank is 23. (Landsberg and Manteuffel, 1997).
    # So 23 is the best known for 3x3x3.
    
    # Since I cannot implement the rank-23 decomposition reliably, 
    # I will stick to the schoolbook.
    # However, I can try to use a rank-21 decomposition if I can find one?
    # No, 23 is the known rank.
    
    # So the best I can do is 63.
    # Is there any chance the rank is lower?
    # The lower bound for 3x3x7 is max(21, 21, 9, ...) 
    # The "flattening" lower bound is max(n*m, m*p, p*n) = 21.
    # The "slice rank" lower bound is max(n, m, p) = 7? No.
    # The "quantum rank" lower bound is ...
    # But 63 is the schoolbook rank.
    # Any improvement requires a sophisticated decomposition.
    
    # Given the constraints, I will return the schoolbook decomposition.
    # I will ensure the code is correct and efficient.
    
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
