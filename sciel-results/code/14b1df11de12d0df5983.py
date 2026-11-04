"""
Strategy: Structured Search with Known Low-Rank Kernels.
Instead of generic tiling or ALS, we leverage the known rank-40 decomposition 
for 3x3x3 matrix multiplication (Landsberg et al.) and extend it to 3x4x8.
The 3x4x8 problem can be viewed as 3x3x8 plus 3x1x8.
We construct the 3x3x8 part using the 3x3x3 rank-40 kernel applied to two 
disjoint 3x3x3 blocks (for k=0..2 and k=3..5) and schoolbook for the rest, 
then optimize/verify.

Actually, a more direct approach for 3x4x8:
We know R(3,3,3) <= 40.
We can decompose 3x4x8 into:
- 3x3x3 (indices i:0..2, j:0..2, k:0..2) -> Rank 40
- 3x3x3 (indices i:0..2, j:0..2, k:3..5) -> Rank 40
- 3x3x2 (indices i:0..2, j:0..2, k:6..7) -> Rank 18 (schoolbook)
- 3x1x8 (indices i:0..2, j:3,   k:0..7) -> Rank 24 (schoolbook)
Total: 40 + 40 + 18 + 24 = 122. This is worse than 88.

Let's look at the dimensions again: n=3, m=4, p=8.
Schoolbook: 96.
Strassen tiling (current): 88.

Is there a better structured decomposition?
R(3,4,8) <= R(3,3,8) + R(0,1,8)? No.
R(3,4,8) <= R(3,4,4) + R(3,4,4)?
R(3,4,4) <= R(3,3,4) + R(0,1,4) = R(3,3,4) + 12.
R(3,3,4) <= R(3,3,3) + R(3,3,1) = 40 + 9 = 49.
So R(3,4,4) <= 49 + 12 = 61.
R(3,4,8) <= 2 * 61 = 122. Still bad.

What about R(3,4,8) <= R(3,4,2) * 4?
R(3,4,2) <= 3*4*2 = 24.
24 * 4 = 96.

Let's try a different split:
R(3,4,8) <= R(2,4,8) + R(1,4,8).
R(2,4,8) <= 2 * R(2,2,4) * 2? No.
R(2,4,8) <= R(2,2,8) * 2.
R(2,2,8) <= 2 * R(2,2,2) * 2? No, R(2,2,8) <= 7 * 4 = 28? No, R(2,2,8) <= R(2,2,4)*2.
R(2,2,4) <= 7 * 2 = 14.
R(2,2,8) <= 14 * 2 = 28.
R(2,4,8) <= 28 * 2 = 56.
R(1,4,8) <= 1 * 4 * 8 = 32.
Total: 56 + 32 = 88. This is the current best.

Can we do better?
R(3,4,8) <= R(3,2,8) + R(0,2,8).
R(3,2,8) <= R(3,2,4) * 2.
R(3,2,4) <= R(3,1,4) + R(0,1,4) = 12 + 4 = 16? No, R(3,1,4) = 12.
Wait, R(3,2,4) <= R(2,2,4) + R(1,2,4) = 14 + 8 = 22.
R(3,2,8) <= 22 * 2 = 44.
R(0,2,8) = R(2,8) = 16.
Total: 44 + 16 = 60.

Let's verify R(3,2,4) <= 22.
R(3,2,4):
Split n=3 into 2+1.
R(2,2,4) = 14 (2 Strassen 2x2x2 blocks).
R(1,2,4) = 1*2*4 = 8.
Total 22. Correct.
R(3,2,8) = R(3,2,4) * 2 = 44.
R(0,2,8) = R(2,8) = 16.
Total 60.

This is a significant improvement over 88!
Let's construct this decomposition.

Decomposition Plan:
1. Split n (rows of A) into n1=2, n2=1.
   A is 3x4. A1 is 2x4, A2 is 1x4.
   C is 3x8. C1 is 2x8, C2 is 1x8.
   
   C1 = A1 * B (2x4 * 4x8)
   C2 = A2 * B (1x4 * 4x8)

2. For C1 = A1 * B:
   A1 is 2x4. B is 4x8.
   Split m (cols of A1/rows of B) into m1=2, m2=2.
   A1 = [A1_1, A1_2] where A1_1 is 2x2, A1_2 is 2x2.
   B = [B_1; B_2] where B_1 is 2x8, B_2 is 2x8.
   C1 = A1_1 * B_1 + A1_2 * B_2.
   
   For A1_1 * B_1 (2x2 * 2x8):
   Split p (cols of B_1) into p1=4, p2=4.
   B_1 = [B_1_1, B_1_2] (2x4, 2x4).
   C1_1 = A1_1 * B_1_1 (2x2 * 2x4).
   C1_2 = A1_1 * B_1_2 (2x2 * 2x4).
   
   R(2,2,4) = 14 (Two 2x2x2 Strassen blocks).
   So C1_1 uses rank 14.
   C1_2 uses rank 14.
   Total for A1_1 * B_1 is 28.
   
   Similarly for A1_2 * B_2 (2x2 * 2x8):
   Split p into 4+4.
   C1_3 = A1_2 * B_2_1 (rank 14).
   C1_4 = A1_2 * B_2_2 (rank 14).
   Total for C1 is 28 + 28 = 56.

3. For C2 = A2 * B (1x4 * 4x8):
   This is just dot product of row A2 with rows of B.
   Rank = 1 * 4 * 8 = 32.
   
   Total Rank = 56 + 32 = 88.
   
   Wait, my previous calculation was:
   R(3,2,8) <= 44.
   R(0,2,8) <= 16.
   Total 60.
   
   Where did I go wrong in the manual construction?
   R(3,2,8): n=3, m=2, p=8.
   My construction above was for n=2, m=4, p=8.
   Let's re-evaluate R(3,2,8).
   n=3, m=2, p=8.
   Split n into 2+1.
   R(2,2,8) + R(1,2,8).
   R(2,2,8): n=2, m=2, p=8.
   Split p into 4+4.
   R(2,2,4) + R(2,2,4) = 14 + 14 = 28.
   R(1,2,8): n=1, m=2, p=8.
   Rank = 1*2*8 = 16.
   Total R(3,2,8) = 28 + 16 = 44.
   
   R(0,2,8): This corresponds to the part where we split m?
   No, the split was n=3 -> 2+1.
   The remaining part is R(1,4,8)? No.
   Original problem: 3x4 * 4x8.
   Split n=3 into 2+1.
   Part 1: 2x4 * 4x8. (n=2, m=4, p=8).
   Part 2: 1x4 * 4x8. (n=1, m=4, p=8).
   
   R(2,4,8):
   Split m=4 into 2+2.
   Part 1.1: 2x2 * 2x8.
   Part 1.2: 2x2 * 2x8.
   R(2,2,8) = 28.
   So R(2,4,8) = 28 + 28 = 56.
   
   R(1,4,8) = 1*4*8 = 32.
   
   Total = 56 + 32 = 88.
   
   My previous "60" calculation was:
   R(3,4,8) <= R(3,2,8) + R(0,2,8).
   This implies splitting m=4 into 2+2.
   Part 1: 3x2 * 2x8. (n=3, m=2, p=8).
   Part 2: 3x2 * 2x8. (n=3, m=2, p=8).
   
   R(3,2,8) = 44 (calculated above).
   So R(3,4,8) <= 44 + 44 = 88.
   
   It seems 88 is the limit of this simple tiling.
   
   Is there a non-tiling improvement?
   Landsberg's 40 for 3x3x3 is much better than 27 (schoolbook) or 32 (Strassen tiling 2+1).
   3x3x3 tiling: 2x2x2 (7) + 2x1x1 (2) + 1x2x1 (2) + 1x1x2 (2) + 1x1x1 (1) = 14? No.
   3x3x3 tiling:
   2x2x2 (7)
   2x1x1 (2)
   1x2x1 (2)
   1x1x2 (2)
   1x1x1 (1)
   Total 14? No, that's for 3x3x3?
   2x2x2 covers i<2, j<2, k<2.
   Remaining:
   i<2, j=2, k<2 (2x1x2) -> 4
   i<2, j<2, k=2 (2x2x1) -> 4
   i=2, j<2, k<2 (1x2x2) -> 4
   i<2, j=2, k=2 (2x1x1) -> 2
   i=2, j<2, k=2 (1x2x1) -> 2
   i=2, j=2, k<2 (1x1x2) -> 2
   i=2, j=2, k=2 (1x1x1) -> 1
   Total: 7 + 4 + 4 + 4 + 2 + 2 + 2 + 1 = 26.
   Schoolbook is 27. Strassen tiling gives 26.
   Landsberg gives 40? No, 40 is for 3x3x3?
   Wait, R(3,3,3) is known to be 40?
   No, R(3,3,3) is between 19 and 23? No.
   R(2,2,2) = 7.
   R(3,3,3) <= 23 (Landsberg).
   Schoolbook 27.
   
   If R(3,3,3) = 23, then:
   R(3,4,8) using 3x3x3 blocks?
   3x4x8 -> 
   3x3x3 (23)
   3x3x3 (23)
   3x3x2 (18)
   3x1x8 (24)
   Total: 23+23+18+24 = 88. Still 88.
   
   What if we mix?
   The current program scores 1.0 (rank 88).
   To improve, we need rank < 88.
   
   Is R(3,4,8) < 88?
   Lower bound: max(3*4, 4*8, 3*8) = 32.
   Also R >= (n*m*p) / max(n,m,p) ? No.
   
   Let's try to use the 3x3x3 rank-23 decomposition more effectively.
   Or maybe 4x4x4?
   R(4,4,4) = 49 (Strassen).
   
   If we use 4x4x4 blocks?
   3x4x8.
   We can't fit a 4x4x4 block because n=3.
   
   How about 2x4x4?
   R(2,4,4) <= R(2,2,4) + R(0,2,4) = 14 + 8 = 22.
   R(2,4,8) = 2 * R(2,4,4) = 44.
   R(1,4,8) = 32.
   Total 76.
   
   Wait, R(2,4,4) = 22?
   n=2, m=4, p=4.
   Split m=4 into 2+2.
   R(2,2,4) + R(0,2,4).
   R(2,2,4) = 14.
   R(0,2,4) = R(2,4) = 8.
   Total 22.
   
   So R(2,4,8) = 44.
   R(1,4,8) = 32.
   Total 76.
   
   This is better than 88!
   
   Let's verify R(2,2,4) = 14.
   n=2, m=2, p=4.
   Split p=4 into 2+2.
   R(2,2,2) + R(2,2,2) = 7 + 7 = 14.
   Yes.
   
   So the decomposition is:
   1. Split n=3 into n1=2, n2=1.
   2. For n1=2 (2x4 * 4x8):
      Split m=4 into m1=2, m2=2.
      2a. 2x2 * 2x8 (A1_1 * B_1).
          Split p=8 into p1=4, p2=4.
          2a1. 2x2 * 2x4 (Rank 14).
          2a2. 2x2 * 2x4 (Rank 14).
          Total 28.
      2b. 2x2 * 2x8 (A1_2 * B_2).
          Split p=8 into p1=4, p2=4.
          2b1. 2x2 * 2x4 (Rank 14).
          2b2. 2x2 * 2x4 (Rank 14).
          Total 28.
      Total for n1=2 is 56.
      
      Wait, I said R(2,4,8) = 44 earlier.
      Why the discrepancy?
      R(2,4,8) = R(2,4,4) * 2?
      R(2,4,4) = 22.
      22 * 2 = 44.
      
      My manual construction:
      2x2 * 2x8.
      R(2,2,8) = 28.
      Two of them: 56.
      
      Why is R(2,4,8) not 56?
      R(2,4,8) <= R(2,2,8) + R(0,2,8).
      R(2,2,8) = 28.
      R(0,2,8) = R(2,8) = 16.
      Total 44.
      
      Ah, I split m=4 into 2+2.
      A1 is 2x4. A1 = [A1_1, A1_2].
      B is 4x8. B = [B_1; B_2].
      C1 = A1_1 B_1 + A1_2 B_2.
      
      A1_1 is 2x2. B_1 is 2x8.
      A1_2 is 2x2. B_2 is 2x8.
      
      My calculation of R(2,2,8) was 28.
      So R(2,4,8) = 28 + 28 = 56.
      
      Where did 44 come from?
      R(2,4,8) <= R(2,2,8) + R(0,2,8).
      This split is m=4 -> 2+2.
      Term 1: n=2, m=2, p=8. (A1_1 * B_1). Rank 28.
      Term 2: n=2, m=2, p=8. (A1_2 * B_2). Rank 28.
      Total 56.
      
      Why did I think R(0,2,8) = 16?
      R(0,2,8) implies n=0? No, n=2.
      The split is m.
      R(n, m1, p) + R(n, m2, p).
      R(2, 2, 8) + R(2, 2, 8) = 28 + 28 = 56.
      
      Okay, so R(2,4,8) = 56.
      R(1,4,8) = 32.
      Total 88.
      
      It seems 88 is robust for this tiling.
      
      Is there a better tiling?
      Split p=8 into 3+5?
      R(3,4,3) + R(3,4,5).
      R(3,4,3) <= R(3,3,3) + R(0,1,3) = 23 + 3 = 26.
      R(3,4,5) <= R(3,3,5) + R(0,1,5).
      R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 18 = 41.
      R(0,1,5) = 5.
      Total 46.
      26 + 46 = 72.
      
      This is better! 72 < 88.
      
      Let's verify R(3,3,3) = 23.
      Yes, Landsberg 1990.
      
      Let's verify R(3,3,2) = 18.
      Schoolbook 18.
      Can we do better?
      R(3,3,2) <= R(2,3,2) + R(1,3,2).
      R(2,3,2) <= R(2,2,2) + R(0,1,2) = 7 + 2 = 9.
      R(1,3,2) = 6.
      Total 15.
      
      So R(3,3,2) <= 15.
      
      Then R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 15 = 38.
      R(3,4,5) <= R(3,3,5) + R(0,1,5) = 38 + 5 = 43.
      R(3,4,3) <= R(3,3,3) + R(0,1,3) = 23 + 3 = 26.
      Total 26 + 43 = 69.
      
      Even better! 69.
      
      Can we improve R(3,3,5) further?
      R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 15 = 38.
      Or R(3,3,5) <= R(3,2,5) + R(0,1,5).
      R(3,2,5) <= R(3,2,3) + R(3,2,2).
      R(3,2,3) <= R(2,2,3) + R(1,2,3) = 14 + 6 = 20.
      R(3,2,2) <= R(3,1,2) + R(0,1,2) = 6 + 2 = 8.
      Total 28.
      R(3,3,5) <= 28 + 5 = 33.
      
      Then R(3,4,5) <= 33 + 5 = 38.
      Total 26 + 38 = 64.
      
      This is a strong candidate: Rank 64.
      
      We need the explicit decomposition for R(3,3,3) rank 23.
      This is complex to hardcode.
      
      Alternative: Use ALS to find the rank 23 decomposition for 3x3x3, then embed it.
      But ALS might not find the exact integer/half-integer solution easily.
      
      However, we can try to construct the 64-rank decomposition using the tiling logic, but using a "black box" for 3x3x3 that we try to solve with ALS at rank 23.
      
      If ALS fails to find rank 23, we fall back to rank 27 (schoolbook) or 26 (Strassen tiling).
      If we use 26 for 3x3x3:
      R(3,3,5) <= 26 + 15 = 41.
      R(3,4,5) <= 41 + 5 = 46.
      R(3,4,3) <= 26 + 3 = 29.
      Total 29 + 46 = 75.
      
      Still better than 88.
      
      Let's implement the tiling with the 3x3x3 block.
      We will try to find the 3x3x3 decomposition using ALS at rank 23, 24, ... 27.
      If found, use it. If not, use Strassen tiling (rank 26).
      
      Structure:
      1. Define a function to get a 3x3x3 decomposition.
         - Try ALS for rank 23.
         - If fail, use Strassen tiling (rank 26).
      2. Construct the 3x4x8 decomposition using the 3x3x3 block.
         - Split p=8 into 3+5.
         - Block 1: 3x4x3.
           - Split m=4 into 3+1.
           - 3x3x3 (Use best 3x3x3).
           - 3x1x3 (Schoolbook, rank 9).
         - Block 2: 3x4x5.
           - Split m=4 into 3+1.
           - 3x3x5.
             - Split p=5 into 3+2.
             - 3x3x3 (Use best 3x3x3).
             - 3x3x2 (Use best 3x3x2, e.g. Strassen tiling rank 15 or schoolbook 18).
           - 3x1x5 (Schoolbook, rank 15).
           
      Let's calculate the rank with Strassen 3x3x3 (26) and Strassen 3x3x2 (15).
      Block 1: 26 + 9 = 35.
      Block 2:
        3x3x5: 26 + 15 = 41.
        3x1x5: 15.
        Total 56.
      Grand Total: 35 + 56 = 91.
      
      Wait, 91 > 88.
      
      My previous estimate for 3x3x3 was 23.
      If 23:
      Block 1: 23 + 9 = 32.
      Block 2:
        3x3x5: 23 + 15 = 38.
        3x1x5: 15.
        Total 53.
      Grand Total: 32 + 53 = 85.
      
      85 < 88.
      
      So if we can find the rank 23 decomposition, we get 85.
      
      What if we use a better 3x3x2?
      R(3,3,2) <= 15.
      
      What if we split 3x3x5 differently?
      R(3,3,5) <= R(3,2,5) + R(0,1,5).
      R(3,2,5) <= 28.
      R(3,3,5) <= 28 + 5 = 33.
      If 33:
      Block 2: 33 + 15 = 48.
      Grand Total: 32 + 48 = 80.
      
      Even better!
      
      So the plan is:
      1. Find best 3x3x3 decomposition (target 23).
      2. Find best 3x2x5 decomposition (target 28).
      3. Assemble.
      
      This requires solving two sub-problems.
      Given the time constraint, I will implement the tiling logic and use ALS for the 3x3x3 block.
      If ALS fails, I will fall back to the known 88-rank Strassen tiling.
      
      To be safe, I will first try to construct the 85-rank solution assuming 3x3x3 rank 23.
      If I can't find it, I return the 88-rank solution.
      
      Actually, I can hardcode the 3x3x3 rank 23 decomposition if I find it online or derive it.
      But I don't have it memorized.
      
      I will rely on ALS to find the 3x3x3 rank 23 decomposition.
      If it fails, I use the 88-rank solution.
      
      The 88-rank solution is guaranteed.
      The 85-rank solution is a bonus.
      
      Let's write the code.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition, rank = n*m*p."""
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

def _strassen_decomposition(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks where possible.
    This is the baseline 'tiling' strategy which achieves rank 88 for 3x4x8.
    """
    def splits_of(N):
        if N == 0:
            return [[]]
        result = []
        for first in range(1, N+1):
            for rest in splits_of(N - first):
                result.append([first] + rest)
        return result
    
    def block_rank(di, dj, dk):
        if di == 2 and dj == 2 and dk == 2:
            return 7
        return di * dj * dk
    
    best_rank = n * m * p
    best_split = None
    
    i_splits = splits_of(n)
    m_splits = splits_of(m)
    p_splits = splits_of(p)
    
    for iseg in i_splits:
        for mseg in m_splits:
            for pseg in p_splits:
                total_rank = 0
                for di in iseg:
                    for dj in mseg:
                        for dk in pseg:
                            total_rank += block_rank(di, dj, dk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (iseg, mseg, pseg)
    
    if best_split is None:
        return None
    
    iseg, mseg, pseg = best_split
    rank = best_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    i0 = 0
    for di in iseg:
        j0 = 0
        for dj in mseg:
            k0 = 0
            for dk in pseg:
                if di == 2 and dj == 2 and dk == 2:
                    strassen_terms = [
                        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
                        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
                        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
                        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
                        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
                        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
                    ]
                    for t in range(7):
                        a_c, b_c, c_c = strassen_terms[t]
                        for a_idx in range(4):
                            ai, aj = a_idx // 2, a_idx % 2
                            U[(i0+ai)*m + (j0+aj), col] = a_c[a_idx]
                        for b_idx in range(4):
                            bj, bk = b_idx // 2, b_idx % 2
                            V[(j0+bj)*p + (k0+bk), col] = b_c[b_idx]
                        for c_idx in range(4):
                            ck, ci = c_idx // 2, c_idx % 2
                            W[(k0+ck)*n + (i0+ci), col] = c_c[c_idx]
                        col += 1
                else:
                    for ai in range(di):
                        for aj in range(dj):
                            for ak in range(dk):
                                U[(i0+ai)*m + (j0+aj), col] = 1
                                V[(j0+aj)*p + (k0+ak), col] = 1
                                W[(k0+ak)*n + (i0+ai), col] = 1
                                col += 1
                k0 += dk
            j0 += dj
        i0 += di
    
    if col != rank:
        return None
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _als_optimize(T, rank, max_iter=100, tol=1e-10):
    """
    Alternating Least Squares to minimize ||T - U*V*W||_F.
    """
    n, m, p = T.shape
    dims = [n*m, m*p, p*n]
    
    U = np.random.randn(dims[0], rank) * 0.1
    V = np.random.randn(dims[1], rank) * 0.1
    W = np.random.randn(dims[2], rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iter):
        B_mat = np.kron(W, V.T)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat, T.transpose(1, 0, 2).reshape(dims[1], dims[0]*dims[2]).T.T, rcond=None)
            # Wait, T is (n*m, m*p, p*n).
            # T_flat for U update: (n*m, m*p*p*n).
            # T.reshape(n*m, -1)
            T_flat_U = T.reshape(dims[0], dims[1]*dims[2])
            U_new, _, _, _ = np.linalg.lstsq(B_mat, T_flat_U.T, rcond=None)
            U = U_new.T
        except:
            break
            
        T_flat_V = T.transpose(1, 0, 2).reshape(dims[1], dims[0]*dims[2])
        B_mat_V = np.kron(W, U.T)
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat_V, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            break
            
        T_flat_W = T.transpose(2, 0, 1).reshape(dims[2], dims[0]*dims[1])
        B_mat_W = np.kron(V, U.T)
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat_W, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            break
            
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W, err

def _find_333_decomposition(deadline):
    """Try to find rank 23 decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Try ranks 23, 24, 25, 26
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 5:
            break
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            U, V, W, err = _als_optimize(T, rank, max_iter=50)
            if err < 1e-8:
                U_int = np.round(U * 2)
                V_int = np.round(V * 2)
                W_int = np.round(W * 2)
                U_check = U_int / 2.0
                V_check = V_int / 2.0
                W_check = W_int / 2.0
                T_approx = np.einsum('ic,jc,kc->ijk', U_check, V_check, W_check)
                if np.linalg.norm(T - T_approx) < 1e-8:
                    return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
    return None

def _construct_improved_348(deadline):
    """
    Construct 3x4x8 decomposition using 3x3x3 block.
    Split p=8 into 3+5.
    Block 1: 3x4x3.
    Block 2: 3x4x5.
    """
    n, m, p = 3, 4, 8
    
    # Get 3x3x3 decomposition
    dec_333 = _find_333_decomposition(deadline)
    if dec_333 is None:
        # Fallback to Strassen tiling for 3x3x3 (rank 26)
        dec_333 = _strassen_decomposition(3, 3, 3)
        if dec_333 is None:
            return None
            
    U333 = np.array(dec_333['U'])
    V333 = np.array(dec_333['V'])
    W333 = np.array(dec_333['W'])
    r333 = U333.shape[1]
    
    # We also need 3x3x2 decomposition. Use Strassen tiling (rank 15) or Schoolbook (18).
    dec_332 = _strassen_decomposition(3, 3, 2)
    if dec_332 is None:
        dec_332 = _schoolbook(3, 3, 2)
    U332 = np.array(dec_332['U'])
    V332 = np.array(dec_332['V'])
    W332 = np.array(dec_332['W'])
    r332 = U332.shape[1]
    
    # Schoolbook for small blocks
    def schoolbook_block(n, m, p, offset_u, offset_v, offset_w):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Total rank estimate
    # Block 1 (3x4x3):
    #   3x3x3 (r333) + 3x1x3 (9)
    # Block 2 (3x4x5):
    #   3x3x5 (r333 + r332) + 3x1x5 (15)
    # Total: r333 + 9 + r333 + r332 + 15 = 2*r333 + r332 + 24
    
    total_rank = 2*r333 + r332 + 24
    
    U = np.zeros((12, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((24, total_rank))
    
    col = 0
    
    # --- Block 1: 3x4x3 (k=0..2) ---
    # Split m=4 into 3+1.
    # Part 1.1: 3x3x3 (i=0..2, j=0..2, k=0..2)
    U[:, col:col+r333] = U333
    V[0:9, col:col+r333] = V333
    W[0:9, col:col+r333] = W333
    col += r333
    
    # Part 1.2: 3x1x3 (i=0..2, j=3, k=0..2)
    U12, V12, W12, r12 = schoolbook_block(3, 1, 3, 0, 0, 0)
    U[:, col:col+r12] = U12
    V[9:12, col:col+r12] = V12
    W[0:9, col:col+r12] = W12
    col += r12
    
    # --- Block 2: 3x4x5 (k=3..7) ---
    # Split m=4 into 3+1.
    # Part 2.1: 3x3x5 (i=0..2, j=0..2, k=3..7)
    #   Split p=5 into 3+2.
    #   Part 2.1.1: 3x3x3 (i=0..2, j=0..2, k=3..5)
    U[:, col:col+r333] = U333
    V[0:9, col:col+r333] = V333
    # W indices for k=3..5, i=0..2.
    # W is (p*n, rank) = (24, rank).
    # k*n+i. k=3..5, i=0..2.
    # Indices: 3*3+0=9, 3*3+1=10, 3*3+2=11, 4*3+0=12, ...
    # W333 is (9, r333).
    # We need to place it at W[9:18, col:col+r333].
    W[9:18, col:col+r333] = W333
    col += r333
    
    #   Part 2.1.2: 3x3x2 (i=0..2, j=0..2, k=6..7)
    U[:, col:col+r332] = U332
    V[0:9, col:col+r332] = V332
    # W indices for k=6..7, i=0..2.
    # k=6: 18, 19, 20. k=7: 21, 22, 23.
    W[18:24, col:col+r332] = W332
    col += r332
    
    # Part 2.2: 3x1x5 (i=0..2, j=3, k=3..7)
    U22, V22, W22, r22 = schoolbook_block(3, 1, 5, 0, 0, 0)
    U[:, col:col+r22] = U22
    V[9:12, col:col+r22] = V22
    # W indices for k=3..7, i=0..2.
    # k=3: 9, 10, 11. k=4: 12, 13, 14. k=5: 15, 16, 17. k=6: 18, 19, 20. k=7: 21, 22, 23.
    # Total 15 indices.
    W[9:24, col:col+r22] = W22
    col += r22
    
    if col != total_rank:
        return None
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try improved decomposition
    result = _construct_improved_348(deadline)
    if result is not None:
        return result
        
    # Fallback to Strassen tiling
    result = _strassen_decomposition(n, m, p)
    if result is not None:
        return result
        
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
