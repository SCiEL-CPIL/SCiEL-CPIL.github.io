"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen decomposition for 6x6 matrix multiplication.
    Strategy:
    - Use Strassen's algorithm recursively to reduce the number of multiplications.
    - For 6x6 matrices, we can decompose them into 2x2 blocks of size 3x3.
    - Strassen on the 2x2 "super" matrix requires 7 multiplications of 3x3 matrices.
    - Each 3x3 matrix multiplication can be done with schoolbook (27 mults) or further Strassen.
    
    Let's analyze the ranks:
    1. Pure Schoolbook: 6*6*6 = 216
    2. Strassen(2x2 of 3x3): 
       The tensor for 6x6 is isomorphic to 7 copies of 3x3 multiplication tensors?
       No, the tensor is $T_6 = \sum_{i,j,k} A_{ij} B_{jk} C_{ik}$.
       Using Strassen decomposition on the indices:
       $i = i_1 + 3 i_0$, $j = j_1 + 3 j_0$, $k = k_1 + 3 k_0$ where $i_1, j_1, k_0 \in \{0, 1\}$ and $i_0, j_0, k_0 \in \{0, 1, 2\}$.
       
       The standard Strassen algorithm decomposes the 2x2 block multiplication into 7 smaller multiplications.
       $M_1 = (A_{11} + A_{22}) (B_{11} + B_{22})$
       $M_2 = (A_{21} + A_{22}) B_{11}$
       $M_3 = A_{11} (B_{12} - B_{22})$
       $M_4 = A_{22} (B_{21} - B_{11})$
       $M_5 = (A_{11} + A_{12}) B_{22}$
       $M_6 = (A_{21} - A_{11}) (B_{11} + B_{12})$
       $M_7 = (A_{12} - A_{22}) (B_{21} + B_{22})$
       
       The resulting blocks are:
       $C_{11} = M_1 + M_4 - M_5 + M_7$
       $C_{12} = M_3 + M_5$
       $C_{21} = M_2 + M_4$
       $C_{22} = M_1 - M_2 + M_3 + M_6$
       
       Each $M_k$ is a product of 3x3 matrices.
       So we need 7 multiplications of 3x3 matrices.
       
       If we use schoolbook for the 3x3 multiplications, the rank is $7 \times (3 \times 3 \times 3) = 7 \times 27 = 189$.
       
       Can we do better?
       We can use Strassen again for the 3x3? No, 3 is not a power of 2.
       However, we can decompose 3x3 into 2x2 and 1x1 parts?
       Actually, 3x3 matrix multiplication can be done with rank 23 (using Strassen on 2x2 blocks and handling the border, or specific small-n algorithms). 
       The minimal rank for 3x3 is known to be 19 (or 20/21/23 depending on exact constraints). 
       Specifically, for 3x3, the best known upper bound is 23 (using Strassen recursively on 2x2 parts? No, 3=2+1).
       
       Let's look at the structure of 3x3 multiplication.
       Indices $i,j,k \in \{0,1,2\}$.
       We can use the fact that $T_3$ can be decomposed.
       A known decomposition for 3x3 is rank 23.
       If we use rank 23 for the 3x3 steps, the total rank would be $7 \times 23 = 161$.
       
       Let's try to implement the Strassen(2x2 of 3x3) with schoolbook 3x3 first to ensure correctness and get 189.
       Then we can try to optimize the 3x3 part.
       
       Is there a better top-level split?
       6 = 3 + 3?
       If we split 6 into 3+3, we have 4 blocks of 3x3.
       Strassen on 2x2 blocks (where block size is 3) gives rank 7 * R_3.
       
       What if we split 6 into 2+2+2?
       That would be a 3x3 grid of 2x2 blocks.
       This is more complex.
       
       Let's stick to the 2x2 grid of 3x3 blocks with Strassen, using schoolbook for the 3x3 mults.
       Rank = 189.
       
       Can we do better than 189?
       If we use a better algorithm for 3x3, say rank 23.
       How to get rank 23 for 3x3?
       One common method is to view 3x3 as a 2x2 matrix multiplication plus corrections.
       Or use the "Strassen-like" decomposition for odd sizes.
       
       Actually, there is a known decomposition for 3x3 of rank 23.
       Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication" (e.g., by Strassen or others).
       
       Let's first implement the 189 rank solution (Strassen 2x2 x Schoolbook 3x3).
       This is a significant improvement over 216.
       
       Implementation details for Strassen 2x2 on 6x6:
       Let A be 6x6, B be 6x6.
       Split A into A11, A12, A21, A22 (each 3x3).
       Split B into B11, B12, B21, B22 (each 3x3).
       
       We compute 7 intermediate products:
       P1 = (A11 + A22) * (B11 + B22)
       P2 = (A21 + A22) * B11
       P3 = A11 * (B12 - B22)
       P4 = A22 * (B21 - B11)
       P5 = (A11 + A12) * B22
       P6 = (A21 - A11) * (B11 + B12)
       P7 = (A12 - A22) * (B21 + B22)
       
       Result:
       C11 = P1 + P4 - P5 + P7
       C12 = P3 + P5
       C21 = P2 + P4
       C22 = P1 - P2 + P3 + P6
       
       Each "product" like (A11+A22)*(B11+B22) is a 3x3 matrix multiplication.
       We need to express this in the tensor decomposition form.
       
       The tensor $T_6$ is defined by $C_{ik} = \sum_j A_{ij} B_{jk}$.
       In the decomposition, we have factors U, V, W.
       We need to map the Strassen steps to U, V, W.
       
       For each of the 7 Strassen multiplications, we have a "virtual" 3x3 multiplication.
       Let's denote the inputs to the 7th Strassen multiplication as $A^{(t)}$ and $B^{(t)}$.
       $A^{(t)}_{i,j}$ is a linear combination of the original $A$ blocks.
       $B^{(t)}_{j,k}$ is a linear combination of the original $B$ blocks.
       
       Since the blocks are 3x3, the indices $i,j$ in $A^{(t)}$ correspond to the local indices within the 3x3 block.
       Let global index $I = 3 I_1 + i_1$, $J = 3 J_1 + j_1$.
       $I_1, J_1 \in \{0,1\}$ are the block indices, $i_1, j_1 \in \{0,1,2\}$ are the local indices.
       
       For a specific Strassen term $t$, the coefficient of $A_{I,J}$ in $A^{(t)}_{i_1, j_1}$ depends on $I_1, J_1$.
       Similarly for $B$.
       
       Then the product is computed by the 3x3 multiplication algorithm (schoolbook here).
       The result $M^{(t)}_{i_1, k_1}$ is added to $C$ with some coefficient.
       
       So, for each term $t \in \{0,..,6\}$, and for each schoolbook term $(i_1, j_1, k_1) \in \{0,1,2\}^3$:
       We have a rank-1 component.
       Total rank = $7 \times 3 \times 3 \times 3 = 189$.
       
       Let's define the coefficients.
       
       Strassen Terms:
       t=0: A1 = A11 + A22, B1 = B11 + B22
       t=1: A2 = A21 + A22, B2 = B11
       t=2: A3 = A11, B3 = B12 - B22
       t=3: A4 = A22, B4 = B21 - B11
       t=4: A5 = A11 + A12, B5 = B22
       t=5: A6 = A21 - A11, B6 = B11 + B12
       t=6: A7 = A12 - A22, B7 = B21 + B22
       
       Output Combination:
       C11 = P0 + P3 - P4 + P6  (Indices 0,1,2,3,4,5,6)
       C12 = P2 + P4
       C21 = P1 + P3
       C22 = P0 - P1 + P2 + P5
       
       Wait, let's re-verify the indices.
       Standard Strassen:
       P1 = (A11+A22)(B11+B22)
       P2 = (A21+A22)B11
       P3 = A11(B12-B22)
       P4 = A22(B21-B11)
       P5 = (A11+A12)B22
       P6 = (A21-A11)(B11+B12)
       P7 = (A12-A22)(B21+B22)
       
       C11 = P1 + P4 - P5 + P7
       C12 = P3 + P5
       C21 = P2 + P4
       C22 = P1 - P2 + P3 + P6
       
       Let's map this to the tensor factors.
       U has shape (n*m, rank) = (36, 189).
       V has shape (m*p, rank) = (36, 189).
       W has shape (p*n, rank) = (36, 189).
       
       The rank index $c$ corresponds to $(t, i_1, j_1, k_1)$.
       $t \in \{0,1,2,3,4,5,6\}$.
       $i_1, j_1, k_1 \in \{0,1,2\}$.
       
       For a given $c$ corresponding to $(t, i_1, j_1, k_1)$:
       
       1. U part:
          $U$ corresponds to $A_{I, J}$.
          $I = 3 I_1 + i_1$? No, $i_1$ is the local row index.
          In the schoolbook multiplication of the 3x3 blocks, the term is $A_{i_1, j_1} B_{j_1, k_1}$.
          So the local indices are $i_1$ (row), $j_1$ (col of A), $k_1$ (col of result).
          
          Wait, the Strassen algorithm operates on the 2x2 grid of 3x3 blocks.
          The inputs to the 7th multiplication are linear combinations of the 3x3 blocks.
          $A^{(t)}$ is a 3x3 matrix. Its entry $(r, c)$ is a linear combination of the corresponding entries in the original A blocks.
          
          For a fixed $t$, $A^{(t)}_{r, c}$ depends on which 3x3 block of A is used.
          Let $A$ be divided into $A_{00}, A_{01}, A_{10}, A_{11}$ (using 0-indexed block coords).
          $A^{(t)}_{r, c}$ is $\sum_{I_1, J_1} \alpha_{t, I_1 J_1} A_{I_1 J_1}[r, c]$.
          
          So, for global indices $I = 3 I_1 + r$, $J = 3 J_1 + c$:
          $U[I, J, c] = \alpha_{t, I_1 J_1}$.
          (Using 0-indexed Strassen terms $t=0..6$ corresponding to P1..P7).
          
          Coefficients $\alpha_{t, I_1 J_1}$:
          t=0 (P1): A11+A22 -> $\alpha_{0, 00}=1, \alpha_{0, 11}=1$, others 0.
          t=1 (P2): A21+A22 -> $\alpha_{1, 10}=1, \alpha_{1, 11}=1$, others 0.
          t=2 (P3): A11 -> $\alpha_{2, 00}=1$, others 0.
          t=3 (P4): A22 -> $\alpha_{3, 11}=1$, others 0.
          t=4 (P5): A11+A12 -> $\alpha_{4, 00}=1, \alpha_{4, 01}=1$, others 0.
          t=5 (P6): A21-A11 -> $\alpha_{5, 10}=1, \alpha_{5, 00}=-1$, others 0.
          t=6 (P7): A12-A22 -> $\alpha_{6, 01}=1, \alpha_{6, 11}=-1$, others 0.
          
          Note: I used 0-indexed block coords:
          00 = A11, 01 = A12, 10 = A21, 11 = A22.
          
          Let's re-check P1..P7 with 0-indexed:
          P1: (A00+A11)(B00+B11)
          P2: (A10+A11)B00
          P3: A00(B01-B11)
          P4: A11(B10-B00)
          P5: (A00+A01)B11
          P6: (A10-A00)(B00+B01)
          P7: (A01-A11)(B10+B11)
          
          C00 = P1 + P4 - P5 + P7
          C01 = P3 + P5
          C10 = P2 + P4
          C11 = P1 - P2 + P3 + P6
          
          Now for V part:
          $V$ corresponds to $B_{J, K}$.
          $J = 3 J_1 + c$ (where $c$ is local col of A, which is local row of B).
          $K = 3 K_1 + k_1$ (where $k_1$ is local col of result).
          $V[J, K, c] = \beta_{t, J_1 K_1}$.
          
          Coefficients $\beta_{t, J_1 K_1}$:
          t=0 (P1): B00+B11 -> $\beta_{0, 00}=1, \beta_{0, 11}=1$.
          t=1 (P2): B00 -> $\beta_{1, 00}=1$.
          t=2 (P3): B01-B11 -> $\beta_{2, 01}=1, \beta_{2, 11}=-1$.
          t=3 (P4): B10-B00 -> $\beta_{3, 10}=1, \beta_{3, 00}=-1$.
          t=4 (P5): B11 -> $\beta_{4, 11}=1$.
          t=5 (P6): B00+B01 -> $\beta_{5, 00}=1, \beta_{5, 01}=1$.
          t=6 (P7): B10+B11 -> $\beta_{6, 10}=1, \beta_{6, 11}=1$.
          
          Now for W part:
          $W$ corresponds to $C_{K, I}$ (trace convention $W[k*n+i, r]$).
          $K = 3 K_1 + k_1$.
          $I = 3 I_1 + i_1$.
          The result block $C_{I_1 K_1}$ receives contribution $\gamma_{t, I_1 K_1}$ from $P_t$.
          The local entry is the identity map (since schoolbook multiplication computes the exact product).
          So $W[K, I, c] = \gamma_{t, I_1 K_1}$.
          
          Coefficients $\gamma_{t, I_1 K_1}$:
          From C00 = P1 + P4 - P5 + P7:
          $\gamma_{0, 00} = 1$ (P1)
          $\gamma_{3, 00} = 1$ (P4)
          $\gamma_{4, 00} = -1$ (P5)
          $\gamma_{6, 00} = 1$ (P7)
          
          From C01 = P3 + P5:
          $\gamma_{2, 01} = 1$ (P3)
          $\gamma_{4, 01} = 1$ (P5)
          
          From C10 = P2 + P4:
          $\gamma_{1, 10} = 1$ (P2)
          $\gamma_{3, 10} = 1$ (P4)
          
          From C11 = P1 - P2 + P3 + P6:
          $\gamma_{0, 11} = 1$ (P1)
          $\gamma_{1, 11} = -1$ (P2)
          $\gamma_{2, 11} = 1$ (P3)
          $\gamma_{5, 11} = 1$ (P6)
          
          All other $\gamma$ are 0.
          
          Let's verify the counts.
          Rank = 7 (t) * 3 (i1) * 3 (j1) * 3 (k1) = 189.
          
          We need to fill U, V, W arrays of size (36, 189).
          Index c = t * 27 + i1 * 9 + j1 * 3 + k1.
          
          Wait, the order of i1, j1, k1 in the rank index doesn't matter as long as it's consistent.
          
          Let's code this up.
          
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for A (U)
    # alpha[t][I1][J1]
    alpha = [
        [[1, 0], [0, 1]],  # t=0: A00+A11
        [[0, 0], [1, 1]],  # t=1: A10+A11
        [[1, 0], [0, 0]],  # t=2: A00
        [[0, 0], [0, 1]],  # t=3: A11
        [[1, 1], [0, 0]],  # t=4: A00+A01
        [[-1, 0], [1, 0]], # t=5: A10-A00
        [[0, 1], [0, -1]], # t=6: A01-A11
    ]
    
    # Coefficients for B (V)
    # beta[t][J1][K1]
    beta = [
        [[1, 0], [0, 1]],  # t=0: B00+B11
        [[1, 0], [0, 0]],  # t=1: B00
        [[0, 1], [0, -1]], # t=2: B01-B11
        [[-1, 0], [1, 0]], # t=3: B10-B00
        [[0, 0], [0, 1]],  # t=4: B11
        [[1, 1], [0, 0]],  # t=5: B00+B01
        [[0, 1], [1, 1]],  # t=6: B10+B11
    ]
    
    # Coefficients for C (W)
    # gamma[t][I1][K1]
    gamma = [
        [[1, 0], [0, 1]],  # t=0: C00(1), C11(1)
        [[0, 0], [1, -1]], # t=1: C10(1), C11(-1)
        [[0, 1], [0, 1]],  # t=2: C01(1), C11(1)
        [[1, 0], [1, 0]],  # t=3: C00(1), C10(1)
        [[-1, 1], [0, 0]], # t=4: C00(-1), C01(1)
        [[0, 0], [0, 1]],  # t=5: C11(1)
        [[1, 0], [0, 0]],  # t=6: C00(1)
    ]
    
    # Fill U, V, W
    # c = t * 27 + i1 * 9 + j1 * 3 + k1
    for t in range(7):
        for i1 in range(3):
            for j1 in range(3):
                for k1 in range(3):
                    c = t * 27 + i1 * 9 + j1 * 3 + k1
                    
                    # U: Index I*m + J. I = 3*I1 + i1, J = 3*J1 + j1
                    for I1 in range(2):
                        for J1 in range(2):
                            coeff = alpha[t][I1][J1]
                            if coeff != 0:
                                I = 3 * I1 + i1
                                J = 3 * J1 + j1
                                U[I * m + J][c] = coeff
                                
                    # V: Index J*p + K. J = 3*J1 + j1, K = 3*K1 + k1
                    for J1 in range(2):
                        for K1 in range(2):
                            coeff = beta[t][J1][K1]
                            if coeff != 0:
                                J = 3 * J1 + j1
                                K = 3 * K1 + k1
                                V[J * p + K][c] = coeff
                                
                    # W: Index K*n + I. K = 3*K1 + k1, I = 3*I1 + i1
                    for I1 in range(2):
                        for K1 in range(2):
                            coeff = gamma[t][I1][K1]
                            if coeff != 0:
                                K = 3 * K1 + k1
                                I = 3 * I1 + i1
                                W[K * n + I][c] = coeff
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
