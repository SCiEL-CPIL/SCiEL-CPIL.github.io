import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    # W indices: W[k*n+i, r] -> W[0,0]=0, W[0,1]=1, W[1,0]=3, W[1,1]=4
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1)
    add_term(4, 4, 0, 1, 1, 1)
    # M1 contributes to C00, C11?
    # C00 += M1, C11 += M1
    # We need to accumulate sums. 
    # The interface returns U, V, W such that C = U^T V? No.
    # The decomposition is T = sum U_r \otimes V_r \otimes W_r.
    # C[i,j] = sum_r U[i,j,r] V[?,r] W[?,r]?
    # Standard tensor decomposition:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Here T is the matrix multiplication tensor.
    # T_{i, j, k} corresponds to A_{ij} B_{jk} -> C_{ik}.
    # Wait, the indices in the problem description:
    # U shape (n*m, rank) -> U[i*m+j, r]
    # V shape (m*p, rank) -> V[j*p+k, r]
    # W shape (p*n, rank) -> W[k*n+i, r]
    # So T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This matches C_{ik} = sum_j A_{ij} B_{jk}.
    
    # So for each term r, we specify which A index, B index, C index it contributes to.
    # But Strassen combines multiple A indices into one factor.
    # e.g., (A00+A11) is a linear combination.
    # So U[0, r] = 1, U[4, r] = 1.
    # V[0, r] = 1, V[4, r] = 1.
    # W[0, r] = 1 (for C00), W[4, r] = 1 (for C11)?
    # No, one term r corresponds to ONE product of linear forms.
    # The product is (Sum A) * (Sum B).
    # This product contributes to MULTIPLE C entries.
    # So for a single rank-1 term r, U has non-zeros at several A indices, V at several B indices, W at several C indices.
    
    # Let's re-evaluate.
    # A rank-1 term is $u \otimes v \otimes w$.
    # $u$ is a vector in $\mathbb{R}^{9}$ (A indices).
    # $v$ is a vector in $\mathbb{R}^{9}$ (B indices).
    # $w$ is a vector in $\mathbb{R}^{9}$ (C indices).
    # The term is $\sum_{i,j,k} u_{ij} v_{jk} w_{ki}$.
    # This equals $(\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk}) (\sum w_{ki} C_{ki})$? No.
    # The tensor is $T = \sum_{r} u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$.
    # $T_{ijk} = \sum_r u^{(r)}_i v^{(r)}_j w^{(r)}_k$.
    # In matrix mult: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So $T_{i, j, k}$ is 1 if we are computing $C_{ik}$ using $A_{ij} B_{jk}$?
    # No, the tensor entries are 1 for the identity map.
    # $T_{i, j, k}$ is the coefficient of $A_{ij} B_{jk}$ in the expansion of $C_{ik}$?
    # Actually, the tensor is defined by $T_{i, j, k} = \delta_{i, k'} \delta_{j, j'}$?
    # Let's stick to the factor matrices.
    # We need $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ if the indices form a valid path $i \to j \to k$ in the multiplication?
    # No. The tensor for matrix multiplication $M_{n,m,p}$ has $T_{i, j, k} = 1$ if we consider the standard basis.
    # Actually, the condition is:
    # $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i,j,k$?
    # No. The tensor $T$ satisfies $T(A,B) = C$.
    # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # This is a bilinear map.
    # The tensor decomposition is for the trilinear form?
    # No, matrix multiplication is a bilinear map, but can be viewed as a tensor.
    # The tensor $T \in \mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{p \times n}$?
    # Usually defined on $\mathbb{R}^{n \times m} \times \mathbb{R}^{m \times p} \to \mathbb{R}^{n \times p}$.
    # The tensor is in $\mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes (\mathbb{R}^{n \times p})^*$.
    # Entry $T_{ij, jk, ik'}$?
    # The problem states:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the decomposition is exact.
    # This implies $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$?
    # Let's check the schoolbook.
    # $U[i*m+j, c] = 1$ for $c = i*m*p + j*p + k$.
    # $V[j*p+k, c] = 1$ for same c.
    # $W[k*n+i, c] = 1$ for same c.
    # Sum over r: Only one r is non-zero for a given (i,j,k).
    # The term is $1 \cdot 1 \cdot 1 = 1$.
    # So yes, the condition is $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i,j,k$.
    
    # So for Strassen term $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # This term contributes to $C_{00}$ and $C_{11}$.
    # Specifically, $M_1$ is added to $C_{00}$ and $C_{11}$.
    # So for this rank-1 term r:
    # $U[0, r] = 1$ ($A_{00}$)
    # $U[4, r] = 1$ ($A_{11}$)
    # $V[0, r] = 1$ ($B_{00}$)
    # $V[4, r] = 1$ ($B_{11}$)
    # $W[0, r] = 1$ ($C_{00}$, index $k=0, i=0 \to 0*3+0=0$)
    # $W[4, r] = 1$ ($C_{11}$, index $k=1, i=1 \to 1*3+1=4$)
    
    # Wait, the product is $(A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
    # In the sum $\sum_{j} A_{ij} B_{jk}$:
    # For $C_{00}$ ($i=0, k=0$): terms are $A_{00}B_{00} + A_{01}B_{10}$.
    # $A_{00}B_{00}$ is in $M_1$.
    # For $C_{11}$ ($i=1, k=1$): terms are $A_{10}B_{01} + A_{11}B_{11}$.
    # $A_{11}B_{11}$ is in $M_1$.
    # The cross terms $A_{00}B_{11}$ and $A_{11}B_{00}$ do NOT appear in $C_{00}$ or $C_{11}$ directly as single products?
    # $A_{00}B_{11}$ would contribute to $C_{0, 1}$? No, $B_{11}$ is $j=1, k=1$. $A_{00}$ is $i=0, j=0$. Mismatch j.
    # So $A_{00}B_{11}$ is 0 in the sum $\sum_j A_{ij} B_{jk}$ because $j$ must match.
    # So the product $(A_{00}+A_{11})(B_{00}+B_{11})$ contains "garbage" terms $A_{00}B_{11}$ and $A_{11}B_{00}$.
    # These garbage terms must be cancelled by other terms in the decomposition.
    # This is how Strassen works.
    
    # So we just need to list the 7 Strassen terms with their coefficients in U, V, W.
    
    # Strassen Terms for 2x2 (indices 0,1,2,3 for A, B, C relative to 2x2 block)
    # A: 00, 01, 10, 11 -> 0, 1, 3, 4 in 3x3
    # B: 00, 01, 10, 11 -> 0, 1, 3, 4 in 3x3
    # C: 00, 01, 10, 11 -> 0, 1, 3, 4 in 3x3 (W index k*3+i)
    
    # M1: (A00+A11)(B00+B11)
    # U: 0, 4. V: 0, 4. W: 0, 4.
    # Coeffs: 1, 1. 1, 1. 1, 1.
    
    # M2: (A10+A11)B00
    # U: 3, 4. V: 0. W: 3.
    # Coeffs: 1, 1. 1. 1.
    
    # M3: A00(B01-B11)
    # U: 0. V: 1, 4. W: 1.
    # Coeffs: 1. 1, -1. 1.
    
    # M4: A11(B10-B11) -- Wait, standard Strassen:
    # M4 = A11(B10 - B11)? No.
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's use this standard set.
    
    def add_strassen_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] = val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] = val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] = val
        col += 1

    # 1. M1 = (A00+A11)(B00+B11)
    add_strassen_term([0, 4], [1, 1], [0, 4], [1, 1], [0, 4], [1, 1])
    
    # 2. M2 = (A10+A11)B00
    add_strassen_term([3, 4], [1, 1], [0], [1], [3], [1])
    
    # 3. M3 = A00(B01-B11)
    add_strassen_term([0], [1], [1, 4], [1, -1], [1], [1])
    
    # 4. M4 = A11(B10-B00)
    add_strassen_term([4], [1], [3, 0], [1, -1], [4], [1])
    
    # 5. M5 = (A00+A01)B11
    add_strassen_term([0, 1], [1, 1], [4], [1], [4], [1])
    
    # 6. M6 = (A10-A00)(B00+B01)
    add_strassen_term([3, 0], [1, -1], [0, 1], [1, 1], [3], [1])
    
    # 7. M7 = (A01-A11)(B10+B11)
    add_strassen_term([1, 4], [1, -1], [3, 4], [1, 1], [1], [1])
    
    # Now the other 17 terms (Rank 4+4+4+2+2+1 = 17)
    
    # T2: A[0:2, 2] B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2 (0,2), 5 (1,2). B indices: 6 (2,0), 7 (2,1). C indices: 0, 1, 3, 4.
    # Outer product. 4 terms.
    # A02 * B20 -> C00
    add_term(2, 6, 0, 1, 1, 1)
    # A02 * B21 -> C01
    add_term(2, 7, 1, 1, 1, 1)
    # A12 * B20 -> C10
    add_term(5, 6, 3, 1, 1, 1)
    # A12 * B21 -> C11
    add_term(5, 7, 4, 1, 1, 1)
    
    # T3: A[0:2, 0:2] B[0:2, 2] -> C[0:2, 2]
    # A 2x2, B 2x1. Rank 4.
    # A00 B02 -> C02
    add_term(0, 2, 6, 1, 1, 1) # W index k=2, i=0 -> 2*3+0=6
    # A01 B12 -> C02
    add_term(1, 5, 6, 1, 1, 1) # W index 6
    # A10 B02 -> C12
    add_term(3, 2, 8, 1, 1, 1) # W index k=2, i=1 -> 2*3+1=8
    # A11 B12 -> C12
    add_term(4, 5, 8, 1, 1, 1) # W index 8
    
    # T4: A[2, 0:2] B[0:2, 0:2] -> C[2, 0:2]
    # A 1x2, B 2x2. Rank 4.
    # A20 B00 -> C20
    add_term(6, 0, 9, 1, 1, 1) # W index k=0, i=2 -> 0*3+2=9
    # A21 B10 -> C20
    add_term(7, 3, 9, 1, 1, 1)
    # A20 B01 -> C21
    add_term(6, 1, 10, 1, 1, 1) # W index k=1, i=2 -> 1*3+2=10
    # A21 B11 -> C21
    add_term(7, 4, 10, 1, 1, 1)
    
    # T5: A[0:2, 2] B[2, 2] -> C[0:2, 2]
    # A 2x1, B 1x1. Rank 2.
    # A02 B22 -> C02
    add_term(2, 8, 6, 1, 1, 1)
    # A12 B22 -> C12
    add_term(5, 8, 8, 1, 1, 1)
    
    # T6: A[2, 0:2] B[0:2, 2] -> C[2, 2]
    # A 1x2, B 2x1. Rank 2.
    # A20 B02 -> C22
    add_term(6, 2, 11, 1, 1, 1) # W index k=2, i=2 -> 2*3+2=11
    # A21 B12 -> C22
    add_term(7, 5, 11, 1, 1, 1)
    
    # T7: A[2, 2] B[2, 2] -> C[2, 2]
    # Rank 1.
    # A22 B22 -> C22
    add_term(8, 8, 11, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
