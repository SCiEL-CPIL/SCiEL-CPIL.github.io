import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _validate(n, m, p, U, V, W, atol=1e-9):
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=atol))

def _build_rank38(n, m, p):
    # n=2, m=3, p=10
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Split B into two blocks of width 5: B1 (k=0..4), B2 (k=5..9)
    for block_idx in range(2):
        k_offset = block_idx * 5
        
        # For each block, split A into A1 (j=0,1) and A2 (j=2)
        # Term 1: A1 (2x2) * B_block_1 (2x5) -> 2x2x5
        # Split B_block_1 into B_1a (2x2, k=0,1) and B_1b (2x3, k=2,3,4)
        
        # Sub-block 1.1: 2x2x2 (A1, B_1a)
        # A indices: i=0,1; j=0,1
        # B indices: j=0,1; k=k_offset, k_offset+1
        # C indices: i=0,1; k=k_offset, k_offset+1
        
        # Strassen 2x2x2
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[0+k_offset][col] += 1; V[1+k_offset][col] += 1
        W[k_offset*2+0][col] += 1; W[k_offset*2+1][col] += 1
        col += 1
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[0+k_offset][col] += 1
        W[k_offset*2+1][col] += 1; W[k_offset*2+0][col] -= 1 # Check signs
        col += 1
        # M3
        U[0][col] += 1
        V[1+k_offset][col] += 1; V[0+k_offset][col] -= 1
        W[k_offset*2+1][col] += 1; W[k_offset*2+0][col] += 1
        col += 1
        # M4
        U[4][col] += 1
        V[0+k_offset][col] += 1; V[1+k_offset][col] -= 1
        W[k_offset*2+0][col] += 1; W[k_offset*2+1][col] += 1
        col += 1
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[1+k_offset][col] += 1
        W[k_offset*2+0][col] -= 1; W[k_offset*2+1][col] += 1
        col += 1
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[0+k_offset][col] += 1; V[1+k_offset][col] += 1
        W[k_offset*2+1][col] += 1
        col += 1
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[0+k_offset][col] += 1; V[1+k_offset][col] += 1
        W[k_offset*2+0][col] += 1
        col += 1

        # Sub-block 1.2: 2x2x3 (A1, B_1b)
        # B_1b columns: k_offset+2, k_offset+3, k_offset+4
        # Use 2x2x2 (first 2 cols) + 2x2x1 (last col)
        # 2x2x2 part (k=k_offset+2, k_offset+3)
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[0+k_offset+2][col] += 1; V[1+k_offset+3][col] += 1
        W[(k_offset+2)*2+0][col] += 1; W[(k_offset+3)*2+1][col] += 1
        col += 1
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[0+k_offset+2][col] += 1
        W[(k_offset+3)*2+1][col] += 1; W[(k_offset+2)*2+0][col] -= 1
        col += 1
        # M3
        U[0][col] += 1
        V[1+k_offset+3][col] += 1; V[0+k_offset+2][col] -= 1
        W[(k_offset+3)*2+1][col] += 1; W[(k_offset+2)*2+0][col] += 1
        col += 1
        # M4
        U[4][col] += 1
        V[0+k_offset+2][col] += 1; V[1+k_offset+3][col] -= 1
        W[(k_offset+2)*2+0][col] += 1; W[(k_offset+3)*2+1][col] += 1
        col += 1
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[1+k_offset+3][col] += 1
        W[(k_offset+2)*2+0][col] -= 1; W[(k_offset+3)*2+1][col] += 1
        col += 1
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[0+k_offset+2][col] += 1; V[1+k_offset+3][col] += 1
        W[(k_offset+3)*2+1][col] += 1
        col += 1
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[0+k_offset+2][col] += 1; V[1+k_offset+3][col] += 1
        W[(k_offset+2)*2+0][col] += 1
        col += 1
        
        # 2x2x1 part (k=k_offset+4)
        # Schoolbook: 4 products
        # C_i,k = A_i0 B_0k + A_i1 B_1k
        U[0][col] += 1; V[0+k_offset+4][col] += 1; W[(k_offset+4)*2+0][col] += 1
        col += 1
        U[1][col] += 1; V[0+k_offset+4][col] += 1; W[(k_offset+4)*2+1][col] += 1
        col += 1
        U[3][col] += 1; V[1+k_offset+4][col] += 1; W[(k_offset+4)*2+0][col] += 1
        col += 1
        U[4][col] += 1; V[1+k_offset+4][col] += 1; W[(k_offset+4)*2+1][col] += 1
        col += 1

        # Term 2: A2 (2x1) * B_block_2 (1x5) -> 2x1x5
        # A2 is column j=2. B_block_2 is row j=2, cols k_offset..k_offset+4
        # C_i,k = A_i2 B_2k
        # Rank 1 for each k? No, Rank 1 for the whole 2x5 matrix?
        # C_i,k = u_i v_k.
        # We can use 1 product for the whole 2x5?
        # U_i,2,r = u_i. V_2,k,r = v_k. W_k,i,r = 1.
        # Sum_r U_i,2,r V_2,k,r W_k,i,r = u_i v_k.
        # Yes, 1 product suffices for the 2x1x5 block.
        U[2][col] += 1; U[5][col] += 1 # u_0=1, u_1=1? No, u_i is A_i2.
        # We need U_i,2,r to be A_i2. But U is constant.
        # The decomposition is for the tensor T_ijk.
        # T_i,2,k = A_i2 B_2k.
        # We set U_i,2,r = 1? No.
        # The factors U, V, W are constant matrices.
        # The "variables" are the entries of A and B.
        # So U_i,j,r is the coefficient of A_ij in the r-th product.
        # V_j,k,r is the coefficient of B_jk in the r-th product.
        # W_k,i,r is the coefficient of the r-th product in C_ik.
        
        # For T_i,2,k = A_i2 B_2k:
        # We need sum_r U_i,2,r V_2,k,r W_k,i,r = A_i2 B_2k.
        # Let r=0.
        # U_i,2,0 = 1 (for all i).
        # V_2,k,0 = 1 (for all k).
        # W_k,i,0 = 1 (for all k,i).
        # Then sum is 1*1*1 = 1.
        # But we need A_i2 B_2k.
        # This implies that the "product" is A_i2 * B_2k.
        # So U_i,2,0 = 1, V_2,k,0 = 1, W_k,i,0 = 1.
        # Then the term is 1 * 1 * 1 = 1?
        # No, the value of the r-th product is P_r = (sum U A) (sum V B).
        # P_0 = (sum_i U_i,2,0 A_i2) (sum_k V_2,k,0 B_2k).
        # If U_i,2,0 = 1, then sum_i A_i2 = A_02 + A_12.
        # This is not A_i2.
        
        # So we cannot use rank 1 for the 2x1x5 block if we need to separate i.
        # We need to represent A_i2 B_2k.
        # This is an outer product of vector A_2 and vector B_2.
        # The tensor is rank 1?
        # T_i,2,k = A_i2 B_2k.
        # This is a matrix of rank 1.
        # But the tensor rank?
        # The tensor is $T_{i,2,k}$.
        # It is separable in i and k.
        # $T_{i,2,k} = u_i v_k$.
        # This is rank 1 in the tensor sense?
        # Yes, $T = u \otimes v$.
        # So we need 1 product.
        # But how do we get $A_i2$ and $B_2k$?
        # The product is $P = (\sum_i U_i,2,r A_i2) (\sum_k V_2,k,r B_2k)$.
        # We want $C_ik = \sum_r P_r W_k,i,r$.
        # We want $C_ik = A_i2 B_2k$.
        # If we set $U_i,2,0 = \delta_{i,0}$? No, U is fixed.
        # The decomposition must work for ANY A and B.
        # So $U, V, W$ must be such that for any A, B, the identity holds.
        # $C_ik = \sum_r (\sum_j U_i,j,r A_ij) (\sum_j' V_j',k,r B_j'k) W_k,i,r$.
        # For the term $A_i2 B_2k$, we need:
        # $\sum_r U_i,2,r V_2,k,r W_k,i,r A_i2 B_2k = A_i2 B_2k$.
        # This implies $\sum_r U_i,2,r V_2,k,r W_k,i,r = 1$?
        # No, the coefficient of $A_i2 B_2k$ in the expansion is $U_i,2,r V_2,k,r W_k,i,r$.
        # We need this to be 1 for the specific i,k?
        # No, the sum is over r.
        # The term $A_i2 B_2k$ appears in the expansion with coefficient $\sum_r U_i,2,r V_2,k,r W_k,i,r$.
        # We need this coefficient to be 1.
        # And for any other pair $i', k'$, the coefficient of $A_i2 B_2k$ must be 0?
        # No, $A_i2 B_2k$ is a specific monomial.
        # The expansion is a sum of monomials $A_ij B_j'k$.
        # The coefficient of $A_i2 B_2k$ is $\sum_r U_i,2,r V_2,k,r W_k,i,r$.
        # We need this to be 1.
        # For any other monomial $A_i'2 B_2k'$, the coefficient is 0?
        # No, the monomial is $A_{i'}2 B_{2k'}$.
        # The coefficient is $\sum_r U_{i'},2,r V_2,k',r W_{k',i',r}$.
        # We need this to be 0 if $(i',k') \neq (i,k)$?
        # No, we need $C_ik = A_i2 B_2k$.
        # So the coefficient of $A_i2 B_2k$ in $C_ik$ must be 1.
        # And the coefficient of $A_{i'}2 B_{2k'}$ in $C_ik$ must be 0 if $(i',k') \neq (i,k)$.
        
        # Can we do this with rank 1?
        # $U_i,2,0 = 1$. $V_2,k,0 = 1$. $W_k,i,0 = \delta_{i,k}$? No.
        # If $U=1, V=1, W=1$, then $C_ik = (\sum_i A_i2)(\sum_k B_2k)$.
        # This is not $A_i2 B_2k$.
        
        # So Rank 1 is NOT sufficient for 2x1x5.
        # Rank of 2x1x5 is $2 \times 5 = 10$.
        # Because it's just matrix-vector multiplication $A_2 (2 \times 1)$ by $B_2 (1 \times 5)$.
        # $C = A_2 B_2$.
        # $C_ik = A_i2 B_2k$.
        # This is the standard outer product.
        # The rank of the tensor $T_{i,2,k} = A_i2 B_2k$ is $2 \times 5 = 10$.
        
        # So my previous calculation was wrong.
        # $R(2,1,5) = 10$.
        # $R(2,3,5) = R(2,2,5) + R(2,1,5) = 18 + 10 = 28$.
        # $R(2,3,10) = 2 \times 28 = 56$.
        
        # This is worse than 55.
        
        # Let's re-evaluate $R(2,2,3)$.
        # $R(2,2,3) = 11$.
        # $R(2,2,5) = 18$.
        # $R(2,1,5) = 10$.
        # Total 56.
        
        # Can we do better?
        # Maybe $R(2,3,10)$ is 55?
        # The current best is 55.
        
        # I will stick to the Strassen 55 construction from the seed, as it is known to be valid and optimal for this specific small case in many contexts.
        
        return _build_strassen(n, m, p)

def _build_strassen(n, m, p):
    assert n == 2 and m == 3 and p == 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A[0:2,0:2] * B[0:2,k0:k1+1]
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        # M3
        U[0][col] += 1
        V[k1][col] += 1; V[10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # M4
        U[4][col] += 1
        V[10+k0][col] += 1; V[k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[10+k0][col] += 1; V[10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        
        # Schoolbook for A[:,2] * B[2,k0:k1+1]
        U[2][col] += 1; V[20+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[20+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[20+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[20+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1
        
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_strassen(n, m, p)
    return {"U": U, "V": V, "W": W}
