"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U: (n*m, rank) = (6, 13)
    # V: (m*p, rank) = (9, 13)
    # W: (p*n, rank) = (6, 13)
    
    rank = 13
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # i, j for U (row i, col j) -> index i*m + j
    # j, k for V (row j, col k) -> index j*p + k
    # k, i for W (row k, col i) -> index k*n + i
    # Note: W index in task description is W[k*n+i, r]
    
    def add_term(i, j, k, u_val, v_val, w_val, col_idx):
        U[i * m + j][col_idx] += u_val
        V[j * p + k][col_idx] += v_val
        W[k * n + i][col_idx] += w_val

    col = 0
    
    # Strassen Terms for C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2]
    # A is 2x3, so A[0:2, 0:2] is rows 0,1 cols 0,1
    # B is 3x3, so B[0:2, 0:2] is rows 0,1 cols 0,1
    
    # M1 = (A00 + A11) * (B00 + B11)
    add_term(0, 0, 0, 1, 1, 1, col) # A00, B00 -> C00
    add_term(0, 0, 1, 1, 0, 0, col) # Wait, W is sum.
    # Let's do it properly.
    # Term 1: (A00 + A11) (B00 + B11)
    # Contributes to C00, C01, C10, C11 with coefficients.
    # It's a single rank-1 tensor in the product space, but expanded to the full space.
    # U vector: e_00 + e_11
    # V vector: e_00 + e_11
    # W vector: e_00 + e_10 + e_01 + e_11 ?
    # No, the rank-1 tensor is (u . a) (v . b) w.
    # So U has 1 at (0,0) and (1,1).
    # V has 1 at (0,0) and (1,1).
    # W has 1 at (0,0), (1,0), (0,1), (1,1)?
    # The product is (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This sum is assigned to W components.
    # C00 gets A00B00 + A00B11 + A11B00 + A11B11? No.
    # In Strassen, M1 is a scalar.
    # C00 = M1 - M4 + M7 - M5
    # So for the column corresponding to M1, we set W such that it adds to C00, C01, C10, C11?
    # No. The decomposition is $\sum_r U_r \otimes V_r \otimes W_r$.
    # The term for r=M1 is $U_{M1} \otimes V_{M1} \otimes W_{M1}$.
    # $U_{M1}$ has 1 at (0,0) and (1,1).
    # $V_{M1}$ has 1 at (0,0) and (1,1).
    # $W_{M1}$ should be such that when combined with other terms, it yields correct C.
    # Actually, in the decomposition, each rank-1 term contributes to ALL C elements?
    # No.
    # $C_{ik} = \sum_r U_{i, *} \cdot V_{*, k}$? No.
    # $C_{ik} = \sum_r U_{i, r} V_{r, k}$? No.
    # Standard form: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Tensor form: $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$?
    # Task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # Reconstruction: $C_{ik} = \sum_j \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$?
    # No, that would be $\sum_r (\sum_j U_{ij} V_{jk}) W_{ki}$.
    # Let's check the dimensions.
    # $U$ is $6 \times R$. $V$ is $9 \times R$. $W$ is $6 \times R$.
    # If $C_{ik} = \sum_r U_{i, r} V_{r, k} W_{k, i}$? No.
    
    # The standard decomposition of the multiplication tensor $T_{ijk}$ is:
    # $T_{ijk} = \sum_r U_{ij, r} V_{jk, r} W_{ki, r}$?
    # Usually, it is defined such that:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # If we use the tensor decomposition:
    # $\sum_r ( \sum_j U_{ij, r} A_{ij} ) ( \sum_j V_{jk, r} B_{jk} ) W_{ki, r}$?
    # No, the variables A and B are multiplied.
    # The rank-1 term is $u_r \otimes v_r \otimes w_r$.
    # The evaluation is $C = \sum_r (u_r \cdot A) (v_r \cdot B) w_r$?
    # Where $u_r$ is a vector of size $n \times m$?
    # Let's look at the shape of U: $(n \times m, rank)$.
    # So $U_{idx, r}$ is a coefficient for $A_{i,j}$.
    # $V_{idx, r}$ is a coefficient for $B_{j,k}$.
    # $W_{idx, r}$ is a coefficient for $C_{k,i}$? (Trace convention $W[k*n+i, r]$).
    # The formula is likely:
    # $C_{k,i} = \sum_r W[k*n+i, r] \times (\sum_{i',j'} U[i'*m+j', r] A_{i'j'}) \times (\sum_{j',k'} V[j'*p+k', r] B_{j'k'})$
    # This is not standard.
    # Standard is: $C_{ik} = \sum_r U_{i, r} V_{r, k}$ is for matrix multiplication of U and V.
    # Here we are decomposing the tensor of the operation.
    # $A_{ij} B_{jk} = C_{ik}$.
    # Rank-1 decomposition: $A_{ij} B_{jk} \approx \sum_r U_{ij, r} V_{jk, r}$.
    # Then $C_{ik}$ is obtained by summing over r?
    # Usually, the decomposition is:
    # $C_{ik} = \sum_r W_{ik, r} ( \sum_{j} U_{ij, r} A_{ij} ) ( \sum_{j} V_{jk, r} B_{jk} )$?
    # No, that would be quadratic in A and B.
    
    # Correct interpretation:
    # The tensor $T$ is defined by $T[i, j, k] = \delta_{jk}$? No.
    # The multiplication tensor $M_{n,m,p}$ has components $T[i, j, k] = 1$ if $j=j$?
    # $T[i, j, k]$ corresponds to the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So $T[i, j, k] = 1$ for all $i,j,k$? No.
    # The term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # So the tensor has non-zero entries at $(i, j, k)$?
    # Actually, the input dimensions are $N=n \times m$ and $P=m \times p$.
    # Output is $Q=p \times n$.
    # The tensor is in $\mathbb{R}^{N \times P \times Q}$.
    # Component $T[a, b, c]$ is 1 if $A_a$ and $B_b$ multiply to give $C_c$.
    # $A_a$ is $A_{i, j}$ where $a = i \cdot m + j$.
    # $B_b$ is $B_{j, k}$ where $b = j \cdot p + k$.
    # $C_c$ is $C_{k, i}$ where $c = k \cdot n + i$.
    # Note the index j must match.
    # So $T[a, b, c] = 1$ if $j_A = j_B$ and $k_B = k_C$ and $i_A = i_C$.
    # Basically, for each $(i,j,k)$, there is a 1 at index $(i \cdot m + j, j \cdot p + k, k \cdot n + i)$.
    
    # Rank-1 decomposition:
    # $T = \sum_r U_r \otimes V_r \otimes W_r$.
    # $U_r$ is a vector of length $n \cdot m$.
    # $V_r$ is a vector of length $m \cdot p$.
    # $W_r$ is a vector of length $p \cdot n$.
    
    # So, for a rank-1 term r:
    # It contributes to $C_c$ by $W_r[c] \times (\sum_a U_r[a] A_a) \times (\sum_b V_r[b] B_b)$.
    # This matches the Strassen logic:
    # $M_1 = (\sum U A) (\sum V B)$.
    # Then $C = \sum_r M_r W_r$.
    # i.e., $C_c = \sum_r W_r[c] M_r$.
    
    # So for each rank-1 term, we define U, V, W vectors.
    # And the total rank is the number of such vectors.
    
    # Let's implement the 13 terms.
    
    # Term 1: M1
    # U: A00, A11 -> indices 0, 2 (0*3+0=0, 1*3+1=4? No. n=2, m=3. i=0, j=0 -> 0. i=1, j=1 -> 1*3+1=4).
    # V: B00, B11 -> indices 0, 5 (0*3+0=0, 1*3+1=4? No. j=0, k=0 -> 0. j=1, k=1 -> 1*3+1=4).
    # W: C00, C10, C01, C11?
    # M1 contributes to C00, C01, C10, C11 with coefficients in the reconstruction formula.
    # C00 = M1 - M4 + M7 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for Term 1 (M1):
    # U: 1 at A00 (0), 1 at A11 (4).
    # V: 1 at B00 (0), 1 at B11 (4).
    # W: 1 at C00 (0*2+0=0). -1 at C11? No.
    # W vector should contain the coefficient for C.
    # $C_c = \sum_r W_r[c] M_r$.
    # So for r=1 (M1), W[0] = 1 (for C00). W[2] (C11, idx 1*2+1=3) = 1.
    # Wait, C11 is $i=1, k=1$. idx $1 \cdot 2 + 1 = 3$.
    # C00 is $i=0, k=0$. idx 0.
    # So W[0] = 1, W[3] = 1.
    # What about C01, C10?
    # C01 does not use M1.
    # C10 does not use M1.
    # So W[1] (C01, idx 0*2+1=1) = 0.
    # W[2] (C10, idx 1*2+0=2) = 0.
    
    # Let's verify indices:
    # C[i, k] -> idx k*n + i.
    # C00 -> 0
    # C01 -> 1
    # C10 -> 2
    # C11 -> 3
    # C02 -> 4
    # C12 -> 5
    
    # Term 1 (M1):
    # U: A00 (0), A11 (4) -> [1, 0, 0, 0, 1, 0]
    # V: B00 (0), B11 (4) -> [1, 0, 0, 0, 1, 0, 0, 0, 0]
    # W: C00 (0), C11 (3) -> [1, 0, 0, 1, 0, 0]
    
    # Term 2 (M2): (A10 + A11) B00
    # U: A10 (3), A11 (4)
    # V: B00 (0)
    # W: C10 (2) -> Coeff 1. C11 (3) -> Coeff -1.
    # Formula: C10 = M2 + M4. C11 = M1 - M2 + ...
    # So W[2] = 1, W[3] = -1.
    
    # Term 3 (M3): A00 (B01 - B11)
    # U: A00 (0)
    # V: B01 (1), -B11 (4)
    # W: C01 (1) -> 1. C11 (3) -> 1.
    # Formula: C01 = M3 + M5. C11 = ... + M3 ...
    
    # Term 4 (M4): A11 (B10 - B00)
    # U: A11 (4)
    # V: B10 (3), -B00 (0)
    # W: C00 (0) -> -1. C10 (2) -> 1.
    # Formula: C00 = M1 - M4 ... C10 = M2 + M4.
    
    # Term 5 (M5): (A00 + A01) B11
    # U: A00 (0), A01 (1)
    # V: B11 (4)
    # W: C00 (0) -> -1. C01 (1) -> 1.
    # Formula: C00 = ... - M5. C01 = M3 + M5.
    
    # Term 6 (M6): (A10 - A00) (B00 + B01)
    # U: A10 (3), -A00 (0)
    # V: B00 (0), B01 (1)
    # W: C11 (3) -> 1.
    # Formula: C11 = ... + M6.
    
    # Term 7 (M7): (A01 - A11) (B10 + B11)
    # U: A01 (1), -A11 (4)
    # V: B10 (3), B11 (4)
    # W: C00 (0) -> 1.
    # Formula: C00 = M1 - M4 + M7 ...
    
    # Terms 8-13 (Tail):
    # T8: A00 B02 -> C02 (4)
    # U: A00 (0)
    # V: B02 (2)
    # W: C02 (4) -> 1
    
    # T9: A01 B12 -> C02 (4)
    # U: A01 (1)
    # V: B12 (5)
    # W: C02 (4) -> 1
    
    # T10: A02 B22 -> C02 (4)
    # U: A02 (2)
    # V: B22 (8)
    # W: C02 (4) -> 1
    
    # T11: A10 B02 -> C12 (5)
    # U: A10 (3)
    # V: B02 (2)
    # W: C12 (5) -> 1
    
    # T12: A11 B12 -> C12 (5)
    # U: A11 (4)
    # V: B12 (5)
    # W: C12 (5) -> 1
    
    # T13: A12 B22 -> C12 (5)
    # U: A12 (5)
    # V: B22 (8)
    # W: C12 (5) -> 1

    # Fill matrices
    # U shape (6, 13)
    # V shape (9, 13)
    # W shape (6, 13)
    
    # Term 1
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Term 2
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1
    
    # Term 3
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # Term 4
    U[4][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += -1; W[2][3] += 1
    
    # Term 5
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # Term 6
    U[3][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Term 7
    U[1][6] += 1; U[4][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Term 8
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # Term 9
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # Term 10
    U[2][9] += 1
    V[8][9] += 1
    W[4][9] += 1
    
    # Term 11
    U[3][10] += 1
    V[2][10] += 1
    W[5][10] += 1
    
    # Term 12
    U[4][11] += 1
    V[5][11] += 1
    W[5][11] += 1
    
    # Term 13
    U[5][12] += 1
    V[8][12] += 1
    W[5][12] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
