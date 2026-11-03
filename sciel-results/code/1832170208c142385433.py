import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build the decomposition block by block.
    # The output index is W[k*n + i, r] where k is col (0..6) and i is row (0..2).
    # The input indices are U[i*m + j, r] and V[j*p + k, r].
    
    # We partition the matrices into:
    # A (3x5) -> Rows 0,1 (2x5) and Row 2 (1x5)
    # B (5x7) -> Cols 0,1 (5x2) and Cols 2..6 (5x5)
    
    # However, Strassen works on 2x2 blocks.
    # Let's decompose the 3x5 by 5x7 multiplication into independent 2x2 and 1x2 / 2x1 / 1x1 blocks.
    
    # Blocks of A (rows) and B (cols):
    # A is 3x5. B is 5x7.
    # Let's split A rows into [0:2] and [2:3].
    # Let's split B cols into [0:2], [2:4], [4:7].
    # This creates 6 blocks of A and 3 blocks of B.
    # This is getting complicated to map manually to a single global rank.
    
    # Simpler approach:
    # Use Strassen for the top-left 2x2 part of the output?
    # Output C is 3x7.
    # Let's partition C into:
    # C[0:2, 0:2] (2x2) -> Requires A[0:2, :] and B[:, 0:2]
    # C[0:2, 2:4] (2x2) -> Requires A[0:2, :] and B[:, 2:4]
    # C[0:2, 4:7] (2x3) -> Requires A[0:2, :] and B[:, 4:7]
    # C[2:3, 0:2] (1x2) -> Requires A[2:3, :] and B[:, 0:2]
    # C[2:3, 2:4] (1x2) -> Requires A[2:3, :] and B[:, 2:4]
    # C[2:3, 4:7] (1x3) -> Requires A[2:3, :] and B[:, 4:7]
    
    # This partitioning doesn't fully exploit Strassen because the inputs A and B are shared across blocks.
    # The rank of the tensor is not simply the sum of ranks of sub-tensors if they share input dimensions.
    # However, if we view the multiplication as a sum of independent tensor products, we can just sum the ranks if the supports are disjoint in the "term" space.
    # Actually, the rank of a tensor is the minimum number of rank-1 tensors to sum to it.
    # The standard schoolbook is a sum of 105 rank-1 tensors.
    # Strassen's algorithm for 2x2 matrix multiplication reduces the rank from 8 to 7.
    # If we can partition the problem into independent 2x2x2 subproblems, we save 1 rank per 8.
    
    # Can we partition 3x5 by 5x7 into independent 2x2x2 problems?
    # No, because the middle dimension (5) is shared.
    # However, we can use a "block Strassen" approach if we pad or if we structure the decomposition carefully.
    
    # Let's try a different perspective:
    # The rank of the matrix multiplication tensor $M_{n,m,p}$ is known to be at most $nmp - 1$ for some cases, but generally close to $nmp$.
    # For 3x5 by 5x7, the schoolbook rank is 105.
    # Is there a known lower rank?
    # If we treat the multiplication as:
    # C = A B
    # We can write A as 3 blocks of rows (heights 2,1) and B as 7 blocks of cols (widths 1..1)? No.
    
    # Let's look at the "Strassen-like" decomposition for rectangular matrices.
    # If we split A into $A_1, A_2$ (2x5 and 1x5) and B into $B_1, B_2$ (5x2 and 5x5)?
    # $C_{11} = A_1 B_1$ (2x2)
    # $C_{12} = A_1 B_2$ (2x5)
    # $C_{21} = A_2 B_1$ (1x2)
    # $C_{22} = A_2 B_2$ (1x5)
    
    # This decomposition is not independent because $A_1$ is used in $C_{11}$ and $C_{12}$, and $B_1$ is used in $C_{11}$ and $C_{21}$.
    # Therefore, we cannot simply sum the ranks of the subproblems.
    
    # However, we can use the fact that we can compute specific 2x2 blocks of the output using Strassen's 7 multiplications, provided we have access to the specific 2x2 sub-matrices of A and B.
    # But A and B are dense.
    
    # Let's try a specific construction:
    # We will compute the output C using a combination of Strassen steps for 2x2 blocks and standard steps for the rest.
    # To do this, we need to define the "rank-1 terms" explicitly.
    
    # Let's attempt to find a rank 99 decomposition by replacing 6 of the 2x2x2 sub-blocks (each of rank 8) with Strassen (rank 7).
    # This requires that these 6 sub-blocks are "independent" in the sense that their inputs do not interfere in a way that forces a higher rank.
    # Actually, if we just sum the rank-1 tensors, the total rank is the sum of the ranks of the components.
    # If we decompose the tensor $T$ into $T_1 + T_2 + ... + T_k$, and each $T_i$ has rank $r_i$, then $rank(T) \le \sum r_i$.
    # We can partition the 105 terms of schoolbook into groups.
    # Group 1: Terms contributing to C[0:2, 0:2]. These depend on A[0:2, :] and B[:, 0:2].
    # Group 2: Terms contributing to C[0:2, 2:4]. These depend on A[0:2, :] and B[:, 2:4].
    # ...
    
    # The problem is that the "inputs" A and B are the same tensors.
    # The tensor $M_{3,5,7}$ is not a direct sum of $M_{2,5,2}$, etc., because the 5-dimension is shared.
    # $M_{n,m,p}$ is isomorphic to a subspace of $M_{N,M,P}$?
    
    # Let's use a known result:
    # The rank of $M_{n,m,p}$ is at most $nmp - 1$ if $n,m,p \ge 2$? No, that's not generally true.
    # For $M_{2,2,2}$, rank is 7 (Strassen). Schoolbook is 8.
    # For $M_{3,3,3}$, rank is at most 23 (Strassen-like). Schoolbook is 27.
    
    # Strategy:
    # We will construct the decomposition by "patching" Strassen blocks into the schoolbook structure.
    # We identify disjoint 2x2x2 sub-tensors in the 3x5x7 tensor.
    # A 2x2x2 sub-tensor corresponds to selecting 2 rows of A, 2 cols of B, and the resulting 2x2 block of C.
    # However, the "middle" dimension (5) is the sum.
    # The term for $C_{ik}$ is $\sum_j A_{ij} B_{jk}$.
    # The tensor is $\sum_{i,j,k} E_{ik} \otimes E_{ji} \otimes E_{kj}$? No.
    # The tensor is $\sum_{i,j,k} E_{ik}^{(C)} \otimes E_{ij}^{(A)} \otimes E_{jk}^{(B)}$.
    
    # If we select a set of indices $I, J, K$ such that the sub-tensor $T_{I,J,K}$ is a "pure" $M_{|I|,|J|,|K|}$ tensor, and if these sub-tensors are orthogonal (disjoint supports in the tensor product space), we can sum their ranks.
    # The supports are disjoint if we partition the indices $i, k$ (output) and $j$ (input) such that each term $(i,j,k)$ belongs to exactly one group.
    # But $j$ is shared. We cannot partition $j$ because $A_{ij}$ and $B_{jk}$ are coupled by $j$.
    # Wait, the tensor is a sum over $j$.
    # $T = \sum_{j=0}^{4} T^{(j)}$ where $T^{(j)}$ is the tensor for the $j$-th column of A and $j$-th row of B.
    # $T^{(j)}$ has rank 1? No.
    # $T^{(j)} = \sum_{i,k} E_{ik} \otimes E_{ij} \otimes E_{jk}$.
    # This is a rank 1 tensor?
    # $E_{ij} \otimes E_{jk}$ is a vector in $A \otimes B$.
    # The map is linear in $A$ and $B$.
    # The tensor is $T(A,B) = \sum_{i,j,k} A_{ij} B_{jk} E_{ik}$.
    # This is a bilinear map.
    # The rank of this map is the tensor rank.
    
    # We can rewrite the sum:
    # $T = \sum_{i,k} E_{ik} \otimes ( \sum_j E_{ij} \otimes E_{jk} )$
    # The term in the parenthesis is a matrix $M^{(i,k)}$ in the space $A \otimes B$.
    # Actually, it's simpler to view it as a sum of rank-1 tensors $u \otimes v \otimes w$.
    # Schoolbook: $u_{ij}=1$ if $(i,j)$ matches, $v_{jk}=1$ if $(j,k)$ matches, $w_{ik}=1$ if $(i,k)$ matches.
    # Rank 105.
    
    # Can we replace 6 groups of 8 terms with 7 terms?
    # This requires finding 6 disjoint sets of indices $\{i_1, i_2\}, \{k_1, k_2\}$ and a corresponding set of $j$'s such that the sub-tensor is isomorphic to $M_{2,?,2}$.
    # The $j$ dimension is the "middle" one.
    # If we fix $i \in \{0,1\}$ and $k \in \{0,1\}$, the terms involve all $j \in \{0..4\}$.
    # The sub-tensor for $i \in \{0,1\}, k \in \{0,1\}$ uses the full $j$ range.
    # This sub-tensor is $M_{2,5,2}$.
    # What is the rank of $M_{2,5,2}$?
    # It is at most $2 \times 5 \times 2 = 20$.
    # Can we do better?
    # $M_{2,5,2}$ can be seen as computing $2 \times 2$ output from $2 \times 5$ and $5 \times 2$.
    # This is equivalent to multiplying two $2 \times 5$ matrices? No.
    # It is $A_{2 \times 5} B_{5 \times 2}$.
    # We can use Strassen if we can decompose the 5 into 2+2+1?
    # If we split the 5 into $2+2+1$, we have:
    # $A = [A_1 (2x2), A_2 (2x2), A_3 (2x1)]$
    # $B = [B_1 (2x2), B_2 (2x2), B_3 (2x1)]^T$? No, B is 5x2.
    # $B = [B_1 (2x2), B_2 (2x2), B_3 (1x2)]$
    # $C = A_1 B_1 + A_2 B_2 + A_3 B_3$.
    # $A_1 B_1$ is 2x2 mult (rank 7).
    # $A_2 B_2$ is 2x2 mult (rank 7).
    # $A_3 B_3$ is 2x1 by 1x2 mult (rank 2).
    # Total rank for this block: $7 + 7 + 2 = 16$.
    # Schoolbook for this block: $2 \times 5 \times 2 = 20$.
    # Saving: 4.
    
    # We have 3 such "2x2 output" blocks in the 3x7 output?
    # Output C is 3x7.
    # We can partition the output indices $i \in \{0,1,2\}$ and $k \in \{0..6\}$.
    # Let's partition $i$ into $\{0,1\}$ and $\{2\}$.
    # Let's partition $k$ into $\{0,1\}, \{2,3\}, \{4,5\}, \{6\}$.
    
    # Block 1: $i \in \{0,1\}, k \in \{0,1\}$. Size 2x2. Input $j \in \{0..4\}$.
    #   Decomposition: Split $j$ into $0,1$ (2), $2,3$ (2), $4$ (1).
    #   Rank: $7 (Strassen) + 7 (Strassen) + 2 (Schoolbook) = 16$.
    #   Schoolbook: 20.
    
    # Block 2: $i \in \{0,1\}, k \in \{2,3\}$. Size 2x2. Input $j \in \{0..4\}$.
    #   Same structure. Rank 16.
    
    # Block 3: $i \in \{0,1\}, k \in \{4,5\}$. Size 2x2. Input $j \in \{0..4\}$.
    #   Same structure. Rank 16.
    
    # Block 4: $i \in \{0,1\}, k \in \{6\}$. Size 2x1. Input $j \in \{0..4\}$.
    #   This is $A_{2 \times 5} B_{5 \times 1}$.
    #   Split $j$ into $0,1$ (2), $2,3$ (2), $4$ (1).
    #   $A_1 B_1$ (2x2 by 2x1? No, $B_1$ is 2x1 part of the 5x1 vector).
    #   Wait, $B$ is 5x1 here. $B_1$ is 2x1. $A_1$ is 2x2.
    #   $A_1 B_1$ is 2x2 by 2x1 -> 2x1. Rank 2 (schoolbook).
    #   $A_2 B_2$ is 2x2 by 2x1 -> 2x1. Rank 2.
    #   $A_3 B_3$ is 2x1 by 1x1 -> 2x1. Rank 2.
    #   Total Rank: 6.
    #   Schoolbook: $2 \times 5 \times 1 = 10$.
    
    # Block 5: $i \in \{2\}, k \in \{0,1\}$. Size 1x2. Input $j \in \{0..4\}$.
    #   $A_{1 \times 5} B_{5 \times 2}$.
    #   Split $j$ into $0,1$ (2), $2,3$ (2), $4$ (1).
    #   $A_1 B_1$ (1x2 by 2x2). Rank 2.
    #   $A_2 B_2$ (1x2 by 2x2). Rank 2.
    #   $A_3 B_3$ (1x1 by 1x2). Rank 2.
    #   Total Rank: 6.
    #   Schoolbook: $1 \times 5 \times 2 = 10$.
    
    # Block 6: $i \in \{2\}, k \in \{2,3\}$. Size 1x2.
    #   Same as Block 5. Rank 6.
    
    # Block 7: $i \in \{2\}, k \in \{4,5\}$. Size 1x2.
    #   Same as Block 5. Rank 6.
    
    # Block 8: $i \in \{2\}, k \in \{6\}$. Size 1x1.
    #   $A_{1 \times 5} B_{5 \times 1}$.
    #   Split $j$ into $0,1$ (2), $2,3$ (2), $4$ (1).
    #   $A_1 B_1$ (1x2 by 2x1). Rank 2.
    #   $A_2 B_2$ (1x2 by 2x1). Rank 2.
    #   $A_3 B_3$ (1x1 by 1x1). Rank 1.
    #   Total Rank: 5.
    #   Schoolbook: $1 \times 5 \times 1 = 5$.
    
    # Total Rank:
    # 16 + 16 + 16 + 6 + 6 + 6 + 6 + 5 = 77.
    
    # Wait, is this valid?
    # The decomposition is a sum of independent tensor components.
    # The indices $i$ and $k$ are partitioned disjointly.
    # The index $j$ is shared, but we are summing the contributions for each $j$ partition.
    # Specifically, for a fixed block (e.g., $i \in \{0,1\}, k \in \{0,1\}$), we decomposed the sum over $j$ into sums over $j \in \{0,1\}, \{2,3\}, \{4\}$.
    # These $j$ ranges are disjoint across the "sub-blocks" of the $j$ dimension within a fixed $(i,k)$ block.
    # However, do the $j$ ranges overlap between different $(i,k)$ blocks?
    # Yes, $j \in \{0,1\}$ is used in Block 1, Block 2, Block 3, Block 4, Block 5, etc.
    # Does this cause a problem?
    # The tensor is $T = \sum_{blocks} T_{block}$.
    # $T_{block}$ is supported on $I_{block} \times J \times K_{block}$.
    # Since $I_{block}$ and $K_{block}$ are disjoint for different blocks, the supports of $T_{block}$ are disjoint in the $I \times K$ space.
    # The tensor product space is $I \otimes J \otimes K$.
    # If the supports in $I$ and $K$ are disjoint, the sub-tensors are orthogonal in the tensor product space?
    # Yes, because the basis vectors $E_{ik}$ are orthogonal.
    # Therefore, the rank of the sum is the sum of the ranks.
    
    # So the total rank is indeed 77.
    # This is a massive improvement over 105.
    
    # Let's implement this construction.
    
    rank_terms = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
        rank_terms.append((i, j, k, 1.0))
        
    # Helper to add a Strassen term for a 2x2 block
    # Inputs: i0, i1 (rows of A), k0, k1 (cols of B), j0, j1 (cols of A / rows of B)
    # Strassen computes C = A B where A is 2x2, B is 2x2.
    # A = [[A00, A01], [A10, A11]] corresponding to A[i0, j0], A[i0, j1], A[i1, j0], A[i1, j1]
    # B = [[B00, B01], [B10, B11]] corresponding to B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express these as rank-1 tensors.
    # Each $M_x$ is a product of a linear combination of A entries and a linear combination of B entries.
    # This results in multiple rank-1 terms.
    # E.g., M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
    # = A[i0,j0]B[j0,k0] + A[i0,j0]B[j1,k1] + A[i1,j1]B[j0,k0] + A[i1,j1]B[j1,k1]
    # Each of these is a rank-1 term in the tensor sum.
    # But wait, Strassen's algorithm reduces the *number of multiplications*, not necessarily the tensor rank of the *product* if we expand it back?
    # NO. The tensor rank of the multiplication map is the number of scalar multiplications required.
    # Strassen says we can compute the 2x2 product with 7 scalar multiplications.
    # This means the tensor $M_{2,2,2}$ has rank 7.
    # The 7 terms correspond to the 7 multiplications.
    # We don't need to expand them into 8 schoolbook terms.
    # We just need to define the 7 "rank-1" tensors that sum to $M_{2,2,2}$.
    
    # The 7 tensors for Strassen are:
    # T1: U = (1,0,1,0) [on A], V = (1,0,1,0) [on B], W = (1,0,0,1) [on C]?
    # Let's be precise.
    # A is vectorized (A00, A01, A10, A11).
    # B is vectorized (B00, B01, B10, B11).
    # C is vectorized (C00, C01, C10, C11).
    
    # M1 = (A00+A11)(B00+B11)
    # Term 1.1: A00 * B00 -> contributes to C00 (+), C01 (0), C10 (0), C11 (+) ?
    # No, M1 is an intermediate scalar.
    # The final C is a linear combination of M1..M7.
    # C00 = M1 + M4 - M5 + M7
    # This means the tensor for C00 is the sum of the tensors for M1, M4, -M5, M7.
    # The tensor for M1 is $u_1 \otimes v_1$ where $u_1$ is the vector for $(A00+A11)$ and $v_1$ is the vector for $(B00+B11)$.
    # But $u_1 \otimes v_1$ is a rank-1 tensor in $A \otimes B$.
    # When mapped to C, it contributes to specific entries.
    # Actually, the tensor $T$ is in $C \otimes A \otimes B$.
    # $T = \sum_{c} E_c \otimes ( \sum_{m} w_{c,m} u_m \otimes v_m )$
    # $= \sum_{m} ( \sum_{c} w_{c,m} E_c ) \otimes u_m \otimes v_m$
    # $= \sum_{m} W_m \otimes U_m \otimes V_m$
    # So yes, each Strassen multiplication corresponds to a rank-1 tensor in the decomposition!
    # The vector $W_m$ has entries $w_{c,m}$.
    # The vector $U_m$ has entries for A.
    # The vector $V_m$ has entries for B.
    
    # So for each 2x2 block, we add 7 terms to our global list.
    
    def add_strassen_block(i0, i1, k0, k1, j0, j1):
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C indices: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
        
        # U vectors (on A space, size nm)
        # V vectors (on B space, size mp)
        # W vectors (on C space, size pn)
        
        # We will create 7 terms.
        # Each term is (U_vec, V_vec, W_vec)
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1 (from C00=M1..., C11=...+M1)
        # Note: C01 and C10 do not use M1.
        
        def make_vec_A(i, j, val=1.0):
            vec = np.zeros(nm)
            vec[i*m + j] = val
            return vec
            
        def make_vec_B(j, k, val=1.0):
            vec = np.zeros(mp)
            vec[j*p + k] = val
            return vec
            
        def make_vec_C(i, k, val=1.0):
            vec = np.zeros(pn)
            vec[k*n + i] = val
            return vec
            
        # M1
        U1 = make_vec_A(i0, j0) + make_vec_A(i1, j1)
        V1 = make_vec_B(j0, k0) + make_vec_B(j1, k1)
        W1 = make_vec_C(i0, k0) + make_vec_C(i1, k1)
        rank_terms.append((U1, V1, W1))
        
        # M2 = (A10+A11)B00
        U2 = make_vec_A(i1, j0) + make_vec_A(i1, j1)
        V2 = make_vec_B(j0, k0)
        W2 = make_vec_C(i1, k0) # C10 = M2 + M4
        # Wait, C11 = M1 - M2 + ... so W2 should also have -1 at C11?
        # Let's check the Strassen formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 contributes to C00 (+1), C11 (+1)
        # M2 contributes to C10 (+1), C11 (-1)
        # M3 contributes to C01 (+1), C11 (+1)
        # M4 contributes to C00 (+1), C10 (+1)
        # M5 contributes to C01 (+1), C00 (-1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # Correcting W vectors:
        
        # M1
        W1 = make_vec_C(i0, k0, 1.0) + make_vec_C(i1, k1, 1.0)
        
        # M2
        W2 = make_vec_C(i1, k0, 1.0) + make_vec_C(i1, k1, -1.0)
        
        # M3 = A00(B10-B11)
        U3 = make_vec_A(i0, j0)
        V3 = make_vec_B(j1, k0) - make_vec_B(j1, k1)
        W3 = make_vec_C(i0, k1, 1.0) + make_vec_C(i1, k1, 1.0)
        
        # M4 = A11(B01-B00)
        U4 = make_vec_A(i1, j1)
        V4 = make_vec_B(j0, k1) - make_vec_B(j0, k0)
        W4 = make_vec_C(i0, k0, 1.0) + make_vec_C(i1, k0, 1.0)
        
        # M5 = (A00+A01)B11
        U5 = make_vec_A(i0, j0) + make_vec_A(i0, j1)
        V5 = make_vec_B(j1, k1)
        W5 = make_vec_C(i0, k1, 1.0) - make_vec_C(i0, k0, 1.0)
        
        # M6 = (A10-A00)(B00+B01)
        U6 = make_vec_A(i1, j0) - make_vec_A(i0, j0)
        V6 = make_vec_B(j0, k0) + make_vec_B(j0, k1)
        W6 = make_vec_C(i1, k1, 1.0)
        
        # M7 = (A01-A11)(B10+B11)
        U7 = make_vec_A(i0, j1) - make_vec_A(i1, j1)
        V7 = make_vec_B(j1, k0) + make_vec_B(j1, k1)
        W7 = make_vec_C(i0, k0, 1.0)
        
        rank_terms.append((U2, V2, W2))
        rank_terms.append((U3, V3, W3))
        rank_terms.append((U4, V4, W4))
        rank_terms.append((U5, V5, W5))
        rank_terms.append((U6, V6, W6))
        rank_terms.append((U7, V7, W7))
        
    # Helper to add a generic rank-1 term for schoolbook-like 1x1 or 1x2 blocks
    # We can just use the schoolbook terms for the non-2x2 blocks.
    # For 2x1, 1x2, 1x1, schoolbook is optimal (rank 2, 2, 1).
    
    # Block 1: i in {0,1}, k in {0,1}. j split 0,1 / 2,3 / 4
    # Strassen for j in {0,1}
    add_strassen_block(0, 1, 0, 1, 0, 1)
    # Strassen for j in {2,3}
    add_strassen_block(0, 1, 0, 1, 2, 3)
    # Schoolbook for j in {4}
    add_schoolbook(0, 4, 0)
    add_schoolbook(0, 4, 1)
    add_schoolbook(1, 4, 0)
    add_schoolbook(1, 4, 1)
    
    # Block 2: i in {0,1}, k in {2,3}
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_schoolbook(0, 4, 2)
    add_schoolbook(0, 4, 3)
    add_schoolbook(1, 4, 2)
    add_schoolbook(1, 4, 3)
    
    # Block 3: i in {0,1}, k in {4,5}
    add_strassen_block(0, 1, 4, 5, 0, 1)
    add_strassen_block(0, 1, 4, 5, 2, 3)
    add_schoolbook(0, 4, 4)
    add_schoolbook(0, 4, 5)
    add_schoolbook(1, 4, 4)
    add_schoolbook(1, 4, 5)
    
    # Block 4: i in {0,1}, k in {6}
    # Schoolbook for j 0..4
    for j in range(5):
        add_schoolbook(0, j, 6)
        add_schoolbook(1, j, 6)
        
    # Block 5: i in {2}, k in {0,1}
    for j in range(5):
        add_schoolbook(2, j, 0)
        add_schoolbook(2, j, 1)
        
    # Block 6: i in {2}, k in {2,3}
    for j in range(5):
        add_schoolbook(2, j, 2)
        add_schoolbook(2, j, 3)
        
    # Block 7: i in {2}, k in {4,5}
    for j in range(5):
        add_schoolbook(2, j, 4)
        add_schoolbook(2, j, 5)
        
    # Block 8: i in {2}, k in {6}
    for j in range(5):
        add_schoolbook(2, j, 6)
        
    # Total terms:
    # 3 blocks * 7 Strassen = 21
    # 3 blocks * 4 Schoolbook (for j=4) = 12
    # 1 block (2x1) * 10 Schoolbook = 10
    # 3 blocks (1x2) * 10 Schoolbook = 30
    # 1 block (1x1) * 5 Schoolbook = 5
    # Total: 21 + 12 + 10 + 30 + 5 = 78.
    
    # Wait, my previous calculation was 77.
    # Let's re-verify Block 4 (2x1).
    # I said Rank 6. Schoolbook 10.
    # My code adds 10 schoolbook terms for Block 4.
    # Can we do better for 2x1?
    # $A_{2 \times 5} B_{5 \times 1}$.
    # Split j into 2,2,1.
    # $A_1 B_1$ (2x2 by 2x1). Rank 2.
    # $A_2 B_2$ (2x2 by 2x1). Rank 2.
    # $A_3 B_3$ (2x1 by 1x1). Rank 2.
    # Total 6.
    # So we should use Strassen-like or just optimized schoolbook for these.
    # Actually, 2x2 by 2x1 is just 2 multiplications.
    # $C_0 = A_{00}B_0 + A_{01}B_1$
    # $C_1 = A_{10}B_0 + A_{11}B_1$
    # This is rank 2? No, it's 4 terms.
    # Can we do it in 3?
    # $M_1 = (A_{00}+A_{10})B_0$
    # $M_2 = (A_{01}+A_{11})B_1$
    # $M_3 = A_{00}B_1$?
    # No, 2x2 by 2x1 is just a dot product for each row.
    # Rank of $M_{2,2,1}$ is 2?
    # $C_0 = A_{00}B_0 + A_{01}B_1$
    # $C_1 = A_{10}B_0 + A_{11}B_1$
    # Tensor: $E_{0} \otimes (A_{00} \otimes B_0 + A_{01} \otimes B_1) + E_{1} \otimes (A_{10} \otimes B_0 + A_{11} \otimes B_1)$
    # This is not a sum of rank-1 tensors in the same way.
    # But we can just use the 4 schoolbook terms. It's small.
    # The main savings are in the 2x2 blocks.
    
    # Let's stick to the 78 rank construction for now, as it's a huge improvement over 105.
    # We can optimize the 2x1 and 1x2 blocks later if time permits.
    
    U_mat = np.zeros((nm, len(rank_terms)))
    V_mat = np.zeros((mp, len(rank_terms)))
    W_mat = np.zeros((pn, len(rank_terms)))
    
    for r, term in enumerate(rank_terms):
        if len(term) == 3:
            U_mat[:, r] = term[0]
            V_mat[:, r] = term[1]
            W_mat[:, r] = term[2]
        else:
            i, j, k, val = term
            U_mat[i*m+j, r] = val
            V_mat[j*p+k, r] = val
            W_mat[k*n+i, r] = val
            
    # Verify
    def verify(U, V, W, R):
        U_3d = np.array(U).reshape(n, m, R)
        V_3d = np.array(V).reshape(m, p, R)
        W_3d = np.array(W).reshape(p, n, R)
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            return np.allclose(T, 1.0, atol=1e-9)
        except:
            return False
            
    if verify(U_mat, V_mat, W_mat, len(rank_terms)):
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 105))
        V_school = np.zeros((mp, 105))
        W_school = np.zeros((pn, 105))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
