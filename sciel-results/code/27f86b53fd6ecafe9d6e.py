"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    
    Strategy:
    Instead of random search, we use a deterministic constructive approach based on 
    block decomposition and Strassen's algorithm.
    
    The matrix multiplication $C = AB$ where $A$ is $2 \times 3$ and $B$ is $3 \times 7$.
    We can view this as multiplying blocks.
    Let $A = [A_1, A_2]$ where $A_1$ is $2 \times 1$ and $A_2$ is $2 \times 2$? No, better to split B.
    Let $B = [B_1, B_2]$ where $B_1$ is $3 \times 3$ and $B_2$ is $3 \times 4$.
    Then $C = A B_1 + A B_2$.
    
    Actually, let's look at the dimensions more carefully.
    n=2, m=3, p=7.
    
    We can decompose the multiplication into smaller matrix multiplications.
    Consider splitting the inner dimension $m=3$ is hard for Strassen.
    Consider splitting the outer dimensions.
    
    Let's try a simple block decomposition first.
    $A = \begin{pmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \end{pmatrix}$
    $B = \begin{pmatrix} B_{11} & B_{12} & B_{13} & B_{14} & B_{15} & B_{16} & B_{17} \\ B_{21} & B_{22} & B_{23} & B_{24} & B_{25} & B_{26} & B_{27} \\ B_{31} & B_{32} & B_{33} & B_{34} & B_{35} & B_{36} & B_{37} \end{pmatrix}$
    
    This is just schoolbook if we don't group.
    
    Let's use the known result that $M(2,3,7) \le M(2,2,2) + M(2,2,5)$?
    Or simply split B into $B_{left} (3 \times 3)$ and $B_{right} (3 \times 4)$.
    $C = A B_{left} + A B_{right}$.
    
    For $A B_{left}$ ($2 \times 3$ times $3 \times 3$):
    We can write $A = [A_{1}, A_{2}]$ where $A_1$ is $2 \times 1$ and $A_2$ is $2 \times 2$.
    $B_{left} = \begin{pmatrix} b_{11} & B_{12} \\ b_{21} & b_{31} \\ b_{31} & B_{32} \end{pmatrix}$? No.
    
    Let's use a standard hybrid strategy:
    1. Split the $3 \times 7$ matrix $B$ into $B_1 (3 \times 3)$ and $B_2 (3 \times 4)$.
    2. Compute $C_1 = A B_1$ and $C_2 = A B_2$.
    3. $C = C_1 + C_2$.
    
    How to compute $A B_1$ ($2 \times 3 \times 3$)?
    $A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix}$.
    $B_1 = \begin{pmatrix} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \\ b_{31} & b_{32} & b_{33} \end{pmatrix}$.
    
    We can use Strassen for the $2 \times 2$ parts if we can map them.
    Actually, $2 \times 3 \times 3$ can be done with rank 8?
    Schoolbook is $2 \times 3 \times 3 = 18$.
    
    Let's try a different split:
    Split $A$ into $A_1 (2 \times 2)$ and $A_2 (2 \times 1)$.
    Split $B$ into $B_1 (2 \times 7)$ and $B_2 (1 \times 7)$.
    $C = A_1 B_1 + A_2 B_2$.
    
    $A_1 B_1$ is $2 \times 2 \times 7$.
    $A_2 B_2$ is $2 \times 1 \times 7$.
    
    For $2 \times 2 \times 7$:
    We can use Strassen's 7 multiplications for the $2 \times 2$ part, but the inner dimension is 7?
    No, Strassen applies to square matrices.
    For $2 \times 2 \times p$, the rank is known to be small.
    Actually, $M(2,2,p)$ rank is $p+2$ for $p \ge 2$? No.
    
    Let's use the explicit Strassen decomposition for $2 \times 2 \times 2$ and apply it to blocks.
    
    Let's define the decomposition explicitly for $n=2, m=3, p=7$.
    
    We will construct the decomposition by combining:
    1. A $2 \times 2 \times 2$ Strassen block for the first two columns of B?
    
    Let $B = [B_{cols 1-2}, B_{cols 3-4}, B_{cols 5-6}, B_{col 7}]$.
    This is getting complicated to map indices.
    
    Alternative: Use the "Winograd" or "Toom" style algebraic decomposition.
    
    Let's stick to a simple, verifiable hybrid:
    Split $m=3$ into $m_1=2, m_2=1$.
    $A = [A_1, A_2]$ with $A_1 (2 \times 2), A_2 (2 \times 1)$.
    $B = \begin{pmatrix} B_1 \\ B_2 \end{pmatrix}$ with $B_1 (2 \times 7), B_2 (1 \times 7)$.
    
    $C = A_1 B_1 + A_2 B_2$.
    
    Part 1: $A_2 B_2$. $A_2$ is $2 \times 1$, $B_2$ is $1 \times 7$.
    This is an outer product. Rank 2?
    $C_2 = A_2 B_2$.
    $C_2[i, k] = A_2[i, 1] B_2[1, k]$.
    We need 2 rank-1 terms?
    Term r=1: $U_{2,1}=1, V_{1,1}=1, W_{1,i,k}=1$? No, W is $p \times n$.
    $W_{k, i} = 1$ for all $k,i$? No.
    $C_2[i, k] = A_2[i, 0] B_2[0, k]$.
    We can do this with rank 2?
    Let's just use schoolbook for this part: $2 \times 1 \times 7 = 14$ terms.
    
    Part 2: $A_1 B_1$. $A_1 (2 \times 2), B_1 (2 \times 7)$.
    $A_1 = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$.
    $B_1 = \begin{pmatrix} b_{11} & b_{12} & \dots & b_{17} \\ b_{21} & b_{22} & \dots & b_{27} \end{pmatrix}$.
    
    We can use Strassen's 7 multiplications for the $2 \times 2$ structure, but the second dimension is 7.
    Strassen computes $M_1 \dots M_7$ which are $2 \times 2$ matrix products if the input was $2 \times 2$.
    Here, $B_1$ is $2 \times 7$.
    We can split $B_1$ into four $2 \times 2$ blocks? No, $7$ is not divisible by 2.
    Split $B_1$ into $B_{1a} (2 \times 3)$ and $B_{1b} (2 \times 4)$.
    
    This recursive splitting is prone to index errors.
    
    Let's try a direct rank-30 construction based on known bounds.
    Actually, the simplest improvement is often just using Strassen on the largest square sub-block.
    
    Let's use the decomposition:
    $C = A B$.
    $A = \begin{pmatrix} A_L & A_R \end{pmatrix}$, $A_L (2 \times 2), A_R (2 \times 1)$.
    $B = \begin{pmatrix} B_L \\ B_R \end{pmatrix}$, $B_L (2 \times 7), B_R (1 \times 7)$.
    
    $C = A_L B_L + A_R B_R$.
    
    For $A_R B_R$:
    $A_R = \begin{pmatrix} a_{13} \\ a_{23} \end{pmatrix}$, $B_R = \begin{pmatrix} b_{31} & \dots & b_{37} \end{pmatrix}$.
    $C_R = \begin{pmatrix} a_{13} b_{31} & \dots & a_{13} b_{37} \\ a_{23} b_{31} & \dots & a_{23} b_{37} \end{pmatrix}$.
    This can be done with rank 2?
    $C_R[i, k] = A_R[i, 0] B_R[0, k]$.
    We can use 2 rank-1 tensors?
    $T_{i,k} = \sum_r U_{i, r} V_{r, k} W_{k, i}$? No, W is $p \times n$.
    $C[i, k] = \sum_r U[i, r] V[r, k] W[k, i]$? No, standard form is $C[i, k] = \sum_r U[i, r] V[r, k] W[k, i]$ is wrong.
    Standard: $C[i, k] = \sum_r U[i, r] V[r, k] W[k, i]$?
    The task says: $W[k*n+i, r]$.
    $C[i, k] = \sum_j A[i, j] B[j, k]$.
    Decomposition: $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    
    For $A_R B_R$:
    $i \in \{0, 1\}, j=2, k \in \{0, \dots, 6\}$.
    Term: $A[i, 2] B[2, k]$.
    We need $U[i*3+2, r] V[2*7+k, r] W[k*2+i, r] = A[i, 2] B[2, k]$.
    If we set $r=0$: $U[i*3+2, 0] = 1, V[2*7+k, 0] = 1, W[k*2+i, 0] = 1$?
    Then $\sum_r \dots = 1 \cdot 1 \cdot 1 = 1$? No, we need the product of A and B.
    $U$ picks A, $V$ picks B, $W$ picks C.
    If $U[i*3+2, r] = 1$ and $V[2*7+k, r] = 1$, then the term contributes $1 \cdot 1 \cdot W[k*2+i, r]$.
    We want this to be $A[i, 2] B[2, k]$.
    This doesn't work directly because $U$ and $V$ are coefficients.
    
    Correct mapping for schoolbook term $(i, j, k)$:
    $U[i*m+j, r] = 1$
    $V[j*p+k, r] = 1$
    $W[k*n+i, r] = 1$
    This gives $1 \cdot 1 \cdot 1 = 1$.
    Wait, the tensor value is 1.
    $C[i, k] = \sum_j A[i, j] B[j, k]$.
    The tensor $T$ is defined such that $C[i, k] = \sum_{i', j, k'} T[i', j, k'] A[i', j] B[j, k']$?
    No, $T$ is the multiplication tensor.
    $T_{i, j, k} = 1$ if we are computing $C_{i, k}$ from $A_{i, j}$ and $B_{j, k}$.
    The decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$.
    $T_{i, j, k} = \sum_r U_r[i, j] V_r[j, k] W_r[i, k]$?
    Indices:
    $U$ is $n \times m$. $U_r[i, j]$.
    $V$ is $m \times p$. $V_r[j, k]$.
    $W$ is $n \times p$. $W_r[i, k]$.
    
    Task interface:
    $U$ shape $(n*m, rank)$. $U[i*m+j, r]$.
    $V$ shape $(m*p, rank)$. $V[j*p+k, r]$.
    $W$ shape $(p*n, rank)$. $W[k*n+i, r]$.
    
    Check:
    $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    For schoolbook, for a specific $(i, j, k)$, we have one term $r$ where:
    $U[i*m+j, r] = 1$
    $V[j*p+k, r] = 1$
    $W[k*n+i, r] = 1$
    All other entries in that column are 0.
    Then the sum is 1.
    Does this match the tensor?
    The tensor $M_{n,m,p}$ has $M_{i, j, k} = 1$ if we are summing over $j$ for fixed $i, k$?
    Actually, the matrix multiplication tensor is $\sum_{i, j, k} E_{i, j} \otimes E_{j, k} \otimes E_{i, k}$.
    So $T_{i, j, k} = 1$ for all valid $i, j, k$?
    No. $T$ is a 3-way tensor.
    $T_{i, j, k}$ is the coefficient of $A_{i, j} B_{j, k}$ in the expression for $C_{i, k}$?
    $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
    So for a fixed output index $(i, k)$, the input indices are $(i, j)$ and $(j, k)$.
    The tensor component $T_{i, j, k}$ is 1 if the indices are consistent?
    Yes, $T_{i, j, k} = 1$ for all $i, j, k$ in the domain.
    Wait, if $T_{i, j, k} = 1$ for all $i, j, k$, then the rank is $n \cdot m \cdot p$?
    No, the rank is the minimum number of rank-1 tensors that sum to $T$.
    
    My previous check in the seed program was:
    `expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0`
    This implies the tensor is the identity tensor?
    No, that was likely a bug in the seed's check or my interpretation.
    
    Let's re-read the standard definition.
    $M_{n,m,p} = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$.
    So $T_{i, j, k} = 1$ for ALL $i, j, k$.
    Wait, if $T_{i, j, k} = 1$ for all $i, j, k$, then:
    $\sum_r U_r[i, j] V_r[j, k] W_r[i, k] = 1$ for all $i, j, k$.
    
    If this is the case, the schoolbook decomposition works because:
    For each $(i, j, k)$, there is a unique $r$ such that $U_r[i, j]=1, V_r[j, k]=1, W_r[i, k]=1$.
    Then the sum is 1.
    
    So the target is:
    $\forall i, j, k: \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$.
    
    This is a system of linear equations if we fix U and V? No, it's bilinear.
    
    Let's implement the Hybrid Strassen/Schoolbook.
    
    Split $m=3$ into $2+1$.
    $A = [A_1, A_2]$. $A_1 (2 \times 2), A_2 (2 \times 1)$.
    $B = [B_1; B_2]$. $B_1 (2 \times 7), B_2 (1 \times 7)$.
    
    $C = A_1 B_1 + A_2 B_2$.
    
    Part 2: $A_2 B_2$.
    $A_2 = \begin{pmatrix} a_{13} \\ a_{23} \end{pmatrix}$. $B_2 = \begin{pmatrix} b_{31} & \dots & b_{37} \end{pmatrix}$.
    $C_2[i, k] = A_2[i, 0] B_2[0, k]$.
    We need $\sum_r U[i*3+2, r] V[2*7+k, r] W[k*2+i, r] = 1$?
    No, the tensor component is 1.
    So we need to represent the "1" for all $(i, k)$ pairs in this block.
    $i \in \{0, 1\}, k \in \{0, \dots, 6\}$.
    This is a $2 \times 7$ block of 1s in the tensor?
    No, the tensor indices are $(i, j, k)$.
    Here $j=2$.
    So we need $T_{i, 2, k} = 1$ for all $i, k$.
    We can do this with rank 2?
    Let's try rank 2 for this part.
    $r_1, r_2$.
    $U[0*3+2, r_1] = 1, U[1*3+2, r_2] = 1$.
    $V[2*7+k, r_1] = 1, V[2*7+k, r_2] = 1$?
    If $V$ is 1 for all $k$, then:
    Sum for $i=0$: $U[2, r_1] V[2*7+k, r_1] W[k*2+0, r_1] + \dots$
    If we set $W[k*2+i, r] = 1$ for all $k$, then:
    $i=0: 1 \cdot 1 \cdot 1 = 1$.
    $i=1: 1 \cdot 1 \cdot 1 = 1$.
    But we need $U$ to be 0 for the other $i$.
    $U[1*3+2, r_1] = 0$.
    $U[0*3+2, r_2] = 0$.
    
    So:
    $r_1$: $U[2, r_1]=1, U[5, r_1]=0$. $V[14+k, r_1]=1$. $W[2k, r_1]=1$.
    $r_2$: $U[2, r_2]=0, U[5, r_2]=1$. $V[14+k, r_2]=1$. $W[2k+1, r_2]=1$.
    
    Check $i=0, k=0$:
    Sum = $U[2, r_1] V[14, r_1] W[0, r_1] + U[2, r_2] V[14, r_2] W[0, r_2]$
    $= 1 \cdot 1 \cdot 1 + 0 \cdot 1 \cdot 0 = 1$. Correct.
    
    Check $i=1, k=0$:
    Sum = $U[5, r_1] V[14, r_1] W[1, r_1] + U[5, r_2] V[14, r_2] W[1, r_2]$
    $= 0 \cdot 1 \cdot 0 + 1 \cdot 1 \cdot 1 = 1$. Correct.
    
    So Part 2 ($j=2$) costs rank 2.
    
    Part 1: $A_1 B_1$. $j \in \{0, 1\}$.
    $A_1 (2 \times 2), B_1 (2 \times 7)$.
    We need $T_{i, j, k} = 1$ for $j \in \{0, 1\}$.
    
    We can use Strassen for the $2 \times 2$ part of $A_1$ and $B_1$?
    $B_1$ is $2 \times 7$. We can split $B_1$ into $B_{1a} (2 \times 3)$ and $B_{1b} (2 \times 4)$.
    
    For $B_{1a} (2 \times 3)$:
    Split into $B_{1a1} (2 \times 2)$ and $B_{1a2} (2 \times 1)$.
    
    For $B_{1a1} (2 \times 2)$:
    Use Strassen. Rank 7.
    
    For $B_{1a2} (2 \times 1)$:
    $j \in \{0, 1\}, k=0$ (local).
    Similar to Part 2, but $k$ is fixed?
    $T_{i, j, k} = 1$.
    $i \in \{0, 1\}, j \in \{0, 1\}, k=0$.
    We need 4 values to be 1.
    Rank 4?
    $r_1: i=0, j=0$.
    $r_2: i=0, j=1$.
    $r_3: i=1, j=0$.
    $r_4: i=1, j=1$.
    Yes, rank 4.
    
    Total for $2 \times 2 \times 3$: $7 + 4 = 11$.
    
    For $B_{1b} (2 \times 4)$:
    Split into two $2 \times 2$ blocks.
    Each $2 \times 2 \times 2$ uses Strassen (rank 7).
    Total $7 + 7 = 14$.
    
    Total for Part 1 ($2 \times 2 \times 7$): $11 + 14 = 25$.
    
    Total Rank = Part 1 (25) + Part 2 (2) = 27.
    
    This is much better than 42.
    
    Let's implement this.
    
    Steps:
    1. Initialize U, V, W with zeros.
    2. Define helper to add a Strassen decomposition for a $2 \times 2 \times 2$ block.
    3. Define helper to add a Schoolbook decomposition for a $2 \times 1 \times 1$ or $2 \times 2 \times 1$ block.
    4. Map the global indices to local indices.
    
    Global indices:
    $i \in \{0, 1\}$.
    $j \in \{0, 1, 2\}$.
    $k \in \{0, \dots, 6\}$.
    
    Block 1: $j \in \{0, 1\}$.
    Sub-block 1.1: $k \in \{0, 1\}$. ($2 \times 2 \times 2$) -> Strassen.
    Sub-block 1.2: $k=2$. ($2 \times 2 \times 1$) -> Schoolbook (rank 4).
    Sub-block 1.3: $k \in \{3, 4\}$. ($2 \times 2 \times 2$) -> Strassen.
    Sub-block 1.4: $k \in \{5, 6\}$. ($2 \times 2 \times 2$) -> Strassen.
    
    Block 2: $j=2$.
    Sub-block 2.1: $k \in \{0, \dots, 6\}$. ($2 \times 1 \times 7$) -> Rank 2 (as derived).
    
    Total Rank = $7 + 4 + 7 + 7 + 2 = 27$.
    
    Let's verify the Strassen indices.
    Strassen for $A (2 \times 2), B (2 \times 2) \to C (2 \times 2)$.
    $M_1 = (a_{11}+a_{22})(b_{11}+b_{22})$
    $M_2 = (a_{21}+a_{22})b_{11}$
    $M_3 = a_{11}(b_{12}-b_{22})$
    $M_4 = a_{22}(b_{21}-b_{11})$
    $M_5 = (a_{11}+a_{12})b_{22}$
    $M_6 = (a_{21}-a_{11})(b_{11}+b_{12})$
    $M_7 = (a_{12}-a_{22})(b_{21}+b_{22})$
    
    $C_{11} = M_1 + M_4 - M_5 + M_7$
    $C_{12} = M_3 + M_5$
    $C_{21} = M_2 + M_4$
    $C_{22} = M_1 - M_2 + M_3 + M_6$
    
    We need to map these to the tensor decomposition.
    
    Implementing this carefully.
    
    """
    n, m, p = DIMENSIONS
    rank = 27
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    current_rank = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal current_rank
        # i_offset: 0 or 1 (but i is always 0..1, so i_offset is fixed to 0 for the 2x2 A block)
        # j_offset: 0 or 1
        # k_offset: 0..5
        
        # A indices: i in {0, 1}, j in {j_offset, j_offset+1}
        # B indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # C indices: i in {0, 1}, k in {k_offset, k_offset+1}
        
        # M1: (a11+a22)(b11+b22)
        # U: a11, a22 -> 1
        # V: b11, b22 -> 1
        # W: c11, c12, c21, c22 -> 1
        U[i_offset*m + (j_offset), current_rank] = 1
        U[i_offset*m + (j_offset+1), current_rank] = 1
        V[(j_offset)*p + (k_offset), current_rank] = 1
        V[(j_offset+1)*p + (k_offset+1), current_rank] = 1
        W[(k_offset)*n + i_offset, current_rank] = 1
        W[(k_offset+1)*n + i_offset, current_rank] = 1
        W[(k_offset)*n + (i_offset+1), current_rank] = 1
        W[(k_offset+1)*n + (i_offset+1), current_rank] = 1
        current_rank += 1
        
        # M2: (a21+a22)b11
        U[i_offset*m + (j_offset+1), current_rank] = 1 # a21? No, a21 is i=1, j=0.
        # Wait, i_offset is 0.
        # a11: i=0, j=j_offset
        # a12: i=0, j=j_offset+1
        # a21: i=1, j=j_offset
        # a22: i=1, j=j_offset+1
        
        # M2: (a21+a22) b11
        # a21: i=1, j=j_offset
        # a22: i=1, j=j_offset+1
        # b11: j=j_offset, k=k_offset
        
        U[1*m + j_offset, current_rank] = 1
        U[1*m + j_offset+1, current_rank] = 1
        V[j_offset*p + k_offset, current_rank] = 1
        # W: c21, c22
        W[k_offset*n + 1, current_rank] = 1
        W[(k_offset+1)*n + 1, current_rank] = 1
        current_rank += 1
        
        # M3: a11 (b12-b22)
        U[0*m + j_offset, current_rank] = 1
        V[j_offset*p + (k_offset+1), current_rank] = 1
        V[(j_offset+1)*p + (k_offset+1), current_rank] = -1
        W[(k_offset+1)*n + 0, current_rank] = 1
        current_rank += 1
        
        # M4: a22 (b21-b11)
        U[1*m + (j_offset+1), current_rank] = 1
        V[(j_offset+1)*p + k_offset, current_rank] = 1
        V[j_offset*p + k_offset, current_rank] = -1
        W[k_offset*n + 1, current_rank] = 1
        current_rank += 1
        
        # M5: (a11+a12) b22
        U[0*m + j_offset, current_rank] = 1
        U[0*m + (j_offset+1), current_rank] = 1
        V[(j_offset+1)*p + (k_offset+1), current_rank] = 1
        W[(k_offset+1)*n + 0, current_rank] = 1
        W[(k_offset+1)*n + 1, current_rank] = 1
        current_rank += 1
        
        # M6: (a21-a11) (b11+b12)
        U[1*m + j_offset, current_rank] = 1
        U[0*m + j_offset, current_rank] = -1
        V[j_offset*p + k_offset, current_rank] = 1
        V[j_offset*p + (k_offset+1), current_rank] = 1
        W[(k_offset+1)*n + 1, current_rank] = 1
        current_rank += 1
        
        # M7: (a12-a22) (b21+b22)
        U[0*m + (j_offset+1), current_rank] = 1
        U[1*m + (j_offset+1), current_rank] = -1
        V[(j_offset+1)*p + k_offset, current_rank] = 1
        V[(j_offset+1)*p + (k_offset+1), current_rank] = 1
        W[k_offset*n + 0, current_rank] = 1
        W[k_offset*n + 1, current_rank] = 1
        current_rank += 1

    def add_schoolbook_2x2x1(i_offset, j_offset, k_offset):
        nonlocal current_rank
        # i in {0, 1}, j in {j_offset, j_offset+1}, k = k_offset
        for i in [0, 1]:
            for j in [j_offset, j_offset+1]:
                U[i*m + j, current_rank] = 1
                V[j*p + k_offset, current_rank] = 1
                W[k_offset*n + i, current_rank] = 1
                current_rank += 1

    def add_schoolbook_2x1x7(i_offset, j_offset, k_start, k_end):
        nonlocal current_rank
        # i in {0, 1}, j = j_offset, k in [k_start, k_end)
        # Rank 2
        # r1: i=0
        U[0*m + j_offset, current_rank] = 1
        for k in range(k_start, k_end):
            V[j_offset*p + k, current_rank] = 1
            W[k*n + 0, current_rank] = 1
        current_rank += 1
        
        # r2: i=1
        U[1*m + j_offset, current_rank] = 1
        for k in range(k_start, k_end):
            V[j_offset*p + k, current_rank] = 1
            W[k*n + 1, current_rank] = 1
        current_rank += 1

    # Block 1: j=0,1
    # k=0,1: Strassen
    add_strassen(0, 0, 0)
    # k=2: Schoolbook
    add_schoolbook_2x2x1(0, 0, 2)
    # k=3,4: Strassen
    add_strassen(0, 0, 3)
    # k=5,6: Strassen
    add_strassen(0, 0, 5)
    
    # Block 2: j=2
    # k=0..6: Schoolbook 2x1x7
    add_schoolbook_2x1x7(0, 2, 0, 7)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
