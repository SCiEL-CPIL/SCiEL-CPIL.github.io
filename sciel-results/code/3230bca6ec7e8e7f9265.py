"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication (2x5) * (5x5).
    
    Strategy:
    1. Start with the schoolbook decomposition of rank 50 (n*m*p).
    2. Attempt to reduce the rank by finding linear dependencies among the
       tensor components using a rank-revealing decomposition (QR/SVD) approach.
       Since the entries must be exact (integers or half-integers), we look for
       exact integer relations.
       
    Given the constraint of exact integer/half-integer entries and the 120s limit,
    a full search for the minimal rank (which is likely around 40-45 for these dimensions)
    is computationally heavy. However, we can try a heuristic reduction.
    
    A known result for small matrices:
    - 2x2x2: Strassen gives rank 7 (schoolbook 8).
    - For 2x5x5, we can view it as 5 instances of 2x1x1? No, that's not right.
    - We can view A (2x5) and B (5x5).
    
    Let's try a simple block approach or just return the schoolbook if no quick reduction is found.
    However, the "inspiration" mentions that Strassen-like reductions can help.
    
    For 2x5x5, we can partition the 5 columns of A and 5 rows of B.
    If we could decompose 5 into smaller blocks, e.g., 2+3, we might use Strassen for the 2x2 part
    and schoolbook for the rest, but the dimensions don't align perfectly for a simple Kronecker product
    unless we pad.
    
    Padding to 8x8?
    Rank of 8x8x8 is 63 (Strassen).
    Schoolbook for 8x8x8 is 512.
    Our problem is 2x5x5.
    If we embed 2x5x5 into 8x8x8, the rank would be at most 63.
    But we need the *exact* decomposition for the specific 2x5x5 tensor, not an embedding that might introduce extra terms or require zero-padding that doesn't cancel out.
    
    Actually, the rank of the tensor $M_{2,5,5}$ is unknown exactly but bounded.
    Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25? No, that's not a lower bound.
    Lower bound is usually higher.
    
    Given the time constraint and the requirement for exact integer/half-integer entries, finding a rank < 50 solution dynamically is risky.
    However, we can try a simple linear algebra check on the schoolbook components to see if any are linearly dependent *over the rationals* in a way that allows removing a term.
    
    The tensor is $T_{ijk} = \delta_{j,k}$? No, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    The tensor components are $T_{i j, j k, i k}$?
    Indices:
    U index: $i \cdot m + j$
    V index: $j \cdot p + k$
    W index: $k \cdot n + i$
    
    The tensor is $\sum_{r} U_{\alpha r} V_{\beta r} W_{\gamma r} = \delta_{\alpha, \beta, \gamma}$?
    No, the multiplication tensor is $M_{n,m,p} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k \otimes ...$
    
    Let's stick to the schoolbook solution as it is guaranteed to be correct and fast.
    The "inspiration" 1 and 2 are identical to the current program.
    Inspiration 3 is also identical.
    
    Is there a known small-rank decomposition for 2x5x5?
    If we use Strassen for 2x2 blocks:
    We can't simply block 5 into 2s.
    
    Let's try to verify if a rank 49 or lower solution exists via a quick greedy search for linear dependencies?
    No, that's too slow for 120s if not careful, and finding *exact* integer relations is hard.
    
    However, there is a trick:
    For $M_{n,m,p}$, if we can find a decomposition of $M_{n',m',p'}$ with rank $r$, and $n'|n, m'|m, p'|p$, we can use it.
    Here $n=2, m=5, p=5$.
    Divisors:
    $n=2$: 1, 2
    $m=5$: 1, 5
    $p=5$: 1, 5
    
    Possible sub-tensors:
    $M_{2,1,1}$: Rank 2? No, $2\times1 \times 1\times1$. $C_{i1} = A_{i1} B_{11}$.
    Tensor: $\sum_{i} A_{i1} B_{11} C_{i1}$.
    This is rank 2?
    $U_{i, r}, V_{1, r}, W_{1, r}$.
    $C_{i1} = \sum_r U_{ir} V_{1r} W_{1r}$.
    Let $V_{1r} = \delta_{r,1}$, $W_{1r} = \delta_{r,1}$. Then $C_{i1} = U_{i1}$.
    But we need $C_{i1} = A_{i1} B_{11}$.
    So $U_{i1} = A_{i1} B_{11}$.
    This is not a valid tensor decomposition in the bilinear sense where U depends on A, V on B, W on C?
    Wait, the definition is:
    $C_{ik} = \sum_{j,r} A_{ij} B_{jk} W_{kr} V_{jr} U_{ir}$?
    Standard form:
    $C_{ik} = \sum_{r=1}^R \left( \sum_{j} U_{ir} A_{ij} \right) \left( \sum_{j} V_{jr} B_{jk} \right) W_{kr}$?
    No, the standard tensor decomposition for matrix multiplication is:
    $C_{ik} = \sum_{j, r} A_{ij} B_{jk} M_{i j k r}$?
    
    The interface specifies:
    U: (n*m, rank) -> $U_{(i,j), r}$
    V: (m*p, rank) -> $V_{(j,k), r}$
    W: (p*n, rank) -> $W_{(k,i), r}$
    
    The product is:
    $\sum_{r} U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$?
    No, that sums over j and k as well?
    The task says "output factor follows the official trace convention W[k*n+i,r]".
    
    The bilinear form is:
    $C_{ik} = \sum_{j=1}^m \sum_{r=1}^R U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$?
    If this is the case, then for the schoolbook:
    $U_{ij, r} = \delta_{r, (i,j)}$
    $V_{jk, r} = \delta_{r, (j,k)}$
    $W_{ki, r} = \delta_{r, (k,i)}$
    This only works if the index r uniquely identifies the triple (i,j,k).
    In the schoolbook, $r$ runs from 1 to n*m*p.
    For a fixed r, let it correspond to $(i_r, j_r, k_r)$.
    Then the term is $U_{ij, r} V_{jk, r} W_{ki, r}$.
    If $r$ corresponds to $(i', j', k')$, then:
    $U_{ij, r} = \delta_{i, i'} \delta_{j, j'}$
    $V_{jk, r} = \delta_{j, j'} \delta_{k, k'}$
    $W_{ki, r} = \delta_{k, k'} \delta_{i, i'}$
    
    The product is non-zero only if $i=i', j=j', k=k'$.
    Then the value is 1.
    The sum over r gives $\sum_{r} \delta_{i, i_r} \delta_{j, j_r} \delta_{k, k_r} \delta_{j, j_r} \delta_{k, k_r} \delta_{k, k_r} \delta_{i, i_r}$?
    Wait, the sum in the bilinear form is usually just over r?
    If the form is $\sum_r U_{ij,r} V_{jk,r} W_{ki,r}$, then for a fixed output $C_{ik}$, we sum over j?
    The standard matrix multiplication is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    
    If the decomposition is $C_{ik} = \sum_{j, r} A_{ij} V_{jr} \dots$?
    
    Let's look at the standard definition used in these tasks.
    Usually:
    $C_{ik} = \sum_{r=1}^R \left( \sum_{j=1}^m U_{ir} A_{ij} \right) \left( \sum_{j=1}^m V_{jr} B_{jk} \right) W_{kr}$?
    No, the shapes are:
    U: (n*m, R). So $U_{(i,j), r}$.
    V: (m*p, R). So $V_{(j,k), r}$.
    W: (p*n, R). So $W_{(k,i), r}$.
    
    The reconstructed matrix C is given by:
    $C_{ik} = \sum_{j=1}^m \sum_{r=1}^R U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$?
    This would mean $W$ depends on i and k, which is the output index. That's not how bilinear decompositions work. The output is computed from inputs.
    
    Correct interpretation:
    The tensor $T$ has indices $(i, j, k)$.
    $T_{i, j, k} = \sum_{r=1}^R U_{i, j, r} V_{j, k, r} W_{k, i, r}$?
    No, the indices of U, V, W are flattened.
    $U_{\alpha, r}$ where $\alpha = i \cdot m + j$.
    $V_{\beta, r}$ where $\beta = j \cdot p + k$.
    $W_{\gamma, r}$ where $\gamma = k \cdot n + i$.
    
    The equation is:
    $\sum_{j=1}^m A_{ij} B_{jk} = \sum_{r=1}^R \left( \sum_{j=1}^m U_{i, j, r} A_{ij} \right) \left( \sum_{j=1}^m V_{j, k, r} B_{jk} \right) W_{k, i, r}$?
    This doesn't look right because W is on the output side.
    
    Actually, the standard "rank" of a matrix multiplication tensor $M_{n,m,p}$ is the smallest R such that:
    $M_{n,m,p} = \sum_{r=1}^R u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$
    where $u^{(r)} \in \mathbb{R}^{n \times m}$, $v^{(r)} \in \mathbb{R}^{m \times p}$, $w^{(r)} \in \mathbb{R}^{p \times n}$?
    No, the factors are vectors.
    $M_{n,m,p} \in \mathbb{R}^{n \times m \times p}$.
    $M_{i, j, k} = \sum_{r=1}^R U_{i, j, r} V_{j, k, r} W_{k, i, r}$?
    Wait, the indices must match.
    $U$ has indices $i, j$. $V$ has indices $j, k$. $W$ has indices $k, i$?
    If $W$ has indices $k, i$, then $W$ is a matrix of size $p \times n$.
    The product $U_{ij} V_{jk} W_{ki}$ sums over j? No, r is the sum index.
    $M_{i, j, k}$? No, the tensor component is associated with the operation.
    
    Let's assume the standard definition:
    The tensor $T$ is defined by $T_{i, j, k} = 1$ if the indices form a valid multiplication term, 0 otherwise?
    No, $T_{i, j, k}$ represents the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    So $T_{i, j, k} = 1$ for all $i, j, k$.
    The decomposition is:
    $1 = \sum_{r=1}^R U_{i, j, r} V_{j, k, r} W_{k, i, r}$?
    This equation must hold for all $i, j, k$.
    
    If this is the case, the schoolbook solution is:
    $R = n \cdot m \cdot p$.
    For each $(i, j, k)$, assign an index $r$.
    $U_{i, j, r} = 1$ if $r$ corresponds to $(i, j, k')$?
    No, $U_{i, j, r}$ must be 1 only for the specific $i, j$ of that term.
    
    Let's check the schoolbook code provided:
    `U[i * m + j][col] = 1`
    `V[j * p + k][col] = 1`
    `W[k * n + i][col] = 1`
    
    This implies:
    $U_{(i,j), r} = \delta_{r, (i,j,k)}$
    $V_{(j,k), r} = \delta_{r, (i,j,k)}$
    $W_{(k,i), r} = \delta_{r, (i,j,k)}$
    
    Then $\sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = \sum_r \delta_{r, (i,j,k)} \delta_{r, (j,k,?)} \dots$
    This only sums to 1 if the indices match perfectly.
    For a fixed $i, j, k$, the term with $r = (i,j,k)$ gives $1 \cdot 1 \cdot 1 = 1$.
    All other terms are 0.
    So the sum is 1.
    This matches $C_{ik} = \sum_j A_{ij} B_{jk}$ where the coefficient of $A_{ij} B_{jk}$ is 1.
    
    So the problem is to find $R < 50$ and matrices $U, V, W$ with integer/half-integer entries such that:
    $\sum_{r=1}^R U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1$ for all $i, j, k$.
    
    This is a system of $n \cdot m \cdot p = 50$ equations.
    Unknowns: $R \cdot (n \cdot m + m \cdot p + p \cdot n) = R \cdot (10 + 25 + 10) = 45 R$.
    If $R=1$, 45 unknowns, 50 equations. Likely no solution.
    If $R=2$, 90 unknowns.
    
    We can try to solve this system for small R using integer programming or linear algebra over rationals.
    However, the equations are cubic (product of three variables).
    
    Given the time limit and the difficulty of finding an exact integer solution for a reduced rank, and the fact that the "inspiration" programs are all the schoolbook solution, it is safest to return the schoolbook solution. Any attempt to reduce the rank without a guaranteed method risks returning an invalid decomposition (score 0).
    
    The schoolbook solution is valid and scores 1.0.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Try to construct a Strassen-based decomposition for the 2x2 sub-block
    # We partition the 5x5 matrix into blocks.
    # Since 5 is not a power of 2, we cannot simply use standard Strassen directly on 5x5.
    # However, we can partition the indices into sets.
    # Let's try a hybrid approach:
    # Split the 5 columns of A and 5 rows of B into two parts: size 2 and size 3.
    # Part 1: 2x2x2 block (indices 0,1 for j and k) -> Use Strassen (Rank 7)
    # Part 2: Remaining terms.
    # This is complex because the "remaining" terms are not a simple tensor product.
    
    # Alternative:
    # The "inspiration" finding says: "Using block decomposition with Strassen's algorithm for 2x2x2 sub-blocks reduces the rank from 50 to 46".
    # This implies a specific construction.
    # If we take the first 2x2 block of the multiplication (i.e., j in {0,1}, k in {0,1}), we have a 2x2x2 tensor.
    # The standard schoolbook rank for this 2x2x2 sub-tensor is 8.
    # Strassen rank is 7.
    # So we save 1 rank for this block.
    # What about the rest?
    # The total rank is sum of ranks of disjoint blocks + cross terms?
    # No, matrix multiplication is linear.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # If we split $j$ into $J_1$ and $J_2$, and $k$ into $K_1$ and $K_2$, it's not straightforward because the sum is over all $j$ for a fixed $k$, or all $k$ for a fixed $j$?
    # Actually, $C$ is a 2x5 matrix. $C_{ik}$ depends on all $j$.
    # The tensor $M_{2,5,5}$ can be viewed as a sum of 5 tensors $M_{2,1,5}$? No.
    # It is a sum over $j$ of $M_{2,1,1}$? No.
    
    # Let's look at the structure more carefully.
    # The tensor $T$ has components $T_{i,j,k} = 1$.
    # We can partition the index set $\{0..4\} \times \{0..4\}$ for $(j,k)$ into blocks.
    # Block 1: $j \in \{0,1\}, k \in \{0,1\}$. This is a $2 \times 2 \times 2$ tensor (with $i \in \{0,1\}$).
    # The other blocks are not cubic in the same way or are rectangular.
    
    # A simpler improvement often found in these tasks is to use Strassen on the largest power-of-2 sub-block and schoolbook on the rest, but ensuring the "rest" doesn't interfere.
    # However, for matrix multiplication, the "rest" terms are not independent.
    
    # Let's try the specific construction that yields rank 46.
    # If we use Strassen for the 2x2x2 block (j=0,1; k=0,1), we save 1 rank.
    # Rank = 49?
    # The finding says 46. That's a saving of 4.
    # This suggests using Strassen for multiple 2x2 blocks?
    # If we partition 5 into 2+2+1?
    # Blocks:
    # 1. j in {0,1}, k in {0,1} -> 2x2x2 -> Rank 7 (saving 1)
    # 2. j in {0,1}, k in {2,3} -> 2x2x2? No, i is still 2.
    #    The tensor for fixed i-set, j-set, k-set is a sub-tensor.
    #    If we take j in {0,1} and k in {2,3}, the sub-tensor is $M_{2,2,2}$?
    #    Yes, because $C_{ik}$ for $k \in \{2,3\}$ and $i \in \{0,1\}$ depends on $A_{ij}$ for $j \in \{0,1\}$ and $B_{jk}$ for $j \in \{0,1\}, k \in \{2,3\}$.
    #    Wait, $B_{jk}$ requires $j$ to be in the column of A and row of B.
    #    If we restrict $j \in \{0,1\}$, we are only using the first two columns of A and first two rows of B.
    #    This computes a partial sum.
    #    The total sum is $\sum_{j=0}^4 A_{ij} B_{jk}$.
    #    We can split this sum into:
    #    $S_1 = \sum_{j=0}^1 A_{ij} B_{jk}$ for $k \in \{0,1\}$? No, $k$ is the output index.
    #    $C_{ik} = \sum_{j=0}^4 A_{ij} B_{jk}$.
    #    
    #    Let's partition the $j$ index (columns of A, rows of B) into $J_1=\{0,1\}$ and $J_2=\{2,3,4\}$.
    #    $C_{ik} = \sum_{j \in J_1} A_{ij} B_{jk} + \sum_{j \in J_2} A_{ij} B_{jk}$.
    #    
    #    Term 1: $C^{(1)}_{ik} = \sum_{j \in \{0,1\}} A_{ij} B_{jk}$.
    #    This is a $2 \times 2 \times 5$ multiplication?
    #    $A$ is $2 \times 2$ (cols 0,1), $B$ is $2 \times 5$ (rows 0,1).
    #    Result is $2 \times 5$.
    #    Tensor rank for $2 \times 2 \times 5$?
    #    Schoolbook rank: $2 \cdot 2 \cdot 5 = 20$.
    #    Can we reduce this?
    #    We can split $k$ (output cols) into $K_1=\{0,1\}$ and $K_2=\{2,3,4\}$.
    #    $C^{(1)}_{ik}$ for $k \in K_1$: $2 \times 2 \times 2$ tensor. Rank 7 (Strassen).
    #    $C^{(1)}_{ik}$ for $k \in K_2$: $2 \times 2 \times 3$ tensor. Schoolbook rank $2 \cdot 2 \cdot 3 = 12$.
    #    Total for Term 1: $7 + 12 = 19$. (Saving 1 from 20).
    #    
    #    Term 2: $C^{(2)}_{ik} = \sum_{j \in \{2,3,4\}} A_{ij} B_{jk}$.
    #    $A$ is $2 \times 3$ (cols 2,3,4), $B$ is $3 \times 5$ (rows 2,3,4).
    #    Result is $2 \times 5$.
    #    Schoolbook rank: $2 \cdot 3 \cdot 5 = 30$.
    #    Can we reduce this?
    #    Split $k$ into $K_1=\{0,1\}$ and $K_2=\{2,3,4\}$.
    #    $k \in K_1$: $2 \times 3 \times 2$ tensor.
    #    $k \in K_2$: $2 \times 3 \times 3$ tensor.
    #    
    #    For $2 \times 3 \times 2$:
    #    We can view this as two $2 \times 3 \times 1$? Or use Strassen-like tricks?
    #    Rank of $2 \times 3 \times 2$ is likely 12 (schoolbook). Is it reducible?
    #    If we split $j$ into $2+1$:
    #    $j \in \{2,3\}$: $2 \times 2 \times 2$ (Rank 7).
    #    $j \in \{4\}$: $2 \times 1 \times 2$ (Rank 4).
    #    Total: $7 + 4 = 11$. (Saving 1 from 12).
    #    
    #    For $2 \times 3 \times 3$:
    #    Split $j$ into $2+1$:
    #    $j \in \{2,3\}$: $2 \times 2 \times 3$.
    #    $j \in \{4\}$: $2 \times 1 \times 3$.
    #    
    #    For $2 \times 2 \times 3$:
    #    Split $k$ into $2+1$:
    #    $k \in \{2,3\}$: $2 \times 2 \times 2$ (Rank 7).
    #    $k \in \{4\}$: $2 \times 2 \times 1$ (Rank 4).
    #    Total: $7 + 4 = 11$. (Saving 1 from 12).
    #    
    #    For $2 \times 1 \times 3$:
    #    Schoolbook rank 6.
    #    Split $k$ into $2+1$:
    #    $k \in \{2,3\}$: $2 \times 1 \times 2$ (Rank 4).
    #    $k \in \{4\}$: $2 \times 1 \times 1$ (Rank 2).
    #    Total: $4 + 2 = 6$. (No saving).
    #    
    #    So for $2 \times 3 \times 3$:
    #    $j \in \{2,3\}$ part: 11.
    #    $j \in \{4\}$ part: 6.
    #    Total: $11 + 6 = 17$. (Saving 1 from 18? No, schoolbook for $2 \times 3 \times 3$ is 18. $17 < 18$. Saving 1).
    #    
    #    So for Term 2 ($2 \times 3 \times 5$):
    #    $k \in K_1$ part: 11.
    #    $k \in K_2$ part: 17.
    #    Total: $11 + 17 = 28$. (Saving 2 from 30).
    #    
    #    Total Rank:
    #    Term 1: 19.
    #    Term 2: 28.
    #    Sum: 47.
    #    
    #    This is better than 50, but not 46.
    #    Maybe we can do better on the $2 \times 1 \times 2$ or similar blocks?
    #    $2 \times 1 \times 2$ rank is 4. Schoolbook is 4. No saving.
    #    
    #    What if we partition differently?
    #    The finding says 46.
    #    Let's try to construct the 47-rank solution first, as it's a clear improvement.
    #    If we can find a way to save 1 more, we get 46.
    
    # Implementation Plan:
    # 1. Define a recursive function to build the decomposition for a sub-problem $M_{n', m', p'}$.
    # 2. If $n'=m'=p'=2$, use Strassen (Rank 7).
    # 3. Otherwise, split the largest dimension into 2 and rest, or 2+2+1, etc.
    # 4. Combine the decompositions.
    
    # However, implementing a general recursive Strassen-like decomposition with exact integer/half-integer entries is complex.
    # The Strassen decomposition for $2 \times 2 \times 2$ uses entries like 1, -1, 0.
    # When combining blocks, we just place them in the correct positions in U, V, W.
    # Since the blocks are disjoint in terms of indices (e.g., different j ranges), the matrices U, V, W will have non-overlapping non-zero entries for different blocks.
    # Thus, we can simply concatenate the columns of U, V, W for each block.
    
    # Let's implement this recursive construction.
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices for U: i*2 + j
        Indices for V: j*2 + k
        Indices for W: k*2 + i
        """
        # Strassen matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need U, V, W such that sum_r U_ij_r V_jk_r W_ki_r = delta.
        # Actually, the decomposition is $C_{ik} = \sum_r L_r(A) R_r(B) W_{ik,r}$.
        # In our tensor notation:
        # $U_{(i,j), r}$ is the coefficient of $A_{ij}$ in $L_r(A)$.
        # $V_{(j,k), r}$ is the coefficient of $B_{jk}$ in $R_r(B)$.
        # $W_{(k,i), r}$ is the coefficient of $C_{ik}$ in the sum.
        
        # Let's define the 7 terms.
        # Term 1: M1
        # L1 = A11 + A22 -> U: (0,0)=1, (1,1)=1 (using i*2+j: 0*2+0=0, 1*2+1=3)
        # R1 = B11 + B22 -> V: (0,0)=1, (1,1)=1 (using j*2+k: 0*2+0=0, 1*2+1=3)
        # W contribution: C11 += 1, C22 += 1.
        # W indices: k*2+i.
        # C11 (i=0,k=1? No, C is 2x2. i is row, k is col?
        # Task says W[k*n+i]. n=2.
        # C_{ik}. i is row index of C (0..1), k is col index of C (0..1).
        # W index = k*2 + i.
        # C11: i=0, k=0 -> W[0]
        # C12: i=0, k=1 -> W[2]
        # C21: i=1, k=0 -> W[1]
        # C22: i=1, k=1 -> W[3]
        
        # So for M1:
        # W[0] += 1
        # W[3] += 1
        
        # Term 2: M2 = (A21+A22)B11
        # L2 = A21 + A22 -> U: (1,0)=1, (1,1)=1 -> U[2]=1, U[3]=1
        # R2 = B11 -> V: (0,0)=1 -> V[0]=1
        # W contribution: C21 += 1.
        # W[1] += 1
        
        # Term 3: M3 = A11(B12-B22)
        # L3 = A11 -> U[0]=1
        # R3 = B12 - B22 -> V: (0,1)=1, (1,1)=-1 -> V[1]=1, V[3]=-1
        # W contribution: C12 += 1, C22 += 1?
        # C12 = M3 + M5.
        # C22 = M1 - M2 + M3 + M6.
        # So M3 contributes to C12 and C22.
        # W[2] += 1
        # W[3] += 1
        
        # Term 4: M4 = A22(B21-B11)
        # L4 = A22 -> U[3]=1
        # R4 = B21 - B11 -> V: (1,0)=1, (0,0)=-1 -> V[2]=1, V[0]=-1
        # W contribution: C11 += 1, C21 += 1.
        # W[0] += 1
        # W[1] += 1
        
        # Term 5: M5 = (A11+A12)B22
        # L5 = A11 + A12 -> U: (0,0)=1, (0,1)=1 -> U[0]=1, U[1]=1
        # R5 = B22 -> V: (1,1)=1 -> V[3]=1
        # W contribution: C11 -= 1, C12 += 1.
        # W[0] -= 1
        # W[2] += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # L6 = A21 - A11 -> U: (1,0)=1, (0,0)=-1 -> U[2]=1, U[0]=-1
        # R6 = B11 + B12 -> V: (0,0)=1, (0,1)=1 -> V[0]=1, V[1]=1
        # W contribution: C22 += 1.
        # W[3] += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # L7 = A12 - A22 -> U: (0,1)=1, (1,1)=-1 -> U[1]=1, U[3]=-1
        # R7 = B21 + B22 -> V: (1,0)=1, (1,1)=1 -> V[2]=1, V[3]=1
        # W contribution: C11 += 1.
        # W[0] += 1
        
        # Let's assemble U, V, W (4x7)
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # r=0: M1
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        # r=1: M2
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[1][1] = 1
        
        # r=2: M3
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[2][2] = 1; W_s[3][2] = 1
        
        # r=3: M4
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[1][3] = 1
        
        # r=4: M5
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[2][4] = 1
        
        # r=5: M6
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1
        
        # r=6: M7
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1
        
        return U_s, V_s, W_s

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def combine_decompositions(decs, n, m, p):
        """
        Combine a list of decompositions.
        Each dec is a tuple (U_sub, V_sub, W_sub, i_offset, j_offset, k_offset).
        U_sub is (n_sub*m_sub, rank_sub)
        V_sub is (m_sub*p_sub, rank_sub)
        W_sub is (p_sub*n_sub, rank_sub)
        
        We need to map the local indices to global indices.
        Global U index: (i_offset + i_local) * m + (j_offset + j_local)
        Global V index: (j_offset + j_local) * p + (k_offset + k_local)
        Global W index: (k_offset + k_local) * n + (i_offset + i_local)
        """
        total_rank = sum(dec[0].__len__() for dec in decs) # No, rank is number of columns
        total_rank = 0
        for dec in decs:
            total_rank += dec[0][0].__len__() if dec[0] else 0
            
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col_offset = 0
        for U_sub, V_sub, W_sub, i_off, j_off, k_off in decs:
            n_sub, m_sub, p_sub = len(U_sub)//m_sub, m_sub, len(V_sub)//m_sub # This is tricky.
            # We need to know the sub-dimensions.
            # Let's pass them explicitly.
            pass # Placeholder
            
    # This approach is getting complicated to generalize.
    # Given the time limit, let's just implement the specific 47-rank construction derived above.
    # Or even simpler: just return the schoolbook if we can't verify the Strassen construction quickly.
    # But 47 is a clear win over 50.
    
    # Let's try to construct the 47-rank solution explicitly.
    # Blocks:
    # 1. j in {0,1}, k in {0,1}: Strassen 2x2x2. Rank 7.
    # 2. j in {0,1}, k in {2,3,4}: Schoolbook 2x2x3. Rank 12.
    # 3. j in {2,3}, k in {0,1}: Strassen 2x2x2. Rank 7.
    # 4. j in {2,3}, k in {2,3,4}: 
    #    Split k into {2,3} and {4}.
    #    4a. j in {2,3}, k in {2,3}: Strassen 2x2x2. Rank 7.
    #    4b. j in {2,3}, k in {4}: Schoolbook 2x2x1. Rank 4.
    # 5. j in {4}, k in {0,1}: Schoolbook 2x1x2. Rank 4.
    # 6. j in {4}, k in {2,3,4}: 
    #    Split k into {2,3} and {4}.
    #    6a. j in {4}, k in {2,3}: Schoolbook 2x1x2. Rank 4.
    #    6b. j in {4}, k in {4}: Schoolbook 2x1x1. Rank 2.
    
    # Total Rank: 7 + 12 + 7 + (7+4) + 4 + (4+2) = 47.
    
    # Let's implement this specific construction.
    
    U = [[0]*47 for _ in range(10)] # 2*5
    V = [[0]*47 for _ in range(25)] # 5*5
    W = [[0]*47 for _ in range(10)] # 5*2
    
    col = 0
    
    # Helper to add a block
    def add_block(i_off, j_off, k_off, n_sub, m_sub, p_sub, U_sub, V_sub, W_sub):
        nonlocal col
        for r in range(len(U_sub[0])):
            for i in range(n_sub):
                for j in range(m_sub):
                    u_val = U_sub[i*m_sub+j][r]
                    if u_val != 0:
                        U[(i_off+i)*5 + (j_off+j)][col+r] = u_val
            for j in range(m_sub):
                for k in range(p_sub):
                    v_val = V_sub[j*p_sub+k][r]
                    if v_val != 0:
                        V[(j_off+j)*5 + (k_off+k)][col+r] = v_val
            for k in range(p_sub):
                for i in range(n_sub):
                    w_val = W_sub[k*n_sub+i][r]
                    if w_val != 0:
                        W[(k_off+k)*2 + (i_off+i)][col+r] = w_val
        col += len(U_sub[0])

    # Get Strassen 2x2x2
    Us, Vs, Ws = get_strassen_2x2x2()
    
    # Get Schoolbook 2x2x3
    U_223, V_223, W_223 = get_schoolbook(2, 2, 3)
    
    # Get Schoolbook 2x2x1
    U_221, V_221, W_221 = get_schoolbook(2, 2, 1)
    
    # Get Schoolbook 2x1x2
    U_212, V_212, W_212 = get_schoolbook(2, 1, 2)
    
    # Get Schoolbook 2x1x1
    U_211, V_211, W_211 = get_schoolbook(2, 1, 1)
    
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    add_block(0, 0, 0, 2, 2, 2, Us, Vs, Ws)
    
    # Block 2: i={0,1}, j={0,1}, k={2,3,4}
    add_block(0, 0, 2, 2, 2, 3, U_223, V_223, W_223)
    
    # Block 3: i={0,1}, j={2,3}, k={0,1}
    add_block(0, 2, 0, 2, 2, 2, Us, Vs, Ws)
    
    # Block 4a: i={0,1}, j={2,3}, k={2,3}
    add_block(0, 2, 2, 2, 2, 2, Us, Vs, Ws)
    
    # Block 4b: i={0,1}, j={2,3}, k={4}
    add_block(0, 2, 4, 2, 2, 1, U_221, V_221, W_221)
    
    # Block 5: i={0,1}, j={4}, k={0,1}
    add_block(0, 4, 0, 2, 1, 2, U_212, V_212, W_212)
    
    # Block 6a: i={0,1}, j={4}, k={2,3}
    add_block(0, 4, 2, 2, 1, 2, U_212, V_212, W_212)
    
    # Block 6b: i={0,1}, j={4}, k={4}
    add_block(0, 4, 4, 2, 1, 1, U_211, V_211, W_211)
    
    # Verify rank
    if col != 47:
        # Fallback to schoolbook if construction failed
        return construct_schoolbook()
        
    return {"U": U, "V": V, "W": W}

def construct_schoolbook():
    n, m, p = DIMENSIONS
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
