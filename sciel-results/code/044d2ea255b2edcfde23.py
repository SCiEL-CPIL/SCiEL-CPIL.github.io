import numpy as np

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor 
    of dimensions 3x4x6.
    
    The matrix multiplication $C = A \times B$ where $A$ is $3 \times 4$ and 
    $B$ is $4 \times 6$ results in $C$ being $3 \times 6$.
    
    The bilinear form is defined as:
    $C_{i,k} = \sum_{j=0}^{3} A_{i,j} B_{j,k}$
    
    We seek a decomposition of the form:
    $C_{i,k} = \sum_{r=0}^{R-1} W_{k \cdot 3 + i, r} \left( \sum_{j=0}^{3} U_{i \cdot 4 + j, r} A_{i,j} \right) \left( \sum_{j=0}^{3} V_{j \cdot 6 + k, r} B_{j,k} \right)$
    
    Note: The standard rank-1 term in the tensor decomposition of bilinear maps 
    $(A, B) \mapsto C$ is typically expressed as:
    $C = \sum_{r} \alpha_r(A) \beta_r(B) \gamma_r$
    where $\gamma_r$ is a matrix in the output space.
    
    In the provided interface:
    U has shape (n*m, rank) = (12, R)
    V has shape (m*p, rank) = (24, R)
    W has shape (p*n, rank) = (18, R)
    
    The condition for exact decomposition is:
    For all $i \in \{0,1,2\}, k \in \{0,..,5\}$:
    $\sum_{j=0}^{3} A_{i,j} B_{j,k} = \sum_{r=0}^{R-1} W_{k \cdot 3 + i, r} \cdot (\text{linear form on A})_r \cdot (\text{linear form on B})_r$
    
    Wait, the standard tensor decomposition of the multiplication map $M_{n,p}: \mathbb{R}^{n \times m} \times \mathbb{R}^{m \times p} \to \mathbb{R}^{n \times p}$
    is a sum of rank-1 tensors in $\mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{n \times p}$.
    
    A rank-1 tensor is $u \otimes v \otimes w$.
    The term contributes $u(A) v(B) w$ to the output? No.
    The tensor $T$ is defined by $T(A,B) = A B$.
    $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
    This implies $A B = \sum_{r} (u_r^T \text{vec}(A)) (v_r^T \text{vec}(B)) w_r$?
    No, the standard identity is:
    $AB = \sum_{i,j,k} A_{ij} B_{jk} E_{ik}$
    Here, each term is a product of three linear forms?
    $A_{ij}$ is a linear form on A.
    $B_{jk}$ is a linear form on B.
    $E_{ik}$ is a fixed matrix in the output space.
    
    So, $U$ corresponds to the coefficients of the linear forms on A.
    $V$ corresponds to the coefficients of the linear forms on B.
    $W$ corresponds to the coefficients of the output matrices $w_r$.
    
    In the schoolbook algorithm:
    For each triple $(i, j, k)$:
    $u_r$ is the vector with 1 at $(i,j)$ and 0 elsewhere.
    $v_r$ is the vector with 1 at $(j,k)$ and 0 elsewhere.
    $w_r$ is the matrix $E_{ik}$ (1 at $(i,k)$ and 0 elsewhere).
    
    The interface defines:
    U: shape $(n \cdot m, \text{rank})$. Entry $U[i \cdot m + j, r]$.
    V: shape $(m \cdot p, \text{rank})$. Entry $V[j \cdot p + k, r]$.
    W: shape $(p \cdot n, \text{rank})$. Entry $W[k \cdot n + i, r]$.
    
    The output is reconstructed as:
    $C_{i,k} = \sum_{r} W[k \cdot n + i, r] \cdot \left( \sum_{j} U[i \cdot m + j, r] A_{i,j} \right) \cdot \left( \sum_{j'} V[j' \cdot p + k, r] B_{j',k} \right)$
    
    Wait, if $U$ and $V$ are just single entries (like in schoolbook), then:
    $\sum_{j} U[i \cdot m + j, r] A_{i,j}$ would be $A_{i, j_r}$ if $U$ is a basis vector.
    But the sum is over $j$. If $U$ has 1 at $j_r$ and 0 else, the sum is $A_{i, j_r}$.
    Similarly for $V$, sum is $B_{j_r, k}$.
    Then the term is $W[k \cdot n + i, r] A_{i, j_r} B_{j_r, k}$.
    For this to match $A_{i, j} B_{j, k}$, we need $j_r = j$ and $W[k \cdot n + i, r] = 1$.
    And we sum over all $r$.
    So for each $r$, we pick a specific $j_r$.
    To cover all $j$, we need multiple $r$'s for each $(i,k)$?
    
    In schoolbook:
    We iterate $i, j, k$.
    For a specific $r$ corresponding to $(i,j,k)$:
    $U[i \cdot m + j, r] = 1$. Other entries in column $r$ of $U$ are 0.
    So $\sum_{j'} U[i \cdot m + j', r] A_{i, j'} = A_{i, j}$.
    $V[j \cdot p + k, r] = 1$. Other entries in column $r$ of $V$ are 0.
    So $\sum_{j'} V[j' \cdot p + k, r] B_{j', k} = B_{j, k}$.
    $W[k \cdot n + i, r] = 1$. Other entries in column $r$ of $W$ are 0.
    
    The term is $1 \cdot A_{i,j} \cdot B_{j,k}$.
    Summing over all $r$ (which corresponds to all triples $i,j,k$):
    $\sum_{i,j,k} A_{i,j} B_{j,k} \delta_{i, i_{target}} \delta_{k, k_{target}}$?
    
    Wait, the output $C_{i_{target}, k_{target}}$ is the sum of terms where the $W$ index matches.
    $W[k \cdot n + i, r]$ is non-zero only if the row index is $k_{target} \cdot n + i_{target}$.
    In schoolbook, for a given $r$ (triple $i,j,k$), $W$ is 1 at $k \cdot n + i$.
    So this term contributes to $C_{i,k}$.
    The contribution is $A_{i,j} B_{j,k}$.
    Summing over all $r$ with the same $i,k$ (which are $m$ terms, one for each $j$):
    $\sum_{j} A_{i,j} B_{j,k}$.
    This is exactly $(AB)_{i,k}$.
    
    So the schoolbook decomposition works with rank $n \cdot m \cdot p$.
    """
    n, m, p = 3, 4, 6
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
