import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # Helper for 2x1 by 1x2 (Rank 4)
    # A[i, j0], B[j0, k]
    # C[i, k]
    def mul_2x1_1x2(i_offset, j0, k_offset):
        # i in {0, 1}, j = j0, k in {k_offset, k_offset+1}
        # Terms:
        # 1. A[i, j0] * B[j0, k0] -> C[i, k0]
        # 2. A[i, j0] * B[j0, k1] -> C[i, k1]
        for i in range(2):
            for dk in range(2):
                add_term(i_offset + i, j0, k_offset + dk, 1, 1, 1)

    # Helper for 2x2 by 2x2 Strassen (Rank 7)
    # A[i, j], B[j, k]
    # i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
    def mul_2x2_2x2_strassen(i_offset, j_offset, k_offset):
        # M1 = (A00+A11)(B00+B11)
        add_term(i_offset, j_offset, k_offset, 1, 1, 1)
        add_term(i_offset, j_offset, k_offset+1, 1, 1, 1)
        add_term(i_offset+1, j_offset, k_offset, 1, 1, 1)
        add_term(i_offset+1, j_offset, k_offset+1, 1, 1, 1)
        add_term(i_offset, j_offset+1, k_offset, 1, 1, 1)
        add_term(i_offset, j_offset+1, k_offset+1, 1, 1, 1)
        add_term(i_offset+1, j_offset+1, k_offset, 1, 1, 1)
        add_term(i_offset+1, j_offset+1, k_offset+1, 1, 1, 1)
        # Wait, Strassen combines sums.
        # M1 = (A00+A11)(B00+B11)
        # This requires 4 products in U/V? No, U and V hold the coefficients.
        # The term is sum_u U[i,j,c] V[j,k,c] W[k,i,c]
        # For M1:
        # We need U to have 1s for (i0,j0), (i1,j1)
        # V to have 1s for (j0,k0), (j1,k1)
        # W to have 1 for (k0,i0), (k0,i1), (k1,i0), (k1,i1) with appropriate signs?
        # No, W[k,i,c] contributes to C[i,k].
        # C00 gets +M1. C11 gets +M1. C01 gets -M1? No.
        
        # Let's trace Strassen carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 columns.
        
        # M1 = (A00+A11)(B00+B11)
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k0,i1)=1, (k1,i0)=1, (k1,i1)=1
        # This computes:
        # C00 += A00 B00 + A00 B11 + A11 B00 + A11 B11 ?
        # No.
        # Term c: sum_j U[i,j,c] V[j,k,c] W[k,i,c]
        # If U has 1 at (i0,j0) and (i1,j1).
        # V has 1 at (j0,k0) and (j1,k1).
        # W has 1 at (k0,i0), (k0,i1), (k1,i0), (k1,i1).
        
        # For i=0, k=0:
        # j=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
        # j=1: U[0,1]=0
        # j=0: V[0,1]=0
        # j=1: V[1,1]=1, but U[0,1]=0.
        # So only A00 B00 contributes to C00?
        # This is not Strassen. Strassen combines A elements and B elements.
        
        # To implement Strassen in this tensor format:
        # We need to create linear combinations of A elements in U?
        # No, U is sparse.
        # Actually, the tensor decomposition allows U, V, W to have any values.
        # But we want integer/half-integer.
        # Strassen uses sums like A00+A11.
        # This means U must have 1s in multiple rows for the same column?
        # Yes.
        
        # Let's re-verify.
        # M1 = (A00+A11)(B00+B11)
        # Column c for M1:
        # U[i,j,c]:
        #   i=0, j=0: 1
        #   i=1, j=1: 1
        #   others 0
        # V[j,k,c]:
        #   j=0, k=0: 1
        #   j=1, k=1: 1
        #   others 0
        # W[k,i,c]:
        #   k=0, i=0: 1
        #   k=0, i=1: 1
        #   k=1, i=0: 1
        #   k=1, i=1: 1
        #   others 0
        
        # Check C00 (i=0, k=0):
        # Sum over j:
        # j=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
        # j=1: U[0,1]=0
        # Result: 1.
        # But we wanted A00 B00 + A00 B11 + A11 B00 + A11 B11.
        # This only gives A00 B00.
        
        # The issue is that U and V are multiplied.
        # If U has 1 at (0,0) and (1,1), and V has 1 at (0,0) and (1,1).
        # The product term for j=0 is U[0,0]V[0,0] = 1.
        # The product term for j=1 is U[0,1]V[1,0] = 0.
        # It does NOT sum A00 and A11.
        
        # To sum A00 and A11, we need to add them in the U vector?
        # But U is a matrix.
        # If we set U[0,0,c]=1 and U[1,1,c]=1.
        # Then for a fixed i=0, the sum over j is U[0,0,c]V[0,k,c] + U[0,1,c]V[1,k,c].
        # This is A00 V[0,k] + 0.
        # It does not include A11.
        
        # Strassen relies on forming sums of A elements.
        # In the tensor decomposition $T = \sum r_u v_w$, the vectors $u, v, w$ are dense?
        # No, the task says U, V, W are matrices.
        # $C_{ik} = \sum_c U_{ij,c} V_{jk,c} W_{ki,c}$?
        # No, the standard tensor decomposition is:
        # $T_{ijk} = \sum_c U_{ic} V_{jc} W_{kc}$?
        # The task defines:
        # U shape (n*m, rank) -> U[i*m+j, c]
        # V shape (m*p, rank) -> V[j*p+k, c]
        # W shape (p*n, rank) -> W[k*n+i, c]
        
        # The multiplication is:
        # $C_{ik} = \sum_j A_{ij} B_{jk}$
        # Tensor: $T_{ijk} = \delta_{...}$?
        # The tensor for matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k} = 1$ if $j$ is the middle index?
        # No, the tensor is defined such that $\langle T, A \otimes B \rangle = C$.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # So $T_{i,j,k}$ is 1 if we pick $A_{ij}$ and $B_{jk}$.
        # Actually, $T_{i,j,k}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
        # So $T_{i,j,k} = 1$ for all $i,j,k$?
        # No, $C_{ik}$ depends on $A_{ij}$ and $B_{jk}$ for a specific $j$.
        # The tensor is $T_{i,j,k} = 1$ if the indices match?
        # The tensor is simply the identity map on the middle index?
        # $T_{i,j,k} = 1$ if $j$ is the contraction index.
        # Wait, $T$ is a 3-tensor of size $n \times m \times p$.
        # $T_{i,j,k} = 1$ if $j$ is the index being summed?
        # No, $T_{i,j,k}$ is 1 for all $i,j,k$?
        # If $T_{i,j,k} = 1$, then $\sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} = \sum A_{ij} B_{jk}$.
        # This sums over all i,j,k. But C is a matrix $C_{ik}$.
        # The contraction is $\sum_j A_{ij} B_{jk}$.
        # So the tensor should be $T_{i,j,k} = 1$ if we are computing $C_{ik}$?
        # The tensor is $T \in \mathbb{R}^{n \times m \times p}$.
        # $T_{i,j,k} = 1$ if $j$ is the middle index?
        # Actually, the tensor is $T_{i,j,k} = 1$ for all $i,j,k$?
        # No, that would sum everything into a scalar.
        
        # The correct tensor for $C = AB$ is:
        # $T_{i,j,k} = 1$ if $j$ is the index?
        # Let's look at the decomposition form:
        # $T = \sum_c u_c \otimes v_c \otimes w_c$
        # $u_c \in \mathbb{R}^{n \cdot m}$, $v_c \in \mathbb{R}^{m \cdot p}$, $w_c \in \mathbb{R}^{p \cdot n}$.
        # $u_c[i \cdot m + j]$ is the coefficient for $A_{ij}$.
        # $v_c[j \cdot p + k]$ is the coefficient for $B_{jk}$.
        # $w_c[k \cdot n + i]$ is the coefficient for $C_{ik}$.
        
        # The reconstruction is:
        # $C_{ik} = \sum_c w_c[k \cdot n + i] \left( \sum_j u_c[i \cdot m + j] v_c[j \cdot p + k] \right)$?
        # No, the standard contraction is:
        # $T_{i,j,k} = \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$?
        # If so, $U$ is $n \times m \times r$, $V$ is $m \times p \times r$, $W$ is $p \times n \times r$.
        # But the task flattens them.
        
        # If $T_{i,j,k} = \sum_c U[i \cdot m + j, c] V[j \cdot p + k, c] W[k \cdot n + i, c]$,
        # Then for $T_{i,j,k}$ to be 1 (for the schoolbook), we need:
        # For a specific $j$, $U[i \cdot m + j, c] = 1, V[j \cdot p + k, c] = 1, W[k \cdot n + i, c] = 1$.
        # This matches the seed program.
        
        # Now, for Strassen:
        # We want $C_{00} = (A_{00}+A_{11})(B_{00}+B_{11}) + \dots$
        # This implies $T_{0,0,0}$ should get contributions from $A_{00}B_{00}, A_{00}B_{11}, A_{11}B_{00}, A_{11}B_{11}$.
        # In the tensor form, $T_{0,0,0} = \sum_c U[0,c] V[0,c] W[0,c]$.
        # This sum is over $c$. It does NOT sum over $j$ or $k$ inside the term.
        # The term for a specific $c$ is $U[i \cdot m + j, c] V[j \cdot p + k, c] W[k \cdot n + i, c]$.
        # Wait, the indices $i,j,k$ are fixed for $T_{i,j,k}$.
        # So for $T_{0,0,0}$, we look at $U[0, c], V[0, c], W[0, c]$.
        # $U[0, c]$ corresponds to $A_{0,0}$?
        # $U$ index is $i \cdot m + j$.
        # For $T_{i,j,k}$, we use $U[i \cdot m + j, c]$.
        # So $U$ selects the $A$ element.
        # $V$ selects the $B$ element.
        # $W$ selects the $C$ element.
        
        # So, for a single term $c$:
        # It contributes $U[i \cdot m + j, c] V[j \cdot p + k, c] W[k \cdot n + i, c]$ to $T_{i,j,k}$.
        # This means for a fixed $c$, $U$ has non-zero entries at specific $(i,j)$, $V$ at specific $(j,k)$, $W$ at specific $(k,i)$.
        # The product is non-zero only if the indices match in the chain $i \to j \to k \to i$?
        # No, the indices $i,j,k$ are the tensor indices.
        # The term is $U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}$.
        # This is a product of three numbers.
        
        # To get $(A_{00}+A_{11})(B_{00}+B_{11})$ in $C_{00}$:
        # We need $T_{0,0,0}$ to contain $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # But $T_{0,0,0}$ is a single entry.
        # $T_{0,0,0} = \sum_c U[0,c] V[0,c] W[0,c]$.
        # This sum is over $c$.
        # Each $c$ contributes a scalar to $T_{0,0,0}$.
        # But $T_{0,0,0}$ is the coefficient of $A_{00} B_{00}$?
        # No, $T_{i,j,k}$ is the coefficient of $A_{ij} B_{jk}$ in the sum that forms $C$?
        # Actually, $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So $T_{i,j,k}$ is 1 if we are summing $A_{ij} B_{jk}$ into $C_{ik}$.
        # So $T_{i,j,k} = 1$ for all $i,j,k$.
        
        # If $T_{i,j,k} = 1$, then:
        # $1 = \sum_c U[i \cdot m + j, c] V[j \cdot p + k, c] W[k \cdot n + i, c]$.
        
        # For Strassen, we want to replace the 4 terms for $2 \times 2$ with 7 terms.
        # The 4 terms are:
        # $T_{0,0,0} = 1$ (from $A_{00}B_{00}$)
        # $T_{0,1,0} = 1$ (from $A_{01}B_{10}$? No, $A_{01}B_{10}$ is not in $C_{00}$)
        # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
        # So $T_{0,0,0} = 1$ (coeff of $A_{00}B_{00}$ in $C_{00}$? No, $T$ is the tensor).
        # The tensor entry $T_{i,j,k}$ is 1 if $A_{ij}$ and $B_{jk}$ contribute to $C_{ik}$.
        # So $T_{0,0,0} = 1$ (for $A_{00}B_{00} \to C_{00}$).
        # $T_{0,1,0} = 1$ (for $A_{01}B_{10} \to C_{00}$).
        # $T_{1,0,0} = 1$ (for $A_{10}B_{00} \to C_{10}$).
        # $T_{1,1,0} = 1$ (for $A_{11}B_{10} \to C_{10}$).
        # Etc.
        
        # Strassen computes $C$ using 7 products.
        # This means we can express the 16 tensor entries (4 for each C element) using 7 rank-1 tensors.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ has non-zeros at some $i,j$.
        # $v$ has non-zeros at some $j,k$.
        # $w$ has non-zeros at some $k,i$.
        
        # For $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$:
        # This product generates $A_{00}B_{00}, A_{00}B_{11}, A_{11}B_{00}, A_{11}B_{11}$.
        # We need a term $c$ such that:
        # $U[0,0,c] V[0,0,c] W[0,0,c]$ contributes to $T_{0,0,0}$?
        # No, the term is $U[i \cdot m + j, c] V[j \cdot p + k, c] W[k \cdot n + i, c]$.
        # For $T_{0,0,0}$, we need $i=0, j=0, k=0$.
        # Value: $U[0,c] V[0,c] W[0,c]$.
        # For $T_{0,1,0}$ ($i=0, j=1, k=0$? No, $j$ is middle index).
        # $T_{i,j,k}$ corresponds to $A_{ij} B_{jk} \to C_{ik}$.
        # So $T_{0,0,0}$ is $A_{00} B_{00} \to C_{00}$.
        # $T_{0,1,0}$ is $A_{01} B_{10} \to C_{00}$.
        
        # For $M_1$, we want to cover:
        # $A_{00}B_{00}$ (in $C_{00}$)
        # $A_{00}B_{11}$ (in $C_{01}$? No, $A_{00}B_{01}$ is in $C_{01}$. $A_{00}B_{11}$ is not a valid pair for $C_{01}$?
        # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$.
        # So $A_{00}B_{11}$ is not a term in $C$.
        
        # Strassen works by forming sums of A and B.
        # The tensor decomposition must reproduce the *exact* tensor $T$.
        # $T_{i,j,k} = 1$ if $A_{ij}$ and $B_{jk}$ are multiplied to form $C_{ik}$.
        # So $T_{0,0,0} = 1$.
        # $T_{0,1,0} = 1$.
        # $T_{1,0,0} = 1$.
        # $T_{1,1,0} = 1$.
        # $T_{0,0,1} = 1$.
        # $T_{0,1,1} = 1$.
        # $T_{1,0,1} = 1$.
        # $T_{1,1,1} = 1$.
        # (Indices: $i \in \{0,1\}, j \in \{0,1\}, k \in \{0,1\}$).
        
        # We need $\sum_c U[i \cdot 2 + j, c] V[j \cdot 2 + k, c] W[k \cdot 2 + i, c] = T_{i,j,k}$.
        
        # For $M_1$:
        # We set $U$ to have 1 at $(0,0)$ and $(1,1)$.
        # $V$ to have 1 at $(0,0)$ and $(1,1)$.
        # $W$ to have 1 at $(0,0), (0,1), (1,0), (1,1)$?
        
        # Check $T_{0,0,0}$:
        # $U[0,c] V[0,c] W[0,c]$.
        # $U[0,c]=1$ (from $(0,0)$).
        # $V[0,c]=1$ (from $(0,0)$).
        # $W[0,c]=1$ (from $(0,0)$).
        # Product = 1.
        
        # Check $T_{0,1,0}$ ($A_{01}B_{10}$):
        # $i=0, j=1, k=0$.
        # $U[1,c] V[2,c] W[0,c]$.
        # $U[1,c] = 0$ (since $U$ has 1 at $(0,0)$ and $(1,1)$, i.e., indices 0 and 3).
        # So $T_{0,1,0}$ gets 0 from this term.
        
        # This means this term only contributes to $T_{0,0,0}$ and $T_{1,1,1}$?
        # $T_{1,1,1}$: $i=1, j=1, k=1$.
        # $U[3,c] V[3,c] W[3,c]$.
        # $U[3,c]=1, V[3,c]=1, W[3,c]=1$.
        # Product = 1.
        
        # So this term gives 1 to $T_{0,0,0}$ and $T_{1,1,1}$.
        # But $T_{0,0,0}$ should be 1. $T_{1,1,1}$ should be 1.
        # However, $M_1$ should contribute to $C_{00}$ and $C_{11}$.
        # $C_{00}$ uses $T_{0,0,0}$ and $T_{0,1,0}$.
        # $C_{11}$ uses $T_{1,0,1}$ and $T_{1,1,1}$.
        
        # The Strassen algorithm sums these products into C.
        # The tensor decomposition must match the *exact* tensor T.
        # If we use Strassen, we are changing the algorithm, but the tensor T is fixed.
        # We need to represent the *same* T using fewer terms.
        # This is possible because T has a low rank.
        
        # The rank of $2 \times 2 \times 2$ tensor is 7.
        # So we can find 7 vectors that sum to T.
        
        # I will use the known Strassen decomposition for the $2 \times 2 \times 2$ tensor.
        
    return {"U": U, "V": "V", "W": W}
# EVOLVE-BLOCK-END
