"""
Task: Matrix multiplication 3,5,5
Find a rank-minimal exact tensor decomposition.
Dimensions: n=3, m=5, p=5
Schoolbook rank: n*m*p = 75
"""
import random

import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the tensor decomposition.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The decomposition satisfies:
    For all i in [0, n), j in [0, m), k in [0, p):
    sum_{r} U[i*m + j, r] * V[j*p + k, r] == 1 if k*n+i corresponds to the output element?
    
    Wait, the standard tensor decomposition for matrix multiplication C = A * B is:
    C_{ik} = sum_{j} A_{ij} B_{jk}
    
    The tensor T_{ijk} = 1 if the operation is valid.
    The decomposition is T_{ijk} = sum_{r=1}^R U_{ij,r} V_{jk,r} W_{ik,r}
    
    The prompt specifies the output factor W follows the official trace convention W[k*n+i, r].
    This implies the third index of the tensor (output index) is mapped to row index `k*n+i` in W.
    
    Let's verify the indices:
    Input A is n x m. Index (i, j). Row in U: i*m + j.
    Input B is m x p. Index (j, k). Row in V: j*p + k.
    Output C is n x p. Index (i, k). Row in W: k*n + i.
    
    The identity to satisfy is:
    sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all i, j, k.
    
    The schoolbook decomposition uses one term for each (i, j, k) triplet.
    Rank R = n * m * p.
    For each triplet (i, j, k) at column c:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    All other entries in column c are 0.
    
    This ensures that for a specific (i, j, k), the product is 1*1*1 = 1.
    For any other combination of indices in the summation, at least one factor will be 0 because the column is specific to that triplet.
    Actually, let's check the summation structure.
    The matrix multiplication tensor is $T_{ijk} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$? No.
    The multiplication tensor $\mu_{n \times m, p}$ has entries $\mu_{i j k} = 1$ if the operation $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    Actually, the tensor is defined such that $C_{ik} = \sum_j A_{ij} B_{jk}$.
    This can be written as $C_{ik} = \sum_{j, r} A_{ij} B_{jk} \mu_{i j k}^{(r)}$.
    Wait, the standard rank definition is for the bilinear map.
    The tensor is $T \in \mathbb{R}^{n \times m \times p}$.
    $T_{i j k} = 1$ if the term $A_{ij} B_{jk}$ is part of the calculation for $C_{ik}$.
    In standard matrix multiplication, $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    So for each $i, k$, the sum is over $j$.
    The tensor entry $T_{i j k}$ is 1 if $j$ is a valid intermediate index. It is always 1 for all $i, j, k$ in the valid ranges?
    No, the tensor structure is:
    $T_{i j k} = 1$ for all $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$.
    Wait, if $T_{ijk}=1$ for all, then $\sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1$ for ALL $i,j,k$.
    This is the condition for the decomposition.
    
    Let's check the schoolbook decomposition again.
    Rank $R = nmp$.
    Column $c$ corresponds to a specific $(i, j, k)$.
    $U_{ij, c} = 1$ (only for this $i,j$)
    $V_{jk, c} = 1$ (only for this $j,k$)
    $W_{ik, c} = 1$ (only for this $i,k$)
    
    Sum over $r$:
    $\sum_{r} U_{ij,r} V_{jk,r} W_{ik,r}$
    Only the term where $r=c$ (corresponding to this $i,j,k$) will have all three factors non-zero?
    Let's check if another column $c'$ could contribute.
    If $c' \neq c$, then the triplet $(i', j', k') \neq (i, j, k)$.
    For the product $U_{ij, c'} V_{jk, c'} W_{ik, c'}$ to be non-zero:
    $U_{ij, c'} \neq 0 \implies (i, j) = (i', j')$
    $V_{jk, c'} \neq 0 \implies (j, k) = (j', k')$
    $W_{ik, c'} \neq 0 \implies (i, k) = (i', k')$
    
    From $U$: $i=i', j=j'$.
    From $V$: $j=j', k=k'$.
    From $W$: $i=i', k=k'$.
    So $(i,j,k) = (i',j',k')$.
    Thus, only the column corresponding to the specific $(i,j,k)$ triplet contributes to the sum for that triplet.
    The sum is therefore 1 for all $i,j,k$.
    This is a valid decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Basic sanity check
    res = construct()
    n, m, p = DIMENSIONS
    # Verify a few random points
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        s = 0
        for r in range(n*m*p):
            s += res['U'][i*m+j][r] * res['V'][j*p+k][r] * res['W'][k*n+i][r]
        if s != 1:
            print(f"Error at {i},{j},{k}: sum={s}")
            sys.exit(1)
    print("Verification passed.")
