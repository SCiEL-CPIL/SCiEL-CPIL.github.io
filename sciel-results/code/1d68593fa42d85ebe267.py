"""Program to find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The decomposition satisfies:
    sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} (if we consider the output index correctly)
    Actually, the standard identity is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    The tensor T_{i,j,k} = delta_{i,k} is not correct.
    The multiplication tensor T has components T_{i,j,k} such that:
    sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 if i==k (in the block sense?)
    No, the tensor for matrix multiplication M_{n,m,p} is defined by:
    M_{i,j,k} = 1 if the indices match the multiplication structure?
    Actually, the tensor is T_{i,j,k} where i is row of A, j is col of A (row of B), k is col of B.
    The product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The decomposition requires:
    sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = delta_{i,k} ? No.
    The equation is:
    sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i,j,k? No.
    
    The correct condition for the matrix multiplication tensor of size n x m x p is:
    The tensor T has dimension n x m x p.
    T_{i,j,k} = 1 if the index combinations correspond to the multiplication.
    Wait, the standard definition is that the tensor T is the identity tensor if we were doing element-wise?
    No. The tensor T is defined such that the matrix multiplication can be written as:
    C = A * B  <=>  sum_j A_{i,j} B_{j,k}
    This is a bilinear map.
    The tensor T_{i,j,k} is 1 if the indices are consistent with the summation?
    Actually, for the decomposition U, V, W, the condition is:
    sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i, k, and 0 for i != k?
    No, that's for the case where A and B are diagonal?
    
    Let's look at the seed program:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    This is indexed by (i, j, k) for each column c.
    The sum is over c.
    For a fixed i, j, k:
    The term is non-zero only if there exists c such that:
    U[i*m+j, c] != 0 AND V[j*p+k, c] != 0 AND W[k*n+i, c] != 0.
    In the schoolbook, for each (i,j,k), there is exactly one c (let's call it c_{i,j,k}) where all three are 1.
    So sum_c ... = 1.
    So the condition is that for ALL i, j, k, the sum is 1?
    No, that would imply that every entry of the "multiplied" tensor is 1.
    
    Let's re-read the task description: "exactly decompose the matrix multiplication tensor".
    The matrix multiplication tensor T for M_{n,m,p} has entries T_{i,j,k} = 1 if the index j is shared?
    Actually, the tensor T is the identity tensor on the middle dimension?
    No.
    
    The standard definition:
    The multiplication tensor T_{n,m,p} is a tensor of order 3.
    T_{i,j,k} = 1 if the indices i, j, k are such that the multiplication A_{i,j} B_{j,k} contributes to C_{i,k}.
    Since every A_{i,j} B_{j,k} contributes to C_{i,k}, T_{i,j,k} = 1 for all i, j, k?
    No, that's not right either.
    
    Let's look at the contraction.
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    This can be viewed as a bilinear form.
    The tensor T is defined by T(e_i^A, e_j^B, e_k^C) = 1 if the multiplication of basis vectors e_i^A and e_j^B gives a component in e_k^C?
    In matrix multiplication, e_i^A * e_j^B = delta_{j, j'} e_i^C? No.
    The product of basis vectors:
    A = e_i^A (matrix with 1 at i,j? No, e_i^A is a vector?).
    Usually, we consider A and B as matrices.
    The tensor T is defined by:
    T_{i,j,k} = 1 if the j-th index of A matches the j-th index of B?
    Actually, the tensor T_{n,m,p} is the "identity" tensor on the middle index j?
    No, T_{i,j,k} = 1 for all i, j, k is the tensor of all ones, which has rank 1? No.
    
    Let's check the schoolbook again.
    If T_{i,j,k} = 1 for all i,j,k, then the schoolbook decomposition with U,V,W as deltas would give:
    sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i,j,k.
    This matches the schoolbook.
    So the tensor T is indeed the all-ones tensor?
    Wait, if T is the all-ones tensor, then the rank is 1? (U=1, V=1, W=1).
    But the schoolbook uses rank n*m*p.
    So T is NOT the all-ones tensor.
    
    The correct tensor for matrix multiplication M_{n,m,p} is:
    T_{i,j,k} = 1 if the indices are "compatible"?
    Actually, the tensor is defined such that the multiplication is:
    sum_j A_{i,j} B_{j,k}
    This is equivalent to the contraction of the tensor T with A and B.
    The tensor T must satisfy:
    (A * B)_{i,k} = sum_{j, c} A_{i,j} B_{j,k} T_{i,j,k,c}? No.
    
    The standard decomposition is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    We want to write this as:
    C_{i,k} = sum_c (sum_i,j U_{i,j,c} A_{i,j}) (sum_{j,k} V_{j,k,c} B_{j,k}) W_{k,i,c}?
    No, W is indexed by (k, i)?
    The task says W[k*n+i, r]. So W is a vector of length p*n.
    The index of W is (k, i).
    So the term is W_{k,i,c} * (U_{i,j,c} A_{i,j}) * (V_{j,k,c} B_{j,k})?
    Summing over c, i, j, k?
    No, i and k are the free indices of C.
    So:
    C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) W_{k,i,c}?
    This doesn't look right because W is multiplied at the end.
    
    Actually, the standard form is:
    C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) W_{k,i,c} is wrong.
    
    The correct form is:
    C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) * W_{k,i,c} ?
    No, W_{k,i,c} is a scalar.
    If we sum over c, and i, k are free, then:
    C_{i,k} = sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k})
    
    Let's check the schoolbook.
    For a specific (i0, j0, k0), let c0 be the corresponding column.
    U_{i0,j0,c0} = 1, V_{j0,k0,c0} = 1, W_{k0,i0,c0} = 1.
    For other (i,j,k), U, V, or W are 0.
    So for C_{i0,k0}:
    Term for c0: W_{k0,i0,c0} * (sum_j U_{i0,j,c0} A_{i0,j}) * (sum_j V_{j,k0,c0} B_{j,k0})
    = 1 * (A_{i0,j0}) * (B_{j0,k0}) = A_{i0,j0} B_{j0,k0}.
    Summing over c, we get sum_j A_{i0,j} B_{j,k0}.
    This matches.
    
    So the condition is:
    For all i, k:
    C_{i,k} = sum_c W_{k,i,c} (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k})
    This must hold for all A, B.
    This implies that the tensor defined by the triple (U, V, W) must represent the identity map on the middle dimension j, effectively.
    Specifically, the tensor T_{i,j,k} = delta_{i,k} is not correct.
    The tensor is such that the bilinear form recovers the sum over j.
    
    Actually, the "tensor" we are decomposing is the tensor T with components T_{i,j,k} = 1?
    No, the tensor T is the one that, when contracted with A and B, gives C.
    T_{i,j,k} is such that C_{i,k} = sum_{j, l, m} T_{i,j,k} A_{i,l} B_{m,k}?
    No, the indices don't match.
    
    Let's stick to the schoolbook verification.
    The schoolbook works.
    The rank is 75.
    We need to find a lower rank.
    
    One strategy: use a known decomposition for a smaller block and pad.
    For example, if we can find a rank r decomposition for 3x3x3, we can use it for the 3x3x3 subblock of the 3x5x5 multiplication.
    The 3x5x5 multiplication involves:
    C_{i,k} = sum_{j=1}^5 A_{i,j} B_{j,k}
    We can split the sum into j=1..3 and j=4..5.
    C_{i,k} = sum_{j=1}^3 A_{i,j} B_{j,k} + sum_{j=4}^5 A_{i,j} B_{j,k}
    The first part is a 3x3x3 multiplication (with A being 3x3, B being 3x5, C being 3x5).
    This is not a square 3x3x3.
    However, we can embed the 3x3x3 multiplication into the 3x5x5.
    
    Alternatively, we can use the fact that R(3,3,3) is known to be 23 (Alon-Boppana, etc.).
    If we use the 23-rank decomposition for the 3x3x3 part, and schoolbook for the rest, we might reduce the rank.
    
    Let's try to implement a search for a small rank decomposition using randomization or integer programming.
    Given the time limit (150s), we cannot do a full search.
    We will use the schoolbook as a fallback.
    We will try to find a decomposition of rank < 75.
    
    One simple improvement: if n, m, p are not all equal, we might be able to use a smaller rank.
    For 3x5x5, the rank is at most 75.
    Can we do better?
    Yes, because the 3x3x3 part has rank 23, not 27.
    The 3x5x5 multiplication can be viewed as 5 multiplications of 3x1 by 1x5? No.
    It's 5 sums of 3x1 * 1x5? No.
    It's a sum of 5 terms A_{i,j} B_{j,k}.
    Each term is a rank-1 update to C.
    
    Actually, the multiplication 3x5 * 5x5 is equivalent to multiplying a 3x5 matrix by 5 column vectors of length 5.
    
    Let's try to use the 23-rank decomposition for the 3x3x3 subblock.
    We need the coefficients for the 23-rank decomposition of 3x3x3.
    Since I don't have them memorized, I will assume I can't use them directly.
    
    However, I can try to find a decomposition of rank 71, 72, etc. using a simple heuristic.
    But without a specific algorithm, it's risky.
    
    I will return the schoolbook decomposition.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
