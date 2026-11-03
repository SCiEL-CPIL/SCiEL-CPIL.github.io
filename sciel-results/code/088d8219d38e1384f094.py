"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-27 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Treat the 2x3x9 multiplication as a 2x3 matrix multiplication over the
    9-dimensional algebra C^9 (diagonal matrices). The multiplication tensor for 2x3x9
    is the Kronecker product of the 2x3x3 multiplication tensor with the 3x3 identity
    tensor (since C^9 = (C^3)^3 and multiplication is component-wise).
    
    We use the schoolbook rank-9 decomposition for 2x3x3 and take its Kronecker product
    with the rank-3 identity, giving rank 9*3 = 27.
    
    For the 2x3x3 case (n=2, m=3, p=3):
    T[a, b, c] = 1 iff a = i*3+j, b = j*3+k, c = k*2+i for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    Schoolbook rank-9 decomposition:
    For each (i, j, k), term r = i*9 + j*3 + k:
      U[i*3+j, r] = 1
      V[j*3+k, r] = 1
      W[k*2+i, r] = 1
    
    For the 2x3x9 case, we extend by the third index l in {0,1,2} (the "diagonal algebra" index):
    a = i*9 + j*3 + l  (since m=9, a = i*m + j where j ranges over 9, but j = j3*3 + l)
    Wait, let me be more careful.
    
    In the 2x3x9 case: n=2, m=3, p=9.
    a = i*m + j = i*3 + j, where i in {0,1}, j in {0,...,8}.
    b = j*p + k = j*9 + k, where j in {0,1,2}, k in {0,...,8}.
    c = k*n + i = k*2 + i, where k in {0,...,8}, i in {0,1}.
    
    The condition is: a = i*3 + j, b = j*9 + k, c = k*2 + i.
    
    Let j = j3*3 + l where j3 in {0,1,2} and l in {0,1,2}.
    Let k = k3*3 + l' where k3 in {0,1,2} and l' in {0,1,2}.
    
    Then b = j*9 + k = (j3*3 + l)*9 + (k3*3 + l') = j3*27 + l*9 + k3*3 + l'.
    But b should be in range [0, m*p) = [0, 27). So j in {0,1,2} and k in {0,...,8}.
    
    Actually, the tensor structure is:
    T[a, b, c] = 1 iff there exist i, j, k such that:
      a = i*m + j  =>  j = a % m, i = a // m
      b = j*p + k  =>  j = b // p, k = b % p
      c = k*n + i  =>  k = c // n, i = c % n
    
    For consistency: a%m = b//p, b%p = c//n, a//m = c%n.
    
    The rank-27 decomposition: for each (j, k) in {0,1,2} x {0,...,8}, create one term.
    r = j*9 + k.
    
    U[a, r] = 1 iff a%m = j. Since a = i*3 + j_a, U[a, r] = 1 iff j_a = j, i.e., a = i*3 + j for any i.
    V[b, r] = 1 iff b = j*9 + k.
    W[c, r] = 1 iff c//n = k, i.e., c = k*2 + i for any i.
    
    As shown in the original program's analysis, this gives [a%m == b//p] * [c//n == b%p],
    which is missing the condition a//m == c%n.
    
    So the naive rank-27 doesn't work. Let me use the Kronecker product approach properly.
    
    The 2x3x9 tensor can be seen as: multiply a 2x3 matrix A by a 3x9 matrix B to get a 2x9 matrix C.
    C[i, k] = sum_{j=0}^{2} A[i, j] * B[j, k].
    
    Write k = k3*3 + l where k3 in {0,1,2}, l in {0,1,2}.
    Then C[i, k3*3+l] = sum_{j=0}^{2} A[i, j] * B[j, k3*3+l].
    
    This is equivalent to: for each l, C_l = A * B_l where C_l is 2x3 and B_l is 3x3 (the l-th "block" of B).
    
    So the 2x3x9 multiplication is equivalent to three independent 2x3x3 multiplications (one for each l),
    but they share the same A matrix.
    
    If we have a rank-r decomposition for 2x3x3, we can use it for each l, but since A is shared,
    we need to be careful.
    
    Actually, the tensor for 2x3x9 is:
    T[a, b, c] where a = i*3+j (i in {0,1}, j in {0,1,2}), b = j*9+k (j in {0,1,2}, k in {0,...,8}),
    c = k*2+i (k in {0,...,8}, i in {0,1}).
    
    Let k = k3*3 + l. Then c = (k3*3+l)*2 + i = k3*6 + 2l + i.
    And b = j*9 + k3*3 + l.
    
    T[a, b, c] = 1 iff a%m = b//p and b%p = c//n and a//m = c%n.
    a%m = j, b//p = j, b%p = k = k3*3+l, c//n = (k3*6+2l+i)//2 = k3*3 + l + i//2... 
    
    This is getting complicated. Let me just use the schoolbook rank-54 which is definitely correct,
    but that's not an improvement.
    
    Let me try a different approach. Use the fact that rank(<2,3,9>) <= rank(<2,3,3>) * 3 if we can
    decompose it as a tensor product.
    
    Actually, I recall that for matrix multiplication, if you have <n,m,p> and you want <n,m,p*q>,
    you can use the decomposition of <n,m,p> and then for each "fiber" in the q dimension, use the
    same decomposition. This gives rank(<n,m,p>) * q.
    
    So rank(<2,3,9>) <= rank(<2,3,3>) * 3.
    
    If rank(<2,3,3>) = 9 (schoolbook), then rank(<2,3,9>) <= 27.
    
    But the naive approach I tried above doesn't work because of the coupling between indices.
    
    Let me construct the rank-27 decomposition properly using the tensor product structure.
    
    For the 2x3x3 case, the schoolbook decomposition has terms (i, j, k) with:
    U[i*3+j, (i,j,k)] = 1
    V[j*3+k, (i,j,k)] = 1
    W[k*2+i, (i,j,k)] = 1
    
    For the 2x3x9 case, we extend k to have two parts: k = k3*3 + l.
    The term is indexed by (i, j, k3, l).
    
    U[i*3+j, (i,j,k3,l)] = 1
    V[j*9 + k3*3 + l, (i,j,k3,l)] = 1
    W[(k3*3+l)*2 + i, (i,j,k3,l)] = 1
    
    Let me verify: T[a, b, c] = sum_{i,j,k3,l} U[a, (i,j,k3,l)] * V[b, (i,j,k3,l)] * W[c, (i,j,k3,l)]
    
    U[a, (i,j,k3,l)] = 1 iff a = i*3 + j.
    V[b, (i,j,k3,l)] = 1 iff b = j*9 + k3*3 + l.
    W[c, (i,j,k3,l)] = 1 iff c = (k3*3+l)*2 + i.
    
    So T[a, b, c] = 1 iff there exist i, j, k3, l such that:
    a = i*3 + j
    b = j*9 + k3*3 + l
    c = (k3*3+l)*2 + i
    
    From a = i*3+j: i = a//3, j = a%3.
    From b = j*9 + k3*3 + l: with j = a%3, we need b = (a%3)*9 + k3*3 + l.
    So k3*3 + l = b - (a%3)*9. Let k = b - (a%3)*9. Then k3 = k//3, l = k%3.
    From c = (k3*3+l)*2 + i = k*2 + i. With i = a//3, we need c = k*2 + a//3.
    
    So the conditions are:
    1. j = a%3
    2. k = b - j*9, and k in [0, 9)
    3. c = k*2 + a//3
    
    This is exactly the matrix multiplication condition! So this rank-27 decomposition is correct.
    
    The number of terms is 2*3*3*3 = 54? No, i in {0,1}, j in {0,1,2}, k3 in {0,1,2}, l in {0,1,2}.
    That's 2*3*3*3 = 54 terms. That's the same as schoolbook.
    
    Wait, I'm double-counting. The term is indexed by (i, j, k3, l) but i and j are determined by a,
    and k3 and l are determined by b. So for each (a, b), there's at most one term that contributes.
    But the rank is the number of terms, which is 2*3*3*3 = 54.
    
    Hmm, that's not an improvement. Let me reconsider.
    
    Actually, I think the correct approach is:
    
    For the 2x3x3 case, the schoolbook has 2*3*3 = 18 terms.
    For the 2x3x9 case, if we treat it as 3 copies of 2x3x3 (one for each l), we get 3*18 = 54 terms.
    
    But we can do better by sharing the A matrix. The key insight is that the A matrix is the same
    for all three "blocks" in the output.
    
    Let me try a different decomposition. For each j in {0,1,2} and k in {0,...,8}, create one term.
    r = j*9 + k.
    
    U[i*3+j, r] = 1 for all i in {0,1}.
    V[j*9+k, r] = 1.
    W[k*2+i, r] = 1 for all i in {0,1}.
    
    As I showed before, this gives [a%3 == b//9] * [c//2 == b%9], which is missing the condition
    a//3 == c%2.
    
    So this doesn't work. The issue is that U and W both depend on i, but in a way that doesn't
    enforce the consistency condition.
    
    I think for this specific case, the minimum rank might actually be 54, or maybe 27 with a more
    clever construction. Let me just go with the schoolbook rank-54 which is correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
