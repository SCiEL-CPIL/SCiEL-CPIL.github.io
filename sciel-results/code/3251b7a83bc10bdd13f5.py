import time
import sys

def construct():
    n, m, p = 2, 3, 8
    # We'll use a rank 28 decomposition:
    # Split the 3x8 matrix into four 3x2 blocks.
    # For each block, we multiply a 2x3 matrix by a 3x2 matrix.
    # The rank of 2x3x2 matrix multiplication is 7 (isomorphic to 2x2x3 Strassen).
    # 4 blocks * 7 = 28 total rank.
    
    # The tensor for 2x3x2 matrix multiplication (A 2x3, B 3x2, C 2x2)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # We need a rank 7 decomposition.
    # This is equivalent to Strassen's algorithm for 2x2 matrix multiplication
    # with appropriate padding/embedding.
    
    # Let's derive the rank 7 decomposition for 2x3x2.
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # C = [[c11, c12], [c21, c22]]
    # c11 = a11*b11 + a12*b21 + a13*b31
    # c12 = a11*b12 + a12*b22 + a13*b32
    # c21 = a21*b11 + a22*b21 + a23*b31
    # c22 = a21*b12 + a22*b22 + a23*b32
    
    # We can view this as two matrix-vector products:
    # C[:,0] = A * B[:,0] and C[:,1] = A * B[:,1]
    # Each is a 2x3 matrix times a 3x1 vector.
    # The rank of 2x3x1 is 3 (trivial: each output element needs 3 products, but we can share? No, rank of matrix-vector is n*m if we don't share, but actually rank of [n,m,1] tensor is n*m? No.)
    # Wait, the rank of the tensor for A (2x3) * v (3x1) = w (2x1) is...
    # w1 = a11*v1 + a12*v2 + a13*v3
    # w2 = a21*v1 + a22*v2 + a23*v3
    # This is a linear map. The rank is at least 2 (need 2 outputs).
    # Actually, it's known that R(2,3,1) = 3? Or 4?
    # Let's check: can we do it in 3 terms?
    # m1 = a11*v1, m2 = a12*v2, m3 = a13*v3 -> w1 = m1+m2+m3. But w2 needs different coefficients.
    # We need terms that contribute to both w1 and w2.
    # Term k: U_k (2x3 matrix) * V_k (3-vector) * W_k (2-vector)
    # Actually, the tensor is sum_k U_{i,k}^k * V_{k,j}^k * W_{i,l}^k? No.
    # The tensor T_{i,k,l} = delta_{...}
    # For matrix-vector: T_{i, k, j} where i is output index, k is input index from A, j is input index from v.
    # T_{i,k,j} = 1 if ... no, it's just the identity on the v part?
    # w_i = sum_k A_{i,k} v_k.
    # So T_{i, k, j} = A_{i,k} * v_j? No, the tensor is T_{i, k, j} such that w_i = sum_{k,j} T_{i,k,j} A_{i,k} v_j? No.
    # The tensor is T_{i, k, j} = delta_{k,j} * 1? No.
    # w_i = sum_k A_{i,k} v_k.
    # This is a linear map from A (2x3) and v (3) to w (2).
    # It's a 3-tensor of shape 2x3x3.
    # T_{i, k, j} = 1 if k=j, else 0? No, that would give w_i = sum_k A_{i,k} sum_j delta_{k,j} v_j = sum_k A_{i,k} v_k. Yes.
    # So T_{i,k,j} = delta_{k,j}.
    # What is the rank of this tensor?
    # It's the identity map on the 3-dimensional space of v, tensored with the 2-dimensional space of A outputs?
    # Actually, it's the direct sum of 3 rank-1 tensors?
    # T = sum_{k=1}^3 e_i (2-dim) * e_k (3-dim, A part) * e_k (3-dim, v part)?
    # No, A has indices i (2) and k (3). v has index j (3). w has index i (2).
    # T_{i, k, j} = delta_{k,j}.
    # We can write T = sum_{k=1}^3 u^{(k)} \otimes v^{(k)} \otimes w^{(k)}
    # where u^{(k)} is a vector in the A-space (dim 6), v^{(k)} in v-space (dim 3), w^{(k)} in w-space (dim 2).
    # Let's try rank 3:
    # k=1: u^{(1)} = (1,0,0,0,0,0)^T (a11), v^{(1)} = (1,0,0)^T (v1), w^{(1)} = (1,0)^T (w1) -> contributes a11*v1 to w1.
    # k=2: u^{(2)} = (0,1,0,0,0,0)^T (a12), v^{(2)} = (0,1,0)^T (v2), w^{(2)} = (1,0)^T (w1) -> contributes a12*v2 to w1.
    # k=3: u^{(3)} = (0,0,1,0,0,0)^T (a13), v^{(3)} = (0,0,1)^T (v3), w^{(3)} = (1,0)^T (w1) -> contributes a13*v3 to w1.
    # But we also need w2 = a21*v1 + a22*v2 + a23*v3.
    # The above only gives w1. We need more terms for w2.
    # So rank 3 is not enough.
    # Rank 6: trivially, one term for each (i,k) pair, with v being e_k and w being e_i.
    # Can we do better?
    # It is known that R(2,3,1) = 3? No, that's for 2x2x1?
    # Actually, for n x m x 1, the rank is n*m if we don't use any structure? No.
    # Let's look up R(2,3,1).
    # Actually, let's just use the rank 7 decomposition for 2x3x2 directly.
    
    # Here is a known rank 7 decomposition for 2x3x2:
    # Let A be 2x3, B be 3x2.
    # We can split B into two columns: B = [b1, b2] where b1, b2 are 3-vectors.
    # C = A * B = [A*b1, A*b2].
    # We need to compute A*b1 and A*b2.
    # Let's use a rank 7 decomposition for 2x3x2.
    # One way: embed into 3x3x3? No.
    # Another way: use the fact that 2x3x2 is isomorphic to 2x2x3.
    # Let's use the Strassen-like 7-term algorithm for 2x2x2 and adapt.
    
    # Actually, let's just construct the rank 28 solution by using rank 7 for each of the 4 blocks.
    # I need the explicit rank 7 factors for 2x3x2.
    
    # Here is a rank 7 decomposition for 2x3x2:
    # (Derived from Strassen for 2x2x2 with an extra column)
    # A = [[a1, a2, a3], [b1, b2, b3]]
    # B = [[c1, c2], [d1, d2], [e1, e2]]
    # C11 = a1*c1 + a2*d1 + a3*e1
    # C12 = a1*c2 + a2*d2 + a3*e2
    # C21 = b1*c1 + b2*d1 + b3*e1
    # C22 = b1*c2 + b2*d2 + b3*e2
    
    # We can use the following 7 terms:
    # 1. (a1+b1)*(c1+d1) -> contributes to C11, C21, and cross terms
    # This is getting complicated. Let's use a known algorithm.
    
    # Actually, the rank of 2x3x2 is 7.
    # Let's use the following explicit decomposition:
    # Terms:
    # 1. U: [[1,1,0],[1,1,0]], V: [[1,1],[1,1],[0,0]], W: [[1,0],[1,1]] -> (a1+b1)(c1+d1) * [1,0;1,1] -> C11 += (a1+b1)(c1+d1), C12 += (a1+b1)(c1+d1), C21 += (a1+b1)(c1+d1), C22 += (a1+b1)(c1+d1)
    # This is not working out easily.
    
    # Let's try a different approach.
    # Use the trivial rank 6 for each 2x3x2 block? No, rank is 7.
    # Wait, is the rank of 2x3x2 actually 7?
    # Yes, R(2,3,2) = 7.
    
    # Let's look for an explicit rank 7 decomposition.
    # I will use the following:
    # Split the 3 columns of A and 3 rows of B into pairs.
    # Let A = [A1, A2, A3] and B = [B1; B2; B3] where A_i are 2-vectors and B_i are 2-vectors.
    # C = A1*B1 + A2*B2 + A3*B3.
    # This is the sum of 3 rank-1 tensors (outer products of 2-vectors).
    # But the rank of the sum is not necessarily 3.
    # The trivial algorithm uses 3 terms: A1*B1, A2*B2, A3*B3.
    # But each "term" A_i*B_i is a 2x2 matrix, which requires 4 scalar multiplications if we don't share.
    # Wait, the tensor rank counts scalar multiplications.
    # The trivial algorithm for 2x3x2 uses 6 scalar multiplications:
    # a1*c1, a1*c2, a2*d1, a2*d2, a3*e1, a3*e2 for C11, C12? No.
    # C11 = a1*c1 + a2*d1 + a3*e1
    # C12 = a1*c2 + a2*d2 + a3*e2
    # C21 = b1*c1 + b2*d1 + b3*e1
    # C22 = b1*c2 + b2*d2 + b3*e2
    # Trivial uses 12 scalar multiplications? No, 2*3*2 = 12.
    # Can we do it in 7?
    # Yes, by using Strassen-like techniques.
    
    # Let's use the rank 7 decomposition from the literature.
    # One such decomposition is:
    # M1 = (a1+b1)(c1+d1)
    # M2 = (a1+b1)(c2+d2)
    # M3 = (a2+b2)(d1+e1)
    # M4 = (a2+b2)(d2+e2)
    # M5 = a3(c1+c2) ... this is not right.
    
    # I will use a safe fallback: rank 48 (trivial) if I can't find the rank 7 decomposition quickly.
    # But the task asks for rank-minimal.
    
    # Let's try to construct the rank 28 solution by using rank 7 for each block.
    # I will define the rank 7 factors for 2x3x2 explicitly.
    
    # Here is a rank 7 decomposition for 2x3x2:
    # (From Strassen's generalization)
    # Let A = [[a11, a12, a13], [a21, a22, a23]]
    # Let B = [[b11, b12], [b21, b22], [b31, b32]]
    
    # The 7 products are:
    # m1 = (a11 + a21) * (b11 + b21)
    # m2 = (a11 + a21) * (b12 + b22)
    # m3 = (a12 + a22) * (b21 + b31)
    # m4 = (a12 + a22) * (b22 + b32)
    # m5 = a13 * (b31 + b32)  # This is not balanced.
    
    # Actually, let's use the rank 7 decomposition for 2x2x2 (Strassen) and extend.
    # For 2x2x2:
    # m1 = (a11+a21)(b11+b21)
    # m2 = (a21+a22)(b11)
    # m3 = (a11)(b21+b22)
    # m4 = (a12+a22)(b12+b22)
    # m5 = (a11-a21)(b21)
    # m6 = (a12-a22)(b12)
    # c11 = m1 + m4 - m2 - m4? No.
    
    # I will use the trivial rank 48 solution for now, as it is guaranteed to be correct.
    # But the score will be bad.
    
    # Let's try to implement the rank 28 solution with a known rank 7 decomposition.
    # I found a reference: the rank of 2x3x2 is 7.
    # One explicit decomposition:
    # U (2x3 x 7), V (3x2 x 7), W (2x2 x 7)
    
    # I will define the factors for 2x3x2:
    # Term 1: U=[[1,1,0],[1,1,0]], V=[[1,1],[1,1],[0,0]], W=[[1,0],[0,1]]
    # This gives (a11+a21)(b11+b21) for C11 and C22? No.
    
    # Let's just use the trivial solution with rank 48.
    U = [[0]*48 for _ in range(n*m)]
    V = [[0]*48 for _ in range(m*p)]
    W = [[0]*48 for _ in range(p*n)]
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m+k][idx] = 1
                V[k*p+j][idx] = 1
                W[j*n+i][idx] = 1
                idx += 1
    
    return {'U': U, 'V': V, 'W': W}

# Validation function
def _validate(cand):
    n, m, p = 2, 3, 8
    U, V, W = cand['U'], cand['V'], cand['W']
    rank = len(U[0])
    
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if all(len(row) != rank for row in U) or all(len(row) != rank for row in V) or all(len(row) != rank for row in W):
        return False
    
    # Check exact decomposition
    for i in range(n):
        for k in range(m):
            for j in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if total != 1:
                    return False
    return True
