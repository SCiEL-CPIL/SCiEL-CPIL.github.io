import time

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = 3, 4, 4

    # We will attempt to find a rank < 44 decomposition.
    # The current best is 44 (score 1.0909).
    # Trivial is 48 (score 1.0).
    # Lower bound is 16.
    
    # Known algorithms:
    # 2x2x2: 7 (Strassen)
    # 2x2x3: 10 (Winograd?)
    # 3x3x3: 23 (Winograd)
    
    # Let's try to construct a rank 40 or 42 solution if possible.
    # If not, return the known rank 44 solution.

    # Helper to verify a decomposition
    def verify(U, V, W):
        # U: (n*m) x rank
        # V: (m*p) x rank
        # W: (p*n) x rank
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 if i,j,k are the standard multiplication tensor indices, else 0.
        # Actually, the tensor for matrix multiplication C = AB is:
        # T_{ijk} = A_{ij} B_{jk} -> C_{ik}
        # In the trace convention W[k*n+i, r], the output index is k*n+i.
        # The input indices are:
        # U index: i*m + j (A_{ij})
        # V index: j*p + k (B_{jk})
        # W index: k*n + i (C_{ik})
        
        # The tensor is T[i,j,k] = delta_{ik}? No.
        # The multiplication tensor M_{n,p,q} has entries M_{ijk} = 1 if the indices match the multiplication pattern.
        # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
        # So the tensor is sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
        # The coefficient of A_{ij} B_{jk} C_{ik} is 1.
        # So T[i,j,k] = 1 if the indices are valid (always), but the structure is defined by which triples (i,j,k) contribute.
        # Wait, the tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # So T[i,j,k] = 1 for all i,j,k? No.
        # T is a tensor in A^* \otimes B^* \otimes C^*.
        # T = \sum_{i,j,k} A_{ij} B_{jk} C_{ik} is the scalar product.
        # The tensor itself is \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # So T[i,j,k] = 1 if the indices are in range.
        # Wait, that's not right.
        # The tensor is defined by the bilinear map.
        # The rank of the tensor T_{n,m,p} is the minimum number of rank-1 tensors that sum to it.
        # T = \sum_{r} u_r \otimes v_r \otimes w_r.
        # The entries of T are T_{ijk} = \sum_r U_{ij,r} V_{jk,r} W_{ki,r} (using trace convention W_{ki} for C_{ik}).
        # We want T_{ijk} to be the multiplication tensor.
        # The multiplication tensor has T_{ijk} = 1 if the indices form a valid multiplication step?
        # No. The multiplication tensor is the identity map on the tensor product?
        # Let's check the definition.
        # C_{ik} = \sum_j A_{ij} B_{jk}.
        # This is a linear map from A \otimes B to C.
        # The tensor is the dual map.
        # T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # So T[i,j,k] = 1 for all i,j,k?
        # If so, the rank is 1? No.
        # The indices i, j, k are not independent.
        # The term e_{ij} \otimes e_{jk} \otimes e_{ik} is non-zero only for specific triples.
        # Actually, the tensor is \sum_{i,j,k} A_{ij} B_{jk} C_{ik} = \sum_{i,j,k} (e_{ij} \otimes e_{jk} \otimes e_{ik}).
        # So the tensor has a 1 at every (i,j,k) position?
        # No, that would be the all-ones tensor.
        # The multiplication tensor is the tensor that represents the bilinear form.
        # Let's look at the 2x2x2 case.
        # T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # The non-zero entries are those where i,j,k are in 0..1.
        # There are 8 such entries.
        # The rank is 7.
        # If T were all-ones, the rank would be 1.
        # So T[i,j,k] is NOT 1 for all i,j,k.
        # T[i,j,k] = 1 if the indices correspond to a term in the sum \sum_j A_{ij} B_{jk}.
        # The sum is over j.
        # So for a fixed i, k, the term is \sum_j A_{ij} B_{jk}.
        # This involves 4 terms for 2x2.
        # The tensor is \sum_{i,k} (\sum_j e_{ij} \otimes e_{jk}) \otimes e_{ik}.
        # = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # So T[i,j,k] = 1 for all i,j,k in range?
        # If so, why is the rank 7?
        # Ah, the tensor is in the space of trilinear forms.
        # The rank is the minimum number of rank-1 forms.
        # The all-ones tensor has rank 1.
        # So T is NOT the all-ones tensor.
        # Let's re-read the definition.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix."
        # The tensor is the multiplication tensor.
        # For 2x2x2, the tensor is:
        # T = e00 \otimes e00 \otimes e00 + e00 \otimes e01 \otimes e01 + e01 \otimes e00 \otimes e10 + e01 \otimes e01 \otimes e11
        # + e10 \otimes e00 \otimes e00 + e10 \otimes e01 \otimes e01 + e11 \otimes e00 \otimes e10 + e11 \otimes e01 \otimes e11
        # So T[i,j,k] = 1 if the indices are (i,j,k) such that the term appears in the sum.
        # The sum is \sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
        # So T[i,j,k] = 1 for ALL i,j,k?
        # If so, the rank is 1.
        # This contradicts the known rank 7.
        # Where is the error?
        # The tensor is NOT \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
        # The tensor is the map (A,B) -> AB.
        # The dual map is T: C^* -> (A^* \otimes B^*).
        # T(C) = \sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
        # So T = \sum_{i,j,k} e_{ik}^* \otimes e_{ij} \otimes e_{jk}.
        # The entries of T are T_{ik, ij, jk} = 1.
        # In the flattened indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # T[ i*m+j, j*p+k, k*n+i ] = 1.
        # All other entries are 0.
        # So T is sparse.
        # For 2x2x2, the non-zero entries are:
        # i=0,j=0,k=0: U[0], V[0], W[0]
        # i=0,j=0,k=1: U[0], V[1], W[1]
        # i=0,j=1,k=0: U[1], V[2], W[4] (W index: 0*2+0=0? No. W[k*n+i]. k=0, i=0 -> 0. Wait.
        # Let's check W index.
        # W[k*n+i].
        # i=0, j=1, k=0: U[0*2+1]=1, V[1*2+0]=2, W[0*2+0]=0.
        # So T[1, 2, 0] = 1.
        # i=0, j=1, k=1: U[1], V[3], W[1*2+0]=2.
        # So T[1, 3, 2] = 1.
        # i=1, j=0, k=0: U[2], V[0], W[0*2+1]=1.
        # So T[2, 0, 1] = 1.
        # i=1, j=0, k=1: U[2], V[1], W[1*2+1]=3.
        # So T[2, 1, 3] = 1.
        # i=1, j=1, k=0: U[3], V[2], W[0*2+1]=1.
        # So T[3, 2, 1] = 1.
        # i=1, j=1, k=1: U[3], V[3], W[1*2+1]=3.
        # So T[3, 3, 3] = 1.
        
        # So the non-zero entries are:
        # (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3).
        # Note: I used 0-indexed U,V,W.
        # Let's verify with Strassen.
        # Strassen rank 7.
        # So we need to check if our decomposition matches these 8 non-zero entries.
        
        # For the given U, V, W:
        # Check T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # We want this to be 1 for the 8 indices above, and 0 otherwise.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    # The target value is 1 if (i,j,k) is a valid multiplication term, else 0.
                    # Valid terms are those where the indices correspond to A[i,j] B[j,k] -> C[i,k].
                    # So ALL (i,j,k) are valid?
                    # No, the tensor is sparse in the sense that only specific combinations of U,V,W indices are non-zero.
                    # But the value T[i,j,k] is the coefficient of A[i,j] B[j,k] C[i,k].
                    # In the sum \sum_{i,j,k} A[i,j] B[j,k] C[i,k], every term is present.
                    # So T[i,j,k] = 1 for all i,j,k?
                    # If so, the rank is 1.
                    # This is the confusion.
                    # The tensor is NOT the all-ones tensor.
                    # The tensor is the map (A,B) -> AB.
                    # The dual map T: C^* -> A^* \otimes B^*.
                    # T(C) = \sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
                    # So T = \sum_{i,j,k} e_{ik} \otimes e_{ij} \otimes e_{jk}.
                    # The entries of T are T_{ik, ij, jk} = 1.
                    # In the flattened indices:
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    # T[ i*m+j, j*p+k, k*n+i ] = 1.
                    # All other entries are 0.
                    # So for a given (i,j,k), the value is 1 if the indices match, else 0.
                    # But the indices ALWAYS match for a given (i,j,k).
                    # So T[i*m+j, j*p+k, k*n+i] = 1.
                    # And T[i*m+j', j'*p+k', k'*n+i'] = 0 if (i,j,k) != (i',j',k').
                    # So the tensor has 1s at the "diagonal" positions defined by the multiplication.
                    # And 0s elsewhere.
                    # So we need to check if our decomposition gives 1 at these positions and 0 elsewhere.
                    
                    # The position is (u_idx, v_idx, w_idx) = (i*m+j, j*p+k, k*n+i).
                    # We need T[u_idx, v_idx, w_idx] = 1.
                    # And T[u_idx, v_idx, w_idx] = 0 for all other (u_idx, v_idx, w_idx) not of this form.
                    
                    # Wait, the tensor is a 3D array.
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
                    # We want T[i*m+j, j*p+k, k*n+i] = 1.
                    # And T[a,b,c] = 0 if (a,b,c) is not of the form (i*m+j, j*p+k, k*n+i).
                    
                    # So we can verify by checking all (a,b,c).
                    
        # This verification is O(n*m*p * n*m * m*p * p*n * rank).
        # For n=3, m=4, p=4, rank=44.
        # 12*16*12 * 44 = 10368 * 44 ~ 450k operations.
        # This is fast.
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = 0
                    for r in range(len(U[0])):
                        val += U[a][r] * V[b][r] * W[c][r]
                    # Determine if (a,b,c) is a valid multiplication index.
                    # a = i*m + j -> i = a//m, j = a%m
                    # b = j*p + k -> j = b//p, k = b%p
                    # c = k*n + i -> k = c//n, i = c%n
                    # Check if i,j,k are consistent.
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        if val != 1:
                            return False
                    else:
                        if val != 0:
                            return False
        return True

    # Construct the rank 44 solution
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        c = start_col
        # m1
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m2
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] -= 1
        c += 1

        # m3
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1

        # m4
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + a_r][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        c += 1

        # m5
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        c += 1

        # m6
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1

        # m7
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] += 1
        c += 1

        return c

    def add_trivial_1x4x2(a_r, b_c, start_col):
        c = start_col
        for j in range(4):
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (b_c = 0)
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(2, 0, col)

    # Right Side (b_c = 2)
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(2, 2, col)

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1

    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Verified rank 44 solution with explicit check to ensure validity, maintaining current best score", q: 0.1, delta: [0.0, 0.0]}
