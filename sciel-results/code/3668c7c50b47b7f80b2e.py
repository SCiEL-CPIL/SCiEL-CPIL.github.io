"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # 1. Add Strassen blocks for 6 disjoint 2x2x2 sub-tensors
    # Blocks at (i0, j0, k0)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for i0, j0, k0 in blocks:
        # Strassen rank 7 decomposition for 2x2x2
        # Indices: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        u = [0]*n; u[i0] = 1; u[i0+1] = 1
        v = [0]*m; v[j0] = 1; v[j0+1] = 1
        w = [0]*p; w[k0] = 1; w[k0+1] = 1
        add_term(u, v, w)
        
        # M2: A[i0+1,j0] * (B[j0,k0] + B[j0+1,k0+1])
        u = [0]*n; u[i0+1] = 1
        v = [0]*m; v[j0] = 1
        w = [0]*p; w[k0] = 1; w[k0+1] = 1
        add_term(u, v, w)
        
        # M3: A[i0,j0] * B[j0+1,k0+1]
        u = [0]*n; u[i0] = 1
        v = [0]*m; v[j0+1] = 1
        w = [0]*p; w[k0+1] = 1
        add_term(u, v, w)
        
        # M4: A[i0,j0+1] * B[j0,k0]
        u = [0]*n; u[i0] = 1
        v = [0]*m; v[j0+1] = 1
        w = [0]*p; w[k0] = 1
        add_term(u, v, w)
        
        # M5: A[i0+1,j0+1] * B[j0,k0]
        u = [0]*n; u[i0+1] = 1
        v = [0]*m; v[j0+1] = 1
        w = [0]*p; w[k0] = 1
        add_term(u, v, w)
        
        # M6: (A[i0+1,j0+1] + A[i0,j0]) * (B[j0+1,k0+1] - B[j0,k0])
        u = [0]*n; u[i0] = 1; u[i0+1] = 1
        v = [0]*m; v[j0] = 1; v[j0+1] = 1
        w = [0]*p; w[k0] = -1; w[k0+1] = 1
        add_term(u, v, w)
        
        # M7: (A[i0+1,j0+1] + A[i0+1,j0]) * (B[j0,k0+1] - B[j0+1,k0])
        u = [0]*n; u[i0+1] = 1
        v = [0]*m; v[j0] = 1; v[j0+1] = 1
        w = [0]*p; w[k0] = -1; w[k0+1] = 1
        add_term(u, v, w)
        
    # 2. Add remaining parts
    # Set 1: i=2, j in 0..3, k in 0..5
    # T[2, j, k] = A[2, j] * B[j, k]
    # This is a matrix M[j,k] = A[2,j] * B[j,k]
    # We can decompose this into sum of rank-1 terms.
    # Since A[2,j] is a scalar for each j, and B[j,k] is a row of B.
    # M = sum_j A[2,j] * e_j * B[j,:]^T ? No.
    # M[j,k] = A[2,j] B[j,k].
    # We can write M as sum of 4 terms (one for each j in 0..3)?
    # Term r corresponding to j=r:
    # u[2]=1, v[r]=1, w[k]=B[r,k]? No, w must be constant for the term.
    # Wait, the tensor term is u_i v_j w_k.
    # We want sum_r u_r(2) v_r(j) w_r(k) = A[2,j] B[j,k].
    # If we pick v_r(j) = delta_{r,j}, then u_r(2) must be A[2,r].
    # Then term r is A[2,r] * delta_{r,j} * w_r(k).
    # Sum is A[2,j] w_j(k).
    # We need w_j(k) = B[j,k].
    # But w_j(k) must be a fixed vector for the term.
    # So for each j in 0..3, we add a term:
    # u: u[2]=1, others 0.
    # v: v[j]=1, others 0.
    # w: w[k] = B[j,k] for all k? No, w is part of the decomposition.
    # The decomposition is exact for ALL inputs.
    # The coefficients in U, V, W are fixed.
    # The inputs A, B are variables.
    # The term is u_i v_j w_k.
    # The contribution to T[i,j,k] is u_i v_j w_k.
    # We need sum of terms to equal A[i,j] B[j,k].
    
    # My previous logic was mixing up the coefficients and the variables.
    # In tensor decomposition, U, V, W contain the coefficients.
    # The tensor T is fixed.
    # T[i,j,k] = 1 if the multiplication uses that index.
    # Wait, the task is to decompose the MULTIPLICATION TENSOR.
    # The multiplication tensor T has entries T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j] B[j,k] involves A[i,j] and B[j,k].
    # Actually, the standard matrix multiplication tensor T_{ijk} is 1 if the indices correspond to a valid product term.
    # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has T_{ijk} = 1 for all i,j,k.
    # No, that's not right.
    # The tensor is defined such that <T, A \otimes B> = C.
    # T_{ijk} = 1 for all i,j,k in the range.
    # We need to find rank-r decomposition T = sum_r u_r \otimes v_r \otimes w_r.
    # This means T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    
    # So for Set 1 (i=2, j in 0..3, k in 0..5):
    # We need sum_r U_{2r} V_{jr} W_{kr} = 1 for all j in 0..3, k in 0..5.
    # And 0 elsewhere?
    # No, the terms from Strassen blocks are 0 here because their support is i in 0,1.
    # So we need to represent the sub-tensor T[2, j, k] = 1 for j in 0..3, k in 0..5.
    # This is a 1 x 4 x 6 sub-tensor of ones.
    # Rank of a 1 x m x p tensor of ones is min(m, p)?
    # T_{j k} = 1.
    # We need sum_r c_r v_r(j) w_r(k) = 1.
    # We can use min(4, 6) = 4 terms?
    # Let r=0..3.
    # v_r(j) = delta_{r,j}.
    # w_r(k) = 1 for all k.
    # c_r = 1.
    # Sum_r delta_{r,j} * 1 = 1 if j in 0..3.
    # Yes.
    # So 4 terms.
    
    for j in range(4):
        u = [0]*n; u[2] = 1
        v = [0]*m; v[j] = 1
        w = [1]*p
        add_term(u, v, w)
        
    # Set 2: i in 0..1, j=4, k in 0..6
    # T[i, 4, k] = 1.
    # Sub-tensor 2 x 1 x 7 of ones.
    # Rank is min(2, 1, 7) = 1?
    # T_{i k} = 1.
    # sum_r c_r u_r(i) w_r(k) = 1.
    # Use 1 term?
    # u_0(i) = 1, w_0(k) = 1, c_0 = 1.
    # v_0(4) = 1.
    # Sum = 1 * 1 * 1 = 1.
    # Yes.
    
    u = [0]*n; u[0] = 1; u[1] = 1
    v = [0]*m; v[4] = 1
    w = [1]*p
    add_term(u, v, w)
    
    # Set 3: i in 0..1, j in 0..3, k=6
    # T[i, j, 6] = 1.
    # Sub-tensor 2 x 4 x 1 of ones.
    # Rank is min(2, 4, 1) = 1?
    # T_{i j} = 1.
    # sum_r c_r u_r(i) v_r(j) = 1.
    # Use 1 term?
    # u_0(i) = 1, v_0(j) = 1, c_0 = 1.
    # w_0(6) = 1.
    # Sum = 1.
    # Yes.
    
    u = [0]*n; u[0] = 1; u[1] = 1
    v = [0]*m; v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 1
    w = [0]*p; w[6] = 1
    add_term(u, v, w)
    
    # Missing Part 1: i=2, j=4, k in 0..6
    # T[2, 4, k] = 1.
    # Sub-tensor 1 x 1 x 7 of ones.
    # Rank 1.
    
    u = [0]*n; u[2] = 1
    v = [0]*m; v[4] = 1
    w = [1]*p
    add_term(u, v, w)
    
    # Missing Part 2: i=2, j in 0..3, k=6
    # T[2, j, 6] = 1.
    # Sub-tensor 1 x 4 x 1 of ones.
    # Rank 1.
    
    u = [0]*n; u[2] = 1
    v = [0]*m; v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 1
    w = [0]*p; w[6] = 1
    add_term(u, v, w)
    
    # Convert to required format
    # U: (n*m, rank) -> Wait, U is (n*m, rank) in the prompt?
    # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # My U is (n, rank). I need to map i to i*m + j?
    # No, U corresponds to A indices. A is n x m.
    # The prompt says U has shape (n*m, rank).
    # This implies U is indexed by (i, j).
    # So U[i*m + j, r].
    # My u vector was of length n. I need to expand it to length n*m.
    # u[i] -> u_flat[i*m + j] for all j?
    # No, the term is u_i v_j w_k.
    # The tensor is T_{ijk}.
    # U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}?
    # Prompt: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Convention W[k*n+i, r].
    
    # Let's re-read carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests W is indexed by (k, i).
    # U is indexed by (i, j).
    # V is indexed by (j, k).
    
    # So my u vector (length n) needs to be broadcast to length n*m.
    # u_flat[i*m + j] = u[i] for all j?
    # If I do that, then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = sum_r u[i] v[j] w[k]
    # This matches T_{ijk} = sum_r u[i] v[j] w[k].
    
    U_flat = []
    V_flat = []
    W_flat = []
    
    for r in range(len(U)):
        u_r = U[r]
        v_r = V[r]
        w_r = W[r]
        
        # Expand u to n*m
        u_flat = [0]*(n*m)
        for i in range(n):
            for j in range(m):
                u_flat[i*m + j] = u_r[i]
                
        # Expand v to m*p
        v_flat = [0]*(m*p)
        for j in range(m):
            for k in range(p):
                v_flat[j*p + k] = v_r[j]
                
        # Expand w to p*n (index k*n + i)
        w_flat = [0]*(p*n)
        for k in range(p):
            for i in range(n):
                w_flat[k*n + i] = w_r[k]
                
        U_flat.append(u_flat)
        V_flat.append(v_flat)
        W_flat.append(w_flat)
        
    # Transpose to (rows, rank)
    rank = len(U_flat)
    U_out = [[U_flat[r][i] for r in range(rank)] for i in range(n*m)]
    V_out = [[V_flat[r][i] for r in range(rank)] for i in range(m*p)]
    W_out = [[W_flat[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
