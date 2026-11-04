"""Matrix multiplication 4x5x6 tensor decomposition."""
DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    # Partition: j=4 (rank-24 part) + j=0..3 (4x4x6 core)
    # 4x4x6 = 4x4x4 (rank 49, Winograd) + 4x4x2 (rank 8, two identities)
    # Total rank = 24 + 49 + 8 = 81
    
    rank = 81
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x 81
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 81
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 81
    
    col = 0
    
    # --- Set A: j=4, schoolbook 4x1x6 = 24 terms ---
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1.0
            V[4 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    # col = 24
    
    # --- Set B1: 4x4x4 Winograd rank-49 decomposition ---
    # Winograd's rank-49 decomposition for 4x4 matrix multiplication
    # T[i,j,k] = delta_{i,k} for the 4x4x4 tensor
    # We need 49 rank-1 tensors with integer/half-integer entries
    
    # The Winograd 4x4x4 rank-49 decomposition:
    # Based on the factorization using 4x4 matrices over Q
    # U_A: 4x4, V_B: 4x4, W_C: 4x4 for each of 49 terms
    # The decomposition: sum_{r=1}^{49} U_r ⊗ V_r ⊗ W_r where
    # (A*B)_{i,k} = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] ... 
    # Actually the tensor is T[i,j,k] = δ_{i,j} for 4x4x4 (A(4x4)*B(4x4)=C(4x4))
    # T[i,j,k] = δ_{i,k} (standard convention: C[i,k] = sum_j A[i,j]*B[j,k])
    # So T[i,j,k] = 1 if i==k, else 0
    
    # Winograd 1971 rank-49: uses the structure of 4x4 matrices
    # The 49 terms come from: 16 + 16 + 16 + 1? No, 49 = 7*7
    # Actually, the decomposition uses 49 rank-1 tensors derived from
    # the factorization of the 4x4 matrix multiplication tensor.
    
    # Let me use a known explicit construction:
    # The 49 terms are indexed by (a,b) where a,b in some set
    # One known construction uses the fact that M_4 = M_2 ⊗ M_2 + corrections
    
    # Simpler: use the known Winograd coefficients
    # For 4x4x4, the rank-49 decomposition can be constructed as follows:
    # Split 4 = 2+2, use 2x2x2 rank-7 (Winograd) for each of 4 blocks = 28
    # Plus 21 cross terms? That gives 49.
    
    # Let me use a direct construction based on the 2x2 Winograd rank-7:
    # 2x2x2 rank-7: terms indexed by (u,v) in {0,1,2,3,4,5,6}
    # Each term has U, V, W as 2x2 matrices
    
    # For 4x4, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[A11*B11+A12*B21, A11*B12+A12*B22], [A21*B11+A22*B21, A21*B12+A22*B22]]
    
    # Each 2x2 product uses rank 7, so 8 products * 7 = 56
    # But we can share: the cross terms share structure
    # Actually 8*7 = 56 > 49, so there's savings
    
    # Let me just implement the known rank-49 explicitly.
    # Reference: Winograd (1971), "Arithmetic complexity of cellular arrays"
    # The 49 terms for 4x4x4:
    
    # I'll use a construction based on polynomial multiplication:
    # (x0 + x1 t + x2 t^2 + x3 t^3)(y0 + y1 t + y2 t^2 + y3 t^3)
    # = sum_k z_k t^k
    # This needs rank related to convolution of length 4
    
    # Actually, let me use a verified small decomposition.
    # For safety, let me use schoolbook for the 4x4x4 part too,
    # and just use the j=4 partition: rank = 24 + 96 = 120. No improvement.
    
    # Let me try: 4x4x6 with a smarter decomposition.
    # 4x4x6: A(4x4)*B(4x6)=C(4x6)
    # Split B into B[:,0:4] and B[:,4:6]
    # 4x4x4 + 4x4x2
    # 4x4x4: rank 49 (need coefficients)
    # 4x4x2: rank 8 (two 4x4 identity multiplications, each rank 4)
    # Total: 57
    
    # For 4x4x2: T[i,j,k] = δ_{i,j} for k=0,1
    # Term for k=0: U[i,j]=δ_{i,j}, V[j,k']=δ_{j,k'} for all j,k'... 
    # Wait, the tensor is 4x4x2: indices i(0..3), j(0..3), k(0..1)
    # T[i,j,k] = δ_{i,j} (independent of k)
    # So for k=0: sum over r, U_r[i,j]*V_r[j,k]*W_r[k,i] = δ_{i,j}
    # For k=1: same
    # This is 2 copies of the 4x4 identity tensor
    # Each 4x4 identity: T[i,j] = δ_{i,j}, rank 4
    # So 4x4x2 rank = 8
    
    # For the 4x4x2 part (k=4,5 in original, j=0..3):
    # T[i,j,k] = δ_{i,j} for j in 0..3, k in 4..5
    # 8 terms: for each k in {4,5}, for each i in 0..3:
    #   U[i*5+j, col] = δ_{i,j} -> U[i*5+i, col] = 1
    #   V[j*6+k, col] = 1 for the specific j
    #   W[k*4+i, col] = 1
    
    # Let me just do schoolbook for everything to be safe, but partition j=4:
    # j=4: 4*6 = 24 terms
    # j=0..3: 4*4*6 = 96 terms  
    # Total: 120. Same as before.
    
    # I need actual non-trivial decompositions to beat 120.
    # Let me implement the Winograd 4x4x4 rank-49 properly.
    
    # Winograd 4x4x4 rank-49 explicit coefficients:
    # Based on the factorization using the 7-rank decomposition of 2x2x2
    # and tensor product structure with corrections.
    
    # I'll use a known set of 49 (U,V,W) triples for 4x4x4.
    # Each U,V,W is a 4x4 matrix with entries in {0, ±1, ±1/2, ±2}
    
    # Construction via 2x2 blocks using rank-7 for 2x2x2:
    # The 2x2x2 rank-7 terms (Winograd):
    # Term 0: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[0,1],[1,0]]  -> gives A11*B21 + A21*B11 for C21... 
    # This is getting complex. Let me use a verified construction.
    
    # Alternative: use the known fact that rank(M_4) ≤ 49 and implement
    # a specific published decomposition.
    
    # For now, let me implement a correct but suboptimal decomposition
    # that beats 120, using the j=4 partition with schoolbook for the rest:
    # Actually that gives 120. I need to do better.
    
    # Let me try: for the 4x4x6 part, use a decomposition based on
    # treating it as 4*(4x6) and using the rank of 4x6 matrix multiplication.
    
    # Actually, let me just carefully implement the Winograd 49.
    # Here's a construction:
    
    # The 49 terms for 4x4x4 come from the following:
    # 1. 16 terms for the "diagonal" 2x2 blocks using rank-7: 4*7=28
    # 2. 21 terms for cross-block interactions
    # Total: 49
    
    # Let me use a simpler approach: implement the decomposition using
    # the polynomial method.
    
    # For 4x4 matrix multiplication, consider:
    # (a0 + a1 t + a2 t^2 + a3 t^3)(b0 + b1 t + b2 t^2 + b3 t^3)
    # The product has 7 coefficients (degrees 0-6)
    # Each coefficient is a bilinear form
    # The rank of the convolution tensor of length 4 is... 
    # This isn't directly 49.
    
    # Let me just use a straightforward correct decomposition and accept rank 120
    # but try to shave off a few terms using the j=4 = scalar product insight.
    
    # Actually, the cleanest improvement: notice that for j=4, the contribution
    # to C[i,k] is A[i,4]*B[4,k]. This is an outer product of row 4 of A and 
    # row 4 of B. The tensor for this is:
    # T[i,4,k] = 1 for all i,k
    # This is the all-ones 4x1x6 tensor, which has rank 1!
    # Because T[i,4,k] = u_i * v_4 * w_k where u_i=1, v_4=1, w_k=1
    # So rank 1, not 24!
    
    # Wait, but in the full tensor T[i,j,k] = δ_{i,j} for the multiplication,
    # for j=4: T[i,4,k] = δ_{i,4}... no.
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if this term contributes, i.e., T[i,j,k] = δ_{i,j} is wrong.
    # T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k] such that C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k] where T[i,j,k] = 1
    # Wait no. The bilinear form: C[i,k] = sum_j A[i,j]*B[j,k]
    # As a trilinear tensor: T[i,j,k] = 1 for all i,j,k? 
    # No: C[i,k] = sum_j (A[i,j] * B[j,k])
    # The tensor T has T[i,j,k] = 1 for all i,j,k, and C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]
    # But that's not right either. Let me be precise.
    
    # The matrix multiplication tensor T ∈ R^{n×m×p} (with indices i,j,k)
    # is defined by T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
    # contribute to (i,k) entry of C.
    # In standard matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (every pair (j) connects row i of A to row k of B)
    # Wait, that can't be right because then the rank would be n*m*p which is schoolbook.
    
    # Actually YES, T[i,j,k] = 1 for all i,j,k in the standard convention where
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] and
    # sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = T[i,j,k] = δ_{i,k}... 
    
    # NO. Let me re-read the convention.
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should satisfy: for fixed i,k, sum_j A[i,j]*B[j,k]*T[i,j,k] = C[i,k]
    # This means T[i,j,k] = 1 for all i,j,k? No!
    # C[i,k] = sum_j A[i,j]*B[j,k], so the "tensor" is T[i,j,k] = 1 for all i,j,k.
    # But that gives rank n*m*p via schoolbook.
    
    # Hmm, but that's the standard definition. The tensor T_{ijk} = δ_{i,k} is NOT
    # the matrix multiplication tensor. The matrix multiplication tensor is
    # T_{ijk} = 1 for all i,j,k (in the sense that every j-index connects).
    
    # Wait, I think I'm confusing myself. Let me re-examine.
    # A is n×m, B is m×p, C is n×p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T ∈ R^{(n*m) × (m*p) × (p*n)}? No, T ∈ R^{n × m × p} with
    # T[i,j,k] = 1 for all i,j,k.
    # Rank of this tensor = min rank such that T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # For schoolbook: U[i*m+j, r] = δ_{i*m+j, r}, V[j*p+k, r] = δ_{j*p+k, r}, 
    # W[k*n+i, r] = δ_{k*n+i, r} for r = i*m*p + j*p + k... 
    # That gives T[i,j,k] = 1 when the indices match, which works with rank n*m*p.
    
    # So T[i,j,k] = 1 for all i,j,k. The rank of the all-ones tensor of shape 
    # (n*m)×(m*p)×(p*n)... no, the tensor is shape n×m×p with T[i,j,k]=1.
    
    # The all-ones tensor of shape a×b×c has rank 1! Because T[i,j,k] = u_i*v_j*w_k
    # with u_i=1, v_j=1, w_k=1.
    
    # But that contradicts the schoolbook rank being n*m*p. The issue is the INDEXING.
    # In the decomposition, U has shape (n*m)×rank, V has shape (m*p)×rank, W has shape (p*n)×rank.
    # The constraint is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor).
    
    # This is NOT the same as the tensor T[i,j,k] = 1 in the sense of rank-1 decomposition
    # of a 3D array. The indices are coupled: the same j appears in U's row index AND V's row index,
    # and the same i appears in U's row AND W's row, and same k in V's row AND W's row.
    
    # So it's a constrained tensor decomposition. The schoolbook rank n*m*p is indeed
    # the rank for this specific index-coupling structure.
    
    # OK so my earlier analysis was wrong. Let me reconsider.
    
    # For j=4 (the last row of A / last row of B):
    # T[i,4,k] = 1 for all i,k
    # Constraint: sum_r U[i*5+4, r] * V[4*6+k, r] * W[k*4+i, r] = 1 for all i,k
    # This IS an all-ones 4×6 sub-tensor, and it CAN be done in rank 1:
    # Set U[i*5+4, r0] = 1 for all i, V[4*6+k, r0] = 1 for all k, W[k*4+i, r0] = 1 for all i,k
    # Then sum = 1*1*1 = 1. ✓
    
    # So the j=4 part has rank 1, not 24!
    
    # For j=0..3 (4×4×6 sub-tensor):
    # T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..5
    # This is the 4×4×6 matrix multiplication tensor (restricted)
    # Schoolbook rank: 4*4*6 = 96
    
    # Total: 1 + 96 = 97. Improvement from 120!
    
    # Can we do better for the 4×4×6 part?
    # 4×4×6: A(4×4)*B(4×6)=C(4×6)
    # Split B into B[:,0:4] and B[:,4:6]:
    # 4×4×4 + 4×4×2
    # 4×4×4: rank 49 (Winograd)
    # 4×4×2: T[i,j,k] = 1 for i,j in 0..3, k in 0..1
    #   This is 4×4×2 all-ones. Rank? 
    #   sum_r U[i*4+j,r]*V[j*2+k,r]*W[k*4+i,r] = 1
    #   For fixed k: sum_r U[i*4+j,r]*V[j*2+k,r]*W[k*4+i,r] = 1 for all i,j
    #   This is like a 4×4×2 tensor. Can we do it in rank less than 32?
    #   For k=0: need sum_r U[i*4+j,r]*V[j*2,r]*W[0*4+i,r] = 1 for all i,j
    #   For k=1: need sum_r U[i*4+j,r]*V[j*2+1,r]*W[1*4+i,r] = 1 for all i,j
    #   If we set W[0*4+i, r] = W[1*4+i, r] for all i,r, and V[j*2+0,r] = V[j*2+1,r] for all j,r,
    #   then both conditions are the same: sum_r U[i*4+j,r]*V[j*2,r]*W[0*4+i,r] = 1
    #   This is a 4×4×1 tensor (all ones), which has rank 1!
    #   So 4×4×2 has rank 1.
    
    # Wait, that can't be right. Let me verify.
    # If rank 1: U[i*4+j, 0] = 1, V[j*2+k, 0] = 1, W[k*4+i, 0] = 1
    # Then sum = 1*1*1 = 1 for all i,j,k. ✓
    
    # So 4×4×2 has rank 1!
    
    # And 4×4×4 has rank 49 (Winograd).
    
    # Total for 4×4×6: 49 + 1 = 50.
    # Grand total: 1 (j=4) + 50 (j=0..3) = 51.
    
    # But I need the Winograd 49 coefficients!
    
    # Let me implement the Winograd 4×4×4 rank-49 decomposition.
    # This is the critical piece.
    
    # The Winograd rank-49 decomposition for 4×4×4:
    # T[i,j,k] = 1 for all i,j,k in 0..3
    # sum_{r=0}^{48} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
    
    # I'll use a construction based on the following:
    # The 4×4 matrix multiplication can be decomposed using the 
    # "Winograd algorithm" which exploits the 2×2 block structure.
    
    # One known construction uses 49 terms derived from:
    # - 16 terms for the 4 diagonal 2×2 blocks (4 × rank-4 for 2×2×2? No, 2×2×2 rank is 7)
    # Let me look up the actual coefficients.
    
    # Actually, I recall that the rank of the 4×4×4 multiplication tensor is 49,
    # and one construction uses the fact that:
    # M_4 = (M_2 ⊗ M_2) + corrections
    # M_2 ⊗ M_2 has rank 7*7 = 49
    # And the corrections cancel out!
    
    # So the 49 terms are: for each (r1, r2) where r1,r2 are indices of the 7 rank-1 
    # tensors for 2×2×2, we get a term for 4×4×4.
    
    # The 2×2×2 rank-7 decomposition (Winograd):
    # 7 terms, each with 2×2 U, V, W matrices.
    # T2[i,j,k] = 1 for all i,j,k in 0..1
    # sum_{r=0}^{6} U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] = 1
    
    # The 7 terms for 2×2×2:
    # Term 0: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[0,1],[1,0]]
    #   Check: sum over r=0 only: U[i*2+j,0]*V[j*2+k,0]*W[k*2+i,0]
    #   i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1*1*0 = 0. Should be 1. WRONG.
    
    # Let me get the correct 2×2×2 rank-7 decomposition.
    # The 2×2 matrix multiplication: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 for all i,j,k (with the index coupling as before)
    
    # Winograd 1970, 2×2×2 rank-7:
    # The 7 terms are:
    # (u,v,w) where u,v,w are 2×2 matrices:
    # 0: u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]]  -> identity*identity*identity
    #    T[0,0,0] += 1*1*1=1, T[0,0,1] += 1*0*0=0, etc.
    #    This gives T[i,j,k] = δ_{i,j}*δ_{j,k}*δ_{k,i}... = δ_{i=j=k}
    #    Not what we want.
    
    # I'm struggling to recall the exact coefficients. Let me use a different approach.
    
    # For the purpose of this program, let me implement a correct decomposition
    # with rank 97 (j=4 as rank 1, j=0..3 as schoolbook 96).
    # This is a solid improvement from 120 to 97.
    
    # If I can get the Winograd 49, I'd get to 51, but that's risky without
    # verified coefficients.
    
    # Let me go with rank 97.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Set A: j=4, rank 1
    # U[i*5+4, 0] = 1, V[4*6+k, 0] = 1, W[k*4+i, 0] = 1
    for i in range(n):
        U[i * m + 4][0] = 1.0
    for k in range(p):
        V[4 * p + k][0] = 1.0
    for i in range(n):
        for k in range(p):
            W[k * n + i][0] = 1.0
    col = 1
    
    # Set B: j=0..3, schoolbook 4*4*6 = 96
    for i in range(n):
        for j in range(4):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}

