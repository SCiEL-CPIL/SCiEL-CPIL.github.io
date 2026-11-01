"""Matrix multiplication 2x3 by 3x3 - rank-minimal decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = 2, 3, 3
    
    # Tensor slices: W_slice[k] is n x m matrix with W_slice[k][i][j] = 1 if (i,j,k) is a valid product term
    # Product C[i,k] = sum_j A[i,j]*B[j,k]
    # So tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Slice along k: W_slice[k] has entries [i][j] = 1 for all i,j (since every A[i,j]*B[j,k] contributes)
    # Wait, that's not right. The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Slice along dimension k: for fixed k, we get an n x m matrix with all 1s
    # That can't be right for rank analysis...
    
    # Actually the tensor T is defined by T[i,j,k] = 1 if the product term A[i,j]*B[j,k] exists
    # For standard matrix multiplication, ALL terms exist, so T[i,j,k] = 1 for all i,j,k
    # This is the all-ones tensor of size 2x3x3, which has tensor rank... 
    
    # Wait, I'm confusing myself. The matrix multiplication tensor for A(n x m) * B(m x p) -> C(n x p)
    # has entries T[i,k,j] = 1 (or the product structure). The standard tensor is:
    # T[i,j,k] = delta (just 1) for all valid indices, because C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # So T is the all-1 tensor of size n x m x x p = 2 x 3 x 3
    # The tensor rank of an all-1 tensor of size a x b x c is... 
    
    # Actually no. The tensor rank of the all-1 tensor [1]^{a x b x c} is min(a,b,c) in some cases,
    # but generally it's more complex. For 2x3x3, let me think...
    
    # The all-1 tensor of size 2x3x3: Can we write it as sum of r rank-1 tensors?
    # A rank-1 tensor is u_i * v_j * w_k.
    # Sum_r u_i^{(r)} * v_j^{(r)} * w_k^{(r)} = 1 for all i,j,k
    
    # For this to work: for each (i,j,k), sum_r u_i^{(r)} v_j^{(r)} w_k^{(r)} = 1
    
    # One approach: fix k. For each k, the n x m matrix M_k[i,j] = sum_r u_i^{(r)} v_j^{(r)} w_k^{(r)} must be all 1s.
    # So M_k = w_k^{(r)} weighted sum of outer products u^{(r)} v^{(r)T} = ones(2,3)
    
    # This means for each k, the rank-1 matrices {w_k^{(r)} * u^{(r)} v^{(r)T}} sum to the all-ones 2x3 matrix.
    # The all-ones 2x3 matrix has rank 1, so it CAN be written as a sum of rank-1 matrices.
    # In fact, ones(2,3) = [1,1,1]^T * [1,1] (outer product of a 2x1 and 1x3 vector) -- wait that's 2x3.
    # ones(2,3) = col2 * row3 where col2=[1,1]^T, row3=[1,1,1]
    
    # So for each k, we need: sum_r (w_k^{(r)} * u_i^{(r)}) * v_j^{(r)} = 1 for all i,j
    # Let a_r(i) = w_k^{(r)} * u_i^{(r)} for fixed k. Then sum_r a_r(i) * v_j^{(r)} = 1 for all i,j.
    
    # This is getting complex. Let me just do a direct search for the tensor rank.
    
    # Alternative approach: Just search for valid decompositions directly.
    # A decomposition of rank r gives:
    # U: (n*m) x r = 6 x r
    # V: (m*p) x r = 9 x r  
    # W: (p*n) x r = 6 x r
    # with T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For all-1 tensor: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k
    
    def verify(U, V, W, tol=0):
        """Verify decomposition. U: (n*m)xrank, V: (m*p)xrank, W: (p*n)xrank"""
        nm, rank_u = len(U), len(U[0]) if U else 0
        mp, rank_v = len(V), len(V[0]) if V else 0
        pn, rank_w = len(W), len(W[0]) if W else 0
        if rank_u != rank_v or rank_v != rank_w:
            return False
        rank = rank_u
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if isinstance(s, Fraction):
                        if s != 1:
                            return False
                    else:
                        if abs(s - 1) > tol:
                            return False
        return True
    
    def verify_half_integer(U, V, W):
        """Check all entries are integers or half-integers (multiples of 0.5)"""
        def is_half_int(x):
            if isinstance(x, Fraction):
                return (x * 2).denominator == 1
            else:
                return abs(x * 2 - round(x * 2)) < 1e-9
        for row in U:
            for x in row:
                if not is_half_int(x):
                    return False
        for row in V:
            for x in row:
                if not is_half_int(x):
                    return False
        for row in W:
            for x in row:
                if not is_half_int(x):
                    return False
        return True
    
    # Schoolbook: rank 18
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n*m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m*p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    assert verify(U_sb, V_sb, W_sb)
    best = (rank_schoolbook, U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Strategy: for small ranks, use structured ansatz
    # Key observation: the tensor is all-1s, so there's symmetry we can exploit
    
    # Try rank 15, 14, 13, 12, 11, 10, 9, 8, 7, 6
    # Lower bound: max(n, m, p) = 3, but that's very loose
    # Actually for the all-1 tensor, rank is related to the matrix flattening ranks
    
    # Flattening 1 (n x (m*p)): 2 x 9 matrix of all 1s, rank 1
    # Flattening 2 (m x (n*p)): 3 x 6 matrix of all 1s, rank 1  
    # Flattening 3 (p x (n*m)): 3 x 6 matrix of all 1s, rank 1
    # So matrix rank lower bound is 1, very loose
    
    # The actual tensor rank of the all-1 tensor [1]^{a x b x c}:
    # Known result: rank of [1]^{a x b x c} = min over partitions... 
    # For 2x3x3: Let me think structurally.
    
    # Ansatz: Use the fact that ones(2,3) = [1,1]^T [1,1,1]
    # For each k, we need sum_c (W[k*n+i, c] * U[i*m+j, c]) * V[j*p+k, c] = 1
    
    # Let me try a parametric approach with small integer/half-integer values
    # Search space: values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    # This is 9^r which is huge, so we need to be smart
    
    # Better approach: Use the structure. Since the tensor is all-1s,
    # we can try decompositions where many entries share structure.
    
    # Try: Can we do rank 9?
    # ones(2,3) for each k. If we use r=9, we have 9 "modes".
    # For mode c, define: U[:,c] is 6-vector, V[:,c] is 9-vector, W[:,c] is 6-vector.
    # Constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is 18 equations (2*3*3=18) with 9*(6+9+6)=135 unknowns.
    # So rank 9 is feasible if a solution exists with the right structure.
    
    # Let me try a very specific structured ansatz.
    # Since the problem is symmetric in j (the shared dimension), 
    # and k (output), let me try:
    
    # For each (i, j, k), the term must equal 1.
    # Try: U[i*m+j, c] depends only on i and c (not j)
    #       V[j*p+k, c] depends only on j and k and c
    #       W[k*n+i, c] depends only on k and i and c
    
    # Actually, let me just try to find ANY valid decomposition with rank < 18
    # using a randomized local search starting from schoolbook.
    
    # Alternative: Try specific small-rank constructions
    
    # Rank 6 attempt: 
    # Can we use 6 modes? We have 18 equations, 6*(6+9+6)=90 unknowns.
    # The constraint is that the tensor must be all 1s.
    
    # Let me try: U[i*m+j,c] = a_i * b_{j,c}, V[j*p+k,c] = b_{j,c} * c_{k}, W[k*n+i,c] = c_k * a_i
    # Then product = a_i^2 * b_{j,c}^2 * c_k^2, sum over c = a_i^2 * c_k^2 * sum_c b_{j,c}^2
    # This must equal 1 for all i,j,k, which means a_i^2 * c_k^2 * sum_c b_{j,c}^2 = 1
    # This requires a_i^2 * c_k^2 to be independent of i and k, so a_i = const, c_k = const
    # Then const * sum_c b_{j,c}^2 = 1 for all j, so sum_c b_{j,c}^2 = 1/const for all j
    # This works! With a_i = 1/sqrt(S), c_k = 1/sqrt(S), b_{j,c} such that sum_c b_{j,c}^2 = S
    # But we need rational/half-integer entries, so sqrt won't work nicely.
    
    # Let me try: a_i = 1, c_k = 1, then sum_c b_{j,c}^2 = 1 for all j
    # With 6 modes, can we find b_{j,c} in {0, 0.5, 1} such that sum_c b_{j,c}^2 = 1?
    # b_{j,c} in {0, 0.5, 1}: b^2 in {0, 0.25, 1}
    # Need sum of 6 values from {0, 0.25, 1} to equal 1.
    # Options: one 1 and rest 0, or four 0.25s and rest 0.
    # So for each j, either one b_{j,c}=1 and rest 0, or four b_{j,c}=0.5 and rest 0.
    
    # If we use the "one 1" pattern: for each j, pick a distinct c.
    # j=0: b_{0,0}=1, j=1: b_{1,1}=1, j=2: b_{2,2}=1
    # Then U[i*m+j,c] = 1 if c==j else 0 (a_i=1)
    # V[j*p+k,c] = 1 if c==j else 0 (c_k=1)  
    # W[k*n+i,c] = 1 (a_i=1, c_k=1)
    # Product: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if c==j, else 0
    # Sum over c: 1 (since exactly one c equals j)
    # Wait, this works! Rank 3!
    
    # Let me verify: rank = 3
    # U is 6x3, V is 9x3, W is 6x3
    # U[i*m+j, c] = 1 if c == j, else 0  (for i in {0,1}, j in {0,1,2}, c in {0,1,2})
    # V[j*p+k, c] = 1 if c == j, else 0  (for j in {0,1,2}, k in {0,1,2}, c in {0,1,2})
    # W[k*n+i, c] = 1 for all (k, i, c)  (for k in {0,1,2}, i in {0,1}, c in {0,1,2})
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c (1 if c==j) * (1 if c==j) * 1
    # = sum_c 1_{c=j} * 1_{c=j}
    # = 1_{j=j} = 1  (since j is in {0,1,2} which is the range of c)
    
    # YES! This works with rank 3!
    
    # Wait, let me double check the dimensions and indices.
    # n=2, m=3, p=3
    # U: (n*m) x rank = 6 x 3. U[i*m+j, c] where i in 0..1, j in 0..2, c in 0..2
    # V: (m*p) x rank = 9 x 3. V[j*p+k, c] where j in 0..2, k in 0..2, c in 0..2
    # W: (p*n) x rank = 6 x 3. W[k*n+i, c] where k in 0..2, i in 0..1, c in 0..2
    
    # The product: C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_j sum_c (1 if c==j) * (1 if c==j) * 1
    # = sum_j sum_c 1_{c=j}^2
    # = sum_j 1  (since for each j, there's exactly one c=j)
    # = 3
    
    # Oops! That gives 3, not 1. The sum over j gives 3.
    
    # I need the total sum to be 1, not 3. Let me fix:
    # Use U[i*m+j,c] = 1/3 if c==j else 0? But 1/3 is not half-integer.
    # Use U[i*m+j,c] = 0.5? No, that doesn't work either.
    
    # Actually wait - I need to re-examine. The tensor entry T[i,j,k] should be 1,
    # and the decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # So for each specific (i,j,k), the sum over c must equal 1.
    # With my construction: for fixed (i,j,k):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c 1_{c=j} * 1_{c=j} * 1
    # = 1_{j is in 0..2} * 1 = 1  (since j is indeed in 0..2)
    
    # Wait, I think I was confusing myself. The decomposition verifies per (i,j,k), not summed over j.
    # Let me re-read the problem.
    
    # "U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T is defined by:
    # T[i,j,k] = 1 (for the standard matrix multiplication, every product term A[i,j]*B[j,k] appears)
    # 
    # And the decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for each (i, j, k) triple, the sum over c must equal 1.
    
    # With my construction:
    # For fixed (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # U[i*m+j,c] = 1 if c==j else 0
    # V[j*p+k,c] = 1 if c==j else 0  
    # W[k*n+i,c] = 1
    # So the product is 1 if c==j, else 0. Sum over c in 0..2: equals 1 (since j in 0..2).
    
    # THIS IS CORRECT! The rank-3 decomposition works!
    
    U3 = [[0]*3 for _ in range(6)]
    V3 = [[0]*3 for _ in range(9)]
    W3 = [[0]*3 for _ in range(6)]
    
    for i in range(2):
        for j in range(3):
            U3[i*3+j][j] = 1
    for j in range(3):
        for k in range(3):
            V3[j*3+k][j] = 1
    for k in range(3):
        for i in range(2):
            for c in range(3):
                W3[k*2+i][c] = 1
    
    if verify(U3, V3, W3):
        if verify_half_integer(U3, V3, W3):
            return {"U": U3, "V": V3, "W": W3}
    
    # If rank 3 somehow failed (shouldn't), fall back to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

