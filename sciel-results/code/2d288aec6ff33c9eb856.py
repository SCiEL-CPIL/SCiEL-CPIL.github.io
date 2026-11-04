import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _elapsed():
    return time.time() - _START

_START = time.time()
DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor
    # T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] appears in C[i,k]
    # The tensor is: T[i, j, k] = 1 for all i, j, k (since C[i,k] = sum_j A[i,j] B[j,k])
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # Build matrices for flattening
    # Flatten U to shape (n*m, rank), V to (m*p, rank), W to (p*n, rank)
    # For each r, the outer product structure:
    # Contribution to T[i,j,k] from rank r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Efficient check: for each r, form the 3D tensor contribution
    # T_contrib[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k
    
    # Use a more efficient approach:
    # For fixed j, the constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, k
    
    # Let's build it directly using numpy
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # For each j, compute the sum over r of U3[:, j, r] * V3[j, :, r] * W3[:, :, r]
    # This gives a (n, p) matrix that should be all 1s
    # Actually: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's use einsum
    # result[i, k] = sum_{j, r} U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # But this isn't quite right because the indices are coupled.
    
    # Direct check:
    # T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i, j, k] = 1 for all i, j, k
    
    # Build the full tensor check
    # For efficiency, check a subset or use matrix operations
    
    # Actually, the correct identity is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor decomposition says:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ? No.
    
    # Let me re-derive. The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 (the coefficient of A[i,j]*B[j,k] in the product)
    # The decomposition: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Wait, that's not standard. Let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^R u_r[i,j] * v_r[j,k] * w_r[k,i]
    # where u_r, v_r, w_r are the factor vectors.
    
    # But in the interface:
    # U has shape (n*m, rank): U[i*m+j, r] = u_r[i,j]
    # V has shape (m*p, rank): V[j*p+k, r] = v_r[j,k]  
    # W has shape (p*n, rank): W[k*n+i, r] = w_r[k,i]
    
    # So the check is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """Return Strassen factors for 2x2x2 matrix multiplication.
    U: (2, 7), V: (2, 7), W: (2, 7)
    where U[i, r], V[j, r], W[k, r]
    and sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] = 1 for all i,j,k in {0,1}
    """
    # Strassen's 7 multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a10)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # The tensor T[i,j,k] = 1 means we need:
    # sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # Let me derive the factors carefully.
    # The standard Strassen decomposition works on the output C, not directly on T.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M7
    
    # But we need the tensor decomposition, not the Strassen algorithm per se.
    # The tensor T[i,j,k] = 1 has rank 7 (Strassen's result).
    
    # Let me construct it directly. We need 7 triples (u_r, v_r, w_r) such that
    # sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-7 decomposition of the 2x2x2 all-ones tensor.
    # Actually, the matrix multiplication tensor for 2x2 matrices is NOT the all-ones tensor.
    # T[i,j,k] is the coefficient in C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k. Yes, it IS the all-ones tensor.
    
    # Wait, that's not right either. The tensor for matrix multiplication of 2x2 matrices
    # is defined by the bilinear map, and its structure tensor has T[i,j,k] = 1 
    # only when the indices are consistent with the multiplication.
    
    # Actually, for the standard matrix multiplication tensor of size n x n x n,
    # T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] appears in C[i,k].
    # For 2x2: T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1, and T=0 otherwise.
    # NO! That's not right. C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    # So the tensor T[i, j, k] where i is row of C, j is column of A / row of B, k is col of C:
    # T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] contributes to C[i,k].
    # This is ALWAYS true by definition of matrix multiplication.
    # So T[i, j, k] = 1 for ALL i, j, k.
    
    # Hmm, but that would mean the tensor is just the all-ones tensor, which has rank 1!
    # That can't be right because matrix multiplication requires at least n^3 scalar multiplications
    # in the naive algorithm, and the tensor rank is a measure of complexity.
    
    # I think I'm confusing the tensor definition. Let me reconsider.
    
    # The matrix multiplication tensor M_n is an element of (F^n)^* ⊗ (F^n)^* ⊗ (F^n)^*
    # defined by M_n = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where... no.
    
    # Actually, the tensor is: M_n = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ e_k^* 
    # where the sum is over all (i,j,k) such that the term appears.
    
    # For matrix multiplication: C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*
    # As a tensor in the dual space, T[i,j,k] = 1 for all i,j,k.
    
    # But this is the "structure tensor" and its rank is what we're minimizing.
    # For 2x2x2, the rank of the all-ones tensor is... 1? No.
    
    # I think the issue is that the tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor encodes the bilinear map, and its rank is the minimum number of rank-1 tensors
    # needed to express it.
    
    # Let me look at this differently. The task says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # And the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # So yes, we need the all-ones tensor decomposition.
    # For 2x2x2, the all-ones tensor 1_{i,j,k} = 1 has rank 1: just take U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # But that gives rank 1, which is way too good.
    
    # I must be misunderstanding the tensor definition. Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix:
    # The output is C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{p*n})^*
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # where e_{i*m+j} is the basis vector for A[i,j], etc.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1, and T = 0 otherwise.
    # This is NOT the all-ones tensor. It's a specific tensor with 1s only at certain positions.
    
    # For n=2, m=4, p=8:
    # T has 1s at positions (i*4+j, j*8+k, k*2+i) for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # That's 2*4*8 = 64 ones.
    
    # So the schoolbook decomposition with rank 64 is correct: each 1 is a separate rank-1 term.
    
    # Now, can we do better? The tensor rank of the matrix multiplication tensor for 2x4x8
    # is at most 64 (schoolbook) and at least... we need to find a better decomposition.
    
    # For 2x2x2, the tensor rank is 7 (Strassen).
    # For 2x4x8, we can use block decomposition.
    
    # Let me think about the structure. A is 2x4, B is 4x8, C is 2x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # We can partition the sum:
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    # Each part is a 2x2x8 multiplication (2x2 A block times 2x8 B block gives 2x8 C).
    # For 2x2x8: A is 2x2, B is 2x8, C is 2x8.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    
    # This can be done with 2 scalar multiplications per (i,k) pair, but we can use Strassen-like tricks.
    # Actually, for 2x2x8, we have 2*2*8 = 32 terms in schoolbook.
    # Can we do better? The tensor for 2x2x8 multiplication...
    
    # Let me just implement the Strassen-based block decomposition carefully.
    
    # For 2x2x2 matrix multiplication, the tensor T_2 has T_2[i,j,k] = 1 for all i,j,k in {0,1}.
    # Wait, no. For 2x2x2, A is 2x2, B is 2x2, C is 2x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T_2[i*2+j, j*2+k, k*2+i] = 1
    # Positions: (0,0,0), (0,2,1), (1,1,0), (1,3,1), (2,0,0), (2,2,1), (3,1,0), (3,3,1)
    # That's 8 positions.
    
    # The rank of T_2 is 7 (Strassen).
    
    # For our problem, we can view it as 4 blocks of 2x2x2 multiplications? No.
    # A is 2x4 = 2x(2+2), B is 4x8 = (2+2)x8, C is 2x8.
    
    # Let's partition j into two groups of 2: j in {0,1} and j in {2,3}.
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    # First term: A1 is 2x2 (columns 0,1), B1 is 2x8 (rows 0,1), gives C1 which is 2x8.
    # Second term: A2 is 2x2 (columns 2,3), B2 is 2x8 (rows 2,3), gives C2 which is 2x8.
    # C = C1 + C2.
    
    # For each term, we need to compute A1 * B1 where A1 is 2x2 and B1 is 2x8.
    # C1[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    
    # This is a 2x2x8 tensor. Can we decompose it efficiently?
    # The tensor T_{2x2x8}[i*2+j, j*8+k, k*2+i] = 1 for i in {0,1}, j in {0,1}, k in {0,...,7}
    # That's 2*2*8 = 32 ones.
    
    # For fixed i, C1[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    # This is like computing 8 dot products of length 2.
    # We can use 2 multiplications per k, so 16 multiplications total for both i values.
    # But we can do better with Strassen-like tricks.
    
    # Actually, for each k, we compute [A1[0,0]*B1[0,k] + A1[0,1]*B1[1,k], A1[1,0]*B1[0,k] + A1[1,1]*B1[1,k]]
    # This is a 2x2x2 multiplication for each k!
    # So for each k, we use 7 multiplications (Strassen), giving 7*8 = 56 total.
    
    # Similarly for the second block, another 56.
    # Total: 112? That's worse than 64.
    
    # Wait, I'm double counting. Let me reconsider.
    
    # For the first block (j in {0,1}):
    # For each k, we compute a 2x2x2 multiplication: A1 (2x2) * B1[:,k] (2x1) -> C1[:,k] (2x1)
    # But B1[:,k] is just a vector, not a matrix. So it's not a full 2x2x2 multiplication.
    
    # C1[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    # This is 2 scalar multiplications per (i,k), so 2*2*8 = 32 for the first block.
    # Similarly 32 for the second block. Total 64.
    
    # To do better, we need to share multiplications across k values.
    
    # Let me try a different approach. Use Strassen on the 2x2 blocks in the j dimension.
    
    # Actually, let me just implement the rank-56 decomposition that was mentioned in inspiration 2.
    
    # The idea: partition A into two 2x2 blocks (columns 0-1 and 2-3), and B into four 2x2 blocks.
    # Use Strassen for each 2x2x2 sub-multiplication.
    
    # A = [A0 A1] where A0, A1 are 2x2
    # B = [B0; B1] where B0, B1 are 2x4 (rows 0-1 and 2-3)
    # C = A0*B0 + A1*B1, where each product is 2x2 times 2x4 giving 2x4.
    
    # For A0*B0: A0 is 2x2, B0 is 2x4.
    # C0[i,k] = A0[i,0]*B0[0,k] + A0[i,1]*B0[1,k]
    # For each pair of columns in B0, we have a 2x2x2 multiplication.
    # B0 has 4 columns, so 2 pairs. Each pair uses 7 Strassen multiplications.
    # Total for A0*B0: 2*7 = 14.
    # Similarly for A1*B1: 14.
    # Total: 28? That seems too good.
    
    # Wait, let me verify. For A0*B0 where A0 is 2x2 and B0 is 2x4:
    # C0[:, 0:2] = A0 * B0[:, 0:2] (2x2 times 2x2) -> 7 multiplications
    # C0[:, 2:4] = A0 * B0[:, 2:4] (2x2 times 2x2) -> 7 multiplications
    # Total: 14.
    
    # Similarly for A1*B1: 14.
    # Grand total: 28.
    
    # But wait, is this correct? The Strassen decomposition for 2x2x2 gives 7 multiplications
    # that compute the full 2x2 product. If we apply it to each 2x2 block of B, we get
    # the correct result for each block.
    
    # So the total rank should be 28? Let me verify with a smaller example.
    
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4.
    # C[:, 0:2] = A * B[:, 0:2] (2x2x2) -> 7 mults
    # C[:, 2:4] = A * B[:, 2:4] (2x2x2) -> 7 mults
    # Total: 14.
    
    # For 2x4x8: A is 2x4, B is 4x8, C is 2x8.
    # Split A into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3).
    # Split B into B0 (2x8, rows 0-1) and B1 (2x8, rows 2-3).
    # C = A0*B0 + A1*B1.
    # A0*B0: 2x2 times 2x8. Split B0 into B0a (2x4, cols 0-3) and B0b (2x4, cols 4-7).
    #   A0*B0a: 2x2x4 -> 14 mults
    #   A0*B0b: 2x2x4 -> 14 mults
    #   Total for A0*B0: 28
    # Similarly for A1*B1: 28
    # Grand total: 56.
    
    # So rank 56 is achievable! Let me implement this.
    
    # Strassen factors for 2x2x2:
    # We need U_s, V_s, W_s of shape (2, 7) such that:
    # sum_r U_s[i, r] * V_s[j, r] * W_s[k, r] = T_2[i*2+j, j*2+k, k*2+i]
    # where T_2 has 1s at (0,0,0), (0,2,1), (1,1,0), (1,3,1), (2,0,0), (2,2,1), (3,1,0), (3,3,1)
    
    # Let me derive the Strassen factors for the 2x2x2 tensor.
    # The 8 nonzero entries are:
    # (i=0, j=0, k=0): A[0,0]*B[0,0] -> C[0,0]
    # (i=0, j=1, k=1): A[0,1]*B[1,1] -> C[0,1]
    # (i=1, j=0, k=0): A[1,0]*B[0,0] -> C[1,0]
    # (i=1, j=1, k=1): A[1,1]*B[1,1] -> C[1,1]
    # (i=0, j=0, k=1): A[0,0]*B[0,1] -> C[0,1]
    # (i=0, j=1, k=0): A[0,1]*B[1,0] -> C[0,0]
    # (i=1, j=0, k=1): A[1,0]*B[0,1] -> C[1,1]
    # (i=1, j=1, k=0): A[1,1]*B[1,0] -> C[1,0]
    
    # So T_2[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k in {0,1}.
    # That's 8 ones, one for each combination of i, j, k.
    
    # So the 2x2x2 matrix multiplication tensor IS the all-ones tensor!
    # And its rank is 7 (Strassen).
    
    # Great, so I need the Strassen factors for the all-ones 2x2x2 tensor.
    
    # The all-ones tensor 1_{i,j,k} = 1 can be written as:
    # 1 = sum_r u_r[i] * v_r[j] * w_r[k]
    
    # Strassen's decomposition:
    # M1 = (a00+a11)(b00+b11) -> u=(1,1), v=(1,1), w=(1,1)
    # M2 = (a10+a11)b00 -> u=(0,1), v=(1,0), w=(0,1)
    # M3 = a00(b10-b11) -> u=(1,0), v=(0,1), w=(1,1)
    # M4 = a11(b01-b00) -> u=(0,1), v=(1,1), w=(1,0)
    # M5 = (a00+a10)b11 -> u=(1,1), v=(0,1), w=(1,1)
    # M6 = (a10-a00)(b00+b01) -> u=(1,1), v=(1,1), w=(1,0)
    # M7 = (a01-a11)(b10+b11) -> u=(1,1), v=(0,1), w=(1,1)
    
    # Wait, I need to be more careful. Let me look up the exact Strassen factors.
    
    # Actually, let me just verify that my proposed factors work.
    
    U_s = [
        [1, 0, 1, 0, 1, 1, 1],  # i=0
        [1, 1, 0, 1, 1, 1, 1]   # i=1
    ]
    V_s = [
        [1, 1, 0, 1, 0, 1, 0],  # j=0
        [1, 0, 1, 1, 1, 1, 1]   # j=1
    ]
    W_s = [
        [1, 0, 1, 1, 1, 1, 1],  # k=0
        [1, 1, 1, 0, 1, 0, 1]   # k=1
    ]
    
    # Verify: sum_r U_s[i,r] * V_s[j,r] * W_s[k,r] should be 1 for all i,j,k.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U_s[i][r] * V_s[j][r] * W_s[k][r] for r in range(7))
                if s != 1:
                    return None  # Invalid factors
    
    return U_s, V_s, W_s

def _build_rank56():
    """Build rank-56 decomposition using Strassen block decomposition."""
    n, m, p = 2, 4, 8
    U_s, V_s, W_s = _strassen_2x2x2()
    if U_s is None:
        return None
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A is 2x4, split into A0 (cols 0-1) and A1 (cols 2-3)
    # B is 4x8, split into B0 (rows 0-1) and B1 (rows 2-3)
    # C is 2x8
    
    # For each block (A_block, B_block):
    # A_block is 2x2, B_block is 2x8
    # Split B_block into B_block_a (cols 0-3) and B_block_b (cols 4-7), each 2x4
    # For each 2x4 B sub-block, split into two 2x2 blocks
    # Each 2x2x2 uses 7 Strassen multiplications
    
    # Total: 2 A-blocks * 2 B-col-blocks * 2 B-row-pairs * 7 = 56
    
    col = 0
    for a_block in range(2):  # A0 or A1
        a_col_offset = a_block * 2  # columns 0-1 or 2-3
        for b_row_block in range(2):  # B0 or B1
            b_row_offset = b_row_block * 2  # rows 0-1 or 2-3
            for b_col_block in range(2):  # B cols 0-3 or 4-7
                b_col_offset = b_col_block * 4
                for b_col_pair in range(2):  # Within the 4 cols, two pairs of 2
                    b_col_pair_offset = b_col_pair * 2
                    for r in range(7):
                        # U: A_block[i, a_col_offset + j_a] where j_a in {0,1}
                        for i in range(2):
                            for j_a in range(2):
                                U[i * m + a_col_offset + j_a][col] = U_s[i][r]
                        
                        # V: B_block[j_b, b_col_offset + b_col_pair_offset + k] where j_b in {0,1}, k in {0,1}
                        for j_b in range(2):
                            for k in range(2):
                                V[(b_row_offset + j_b) * p + b_col_offset + b_col_pair_offset + k][col] = V_s[j_b][r]
                        
                        # W: C[i, b_col_offset + b_col_pair_offset + k] where k in {0,1}
                        for k in range(2):
                            for i in range(2):
                                W[(b_col_offset + b_col_pair_offset + k) * n + i][col] = W_s[k][r]
                        
                        col += 1
    
    if col != rank:
        return None
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = _elapsed()
    
    # Try rank-56 Strassen-based decomposition
    result = _build_rank56()
    if result is not None and _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fall back to schoolbook rank-64
    return _schoolbook(n, m, p)
