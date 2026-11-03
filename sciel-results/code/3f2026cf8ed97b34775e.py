import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook rank = 84
    # We will use Strassen for 2x2x2 blocks and schoolbook for the rest.
    
    # Define the blocks:
    # i in {0,1} (rows 0,1 of A)
    # j in {0,1} and {2,3} (cols 0,1 and 2,3 of A / rows of B)
    # k in {0,1}, {2,3}, {4,5} (cols 0,1; 2,3; 4,5 of B / cols of C)
    
    # This gives 2 * 2 * 3 = 12 blocks of size 2x2x2.
    # Wait, m=4 so j has 4 values. We can split into {0,1} and {2,3}.
    # p=7 so k has 7 values. We can split into {0,1}, {2,3}, {4,5} and {6}.
    # So for j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}, we have 2x2x2 blocks.
    # That's 2 (j-groups) * 3 (k-groups) = 6 blocks of size 2x2x2.
    # The remaining elements are:
    # - i in {2} (row 2 of A) -> all j, k (4 * 7 = 28 elements)
    # - i in {0,1}, j in {0,1},{2,3}, k in {6} (2 * 4 * 1 = 8 elements)
    # Total schoolbook elements = 28 + 8 = 36.
    # Total Strassen elements = 6 * 4 = 24 (each 2x2x2 block has 8 schoolbook, 7 Strassen).
    # Wait, 6 blocks * 8 tuples = 48 tuples covered by Strassen.
    # Each Strassen block replaces 8 schoolbook with 7.
    # So we save 6 * 1 = 6 from the 48 tuples.
    # Total rank = 84 - 6 = 78.
    
    # Let's construct the Strassen decomposition for each block.
    # Standard Strassen for 2x2 matrices:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor decomposition, each M corresponds to a rank-1 component.
    # For a block with local indices (i_loc, j_loc, k_loc) in {0,1}:
    # U[i*m + j] gets the coefficient of A[i_loc, j_loc]
    # V[j*p + k] gets the coefficient of B[j_loc, k_loc]
    # W[k*n + i] gets the coefficient of C[k_loc, i_loc]  (note: trace convention W[k*n+i])
    
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This means C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, standard matrix multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = A[i,j] * B[j,k] * C[k,i] ? No.
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Here T[i,j,k] is 1 if we are computing C[i,k] from A[i,j]*B[j,k], else 0.
    # Actually, the matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j sum_r A[i,j] * U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k? No.
    # It should equal 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Yes, it always contributes. So T[i,j,k] = 1 for all i,j,k?
    # No, that's not right. The tensor decomposition decomposes the *multiplication map*.
    # The map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # This is a trilinear map. The tensor is T with T[i,j,k] = 1.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is at least max(n*m, m*p, p*n).
    # But schoolbook has rank n*m*p.
    # Let's re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor <n,m,p> has entries 1 at (i,j,k) for all i,j,k?
    # No, that would be the all-ones tensor.
    # The matrix multiplication tensor is defined by the bilinear form.
    # In coordinates, the tensor T has T[i,j,k] = 1 if the product A[i,j] B[j,k] appears in C[i,k].
    # Since it always appears, T[i,j,k] = 1 for all i,j,k?
    # If so, the rank of the all-ones tensor of size n x m x p is max(n,m,p).
    # But schoolbook is n*m*p. So that's not it.
    
    # Let's check the seed program.
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # This means for each (i,j,k), we have a rank-1 tensor with 1 at (i,j,k).
    # So the sum of these rank-1 tensors is the tensor with 1 at every (i,j,k).
    # So yes, the matrix multiplication tensor is the all-ones tensor?
    # No, that can't be right because then Strassen wouldn't work.
    
    # Let's look at the definition of matrix multiplication tensor more carefully.
    # The tensor <n,m,p> is in U* ⊗ V* ⊗ W* where dim U = n*m, dim V = m*p, dim W = p*n.
    # The entries are T[(i,j), (j,k), (k,i)]? No.
    # The bilinear map is m: U ⊗ V -> W, where U is space of n x m matrices, V is space of m x p matrices, W is space of n x p matrices.
    # m(A,B)[i,k] = sum_j A[i,j] B[j,k].
    # As a trilinear form, it's T(A,B,C) = sum_{i,j,k} A[i,j] B[j,k] C[k,i].
    # So the tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # Wait, if T is the all-ones tensor, its rank is max(n*m, m*p, p*n)?
    # No, the rank of the all-ones tensor of size a x b x c is max(a,b,c) if we allow complex entries?
    # Actually, the rank of the all-ones tensor is 1 if all dimensions are 1.
    # For a x b x c, the rank is not simply max.
    # But schoolbook is a*b*c.
    # So the tensor is NOT the all-ones tensor.
    
    # Let's re-examine the seed program.
    # It sets U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # This decomposes the tensor that has 1 at (i*m+j, j*p+k, k*n+i) for each (i,j,k).
    # This is a permutation of the all-ones tensor on the product space.
    # Specifically, it's the tensor T such that T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k.
    # This is exactly the matrix multiplication tensor in these coordinates.
    
    # So for Strassen, we need to replace 8 such basis tensors with 7.
    # The 8 basis tensors correspond to (i_loc, j_loc, k_loc) in {0,1}^3.
    # Strassen's algorithm computes the product of two 2x2 matrices using 7 multiplications.
    # This corresponds to a decomposition of the 2x2x2 matrix multiplication tensor into 7 rank-1 tensors.
    # We need to find these 7 rank-1 tensors.
    
    # The standard Strassen decomposition for the 2x2x2 tensor (which is the all-ones tensor on 2x2x2? No, it's the specific tensor for 2x2 matrix mult):
    # T2x2x2 has T[i,j,k] = 1 for i,j,k in {0,1}.
    # Wait, is the 2x2x2 matrix multiplication tensor the all-ones tensor on {0,1}^3?
    # Yes! Because C[i,k] = sum_j A[i,j] B[j,k], and for 2x2, the tensor is T[i,j,k] = 1.
    # So the 2x2x2 matrix multiplication tensor IS the all-ones tensor on 2x2x2.
    # And its rank is 7 (Strassen).
    # The all-ones tensor on 2x2x2 has rank 7?
    # Yes, that's a well-known result.
    
    # So we can use the known Strassen decomposition for the 2x2x2 all-ones tensor.
    # The 7 rank-1 components are:
    # 1. (1,1,1) * (1,1,1) * (1,1,1)  -> (1,1,1) all ones? No.
    # Let's use the explicit Strassen formulas.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Each M is a product of a linear form in A and a linear form in B.
    # M1 = (a00+a11) * (b00+b11)
    # This corresponds to U vector: [1, 0, 0, 1] (for a00, a01, a10, a11)
    # V vector: [1, 0, 0, 1] (for b00, b01, b10, b11)
    # W vector: [1, 1, 1, 1] (for c00, c01, c10, c11)
    # Because M1 contributes to C00, C01, C10, C11 with coefficients 1, -1, 0, 1?
    # No, M1 = C00 + C11 - C01 - C10? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes +1 to C00 and +1 to C11.
    # So W1 = [1, 0, 0, 1] (for c00, c01, c10, c11)
    
    # Let's list all 7:
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # M3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # M4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,0,0]
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[0,-1,0,0]  # Wait, C01 = M3+M5, C00 has -M5
    #    So M5 contributes -1 to C00, +1 to C01.
    #    W5 = [-1, 1, 0, 0]
    # M6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,1]  # C11 has +M6
    # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]  # C00 has +M7
    
    # Let's verify M5:
    # M5 = (a00+a01)b11
    # C00 = M1 + M4 - M5 + M7 -> coeff of M5 is -1
    # C01 = M3 + M5 -> coeff of M5 is +1
    # So W5 = [-1, 1, 0, 0]
    
    # Let's verify M7:
    # M7 = (a01-a11)(b10+b11)
    # C00 = ... + M7 -> coeff +1
    # So W7 = [1, 0, 0, 0]
    
    # So the 7 components are:
    # 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # 4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,0,0]
    # 5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    # 6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # 7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    # Now we map these to the global indices.
    # For a block with i0, j0, k0 (base indices):
    # Local i_loc in {0,1} -> global i = i0 + i_loc
    # Local j_loc in {0,1} -> global j = j0 + j_loc
    # Local k_loc in {0,1} -> global k = k0 + k_loc
    
    # U global index: i * m + j
    # V global index: j * p + k
    # W global index: k * n + i
    
    # Let's construct the full decomposition.
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for a 2x2x2 block
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,-1,1,0],
        [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [0,0,1,-1],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [-1,1,0,0],
        [0,0,0,1],
        [1,0,0,0]
    ]
    
    col = 0
    # Blocks: i0=0, j0 in {0,2}, k0 in {0,2,4}
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                for s in range(7):
                    # Add Strassen component s
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = strassen_U[s][i_loc * 2 + j_loc]
                            if val != 0:
                                i = i0 + i_loc
                                j = j0 + j_loc
                                U[i * m + j][col] += val
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = strassen_V[s][j_loc * 2 + k_loc]
                            if val != 0:
                                j = j0 + j_loc
                                k = k0 + k_loc
                                V[j * p + k][col] += val
                    for k_loc in range(2):
                        for i_loc in range(2):
                            val = strassen_W[s][k_loc * 2 + i_loc]
                            if val != 0:
                                k = k0 + k_loc
                                i = i0 + i_loc
                                W[k * n + i][col] += val
                    col += 1
    
    # Schoolbook for remaining
    # Remaining (i,j,k):
    # i=2, all j,k (4*7=28)
    # i in {0,1}, all j, k=6 (2*4*1=8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 or i == 1:
                    if k == 6:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
                else:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    # Verify col == rank
    # assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
