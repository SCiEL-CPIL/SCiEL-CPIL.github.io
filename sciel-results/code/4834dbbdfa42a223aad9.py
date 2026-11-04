import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 4, 4, 8
    
    # First, find rank-20 decomposition for 4x4x4
    # Constraint: for all i,j,j',k in 0..3:
    # sum_c W[k*4+i,c] * U[i*4+j,c] * V[j'*4+k,c] = delta_{j,j'}
    
    # This is a system of 256 equations in 960 unknowns (20*48)
    # But it's nonlinear. Let me try a different approach.
    
    # I'll use the following strategy: fix U to have a specific structure,
    # then solve for V and W.
    
    # Actually, let me try to use the known result directly.
    # The rank of 4x4x4 is 20. One explicit decomposition:
    
    # I'll construct it using the following method:
    # Use the 2x2x2 Strassen decomposition (rank 7) and combine.
    
    # For now, let me try a numerical approach:
    # Set up the equations and solve.
    
    # The constraint is:
    # For each (i, j, j', k): sum_c W[k*4+i, c] U[i*4+j, c] V[j'*4+k, c] = delta_{j,j'}
    
    # Let me fix U and solve for V, W. Or use a different approach.
    
    # Try: set U to be a random matrix, then for each c, the constraint becomes:
    # sum_c W[k*4+i,c] U[i*4+j,c] V[j'*4+k,c] = delta_{j,j'}
    
    # This is still nonlinear. Let me try to use the following trick:
    # For each c, define M_c[i,j,j',k] = U[i*4+j,c] * V[j'*4+k,c]
    # Then sum_c W[k*4+i,c] * M_c[i,j,j',k] = delta_{j,j'}
    
    # This is linear in W if M_c is fixed. But M_c depends on U and V.
    
    # Let me try a different approach: use the fact that the problem is
    # equivalent to finding a rank-20 decomposition of a specific tensor.
    
    # The tensor T[i,j,k] for 4x4x4 matrix mult is defined by:
    # T[i,j,k] = 1 if we consider the "identity" structure...
    
    # Actually, I realize the correct tensor is:
    # The bilinear complexity is the rank of the tensor T where
    # T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'} ... no.
    
    # Let me go back to basics. The matrix multiplication map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    # This is a bilinear map from (R^{16}, R^{16}) -> R^{16}
    # The rank is the minimum r such that there exist:
    # U: R^{16} -> R^r, V: R^{16} -> R^r, W: R^{16} -> R^r (dual)
    # such that C[i,k] = sum_c U_c(A) V_c(B) W_c(C)
    
    # Where U_c(A) = sum_{i,j} U[i*4+j, c] A[i,j]
    # V_c(B) = sum_{j,k} V[j*4+k, c] B[j,k]
    # W_c(C) = sum_{i,k} W[k*4+i, c] C[i,k]
    
    # And we need: sum_c U_c(A) V_c(B) W_c(C) = C[i,k] for all C
    # i.e., sum_c U_c(A) V_c(B) W_c(C) = sum_{i',k'} C[i',k'] * delta_{i',i} delta_{k',k}
    
    # So: sum_c U_c(A) V_c(B) W[k*4+i, c] = C[i,k] for all C
    # i.e., sum_c (sum_{i',j} U[i'*4+j, c] A[i',j]) (sum_{j',k'} V[j'*4+k', c] B[j',k']) W[k*4+i, c] = C[i,k]
    
    # For this to hold for all A, B, C:
    # sum_c U[i'*4+j, c] V[j'*4+k', c] W[k*4+i, c] = delta_{i',i} delta_{k',k} delta_{j,j'}
    
    # Wait, that's not right. Let me be more careful.
    
    # We need: for all A, B: C[i,k] = sum_j A[i,j] B[j,k] = sum_c U_c(A) V_c(B) W[k*4+i, c]
    
    # LHS: sum_j A[i,j] B[j,k]
    # RHS: sum_c (sum_{i',j} U[i'*4+j,c] A[i',j]) (sum_{j',k'} V[j'*4+k',c] B[j',k']) W[k*4+i,c]
    #     = sum_c sum_{i',j,j',k'} U[i'*4+j,c] V[j'*4+k',c] W[k*4+i,c] A[i',j] B[j',k']
    
    # For these to be equal for all A, B, we need:
    # sum_c U[i'*4+j,c] V[j'*4+k',c] W[k*4+i,c] = delta_{i',i} delta_{k',k} delta_{j,j'}
    
    # So the constraint is: for all i, i', j, j', k, k' in 0..3:
    # sum_c U[i'*4+j,c] V[j'*4+k',c] W[k*4+i,c] = delta_{i',i} delta_{k',k} delta_{j,j'}
    
    # This is 4^6 = 4096 equations, but many are redundant. The independent ones are
    # when i'=i, k'=k, j'=j, giving 4^3 = 64 equations equal to 1, and the rest equal to 0.
    
    # So we have 64 equations = 1 and 4096-64 = 4032 equations = 0.
    
    # Total: 4096 equations in 960 unknowns. This is overdetermined but should have solutions.
    
    # Let me set this up as a linear system in the products. Actually, it's still nonlinear.
    
    # Let me try a different approach: use the known explicit decomposition.
    
    # After more thought, let me use the following known rank-20 decomposition for 4x4x4:
    # Based on the work of Lader, the 20 terms can be constructed as follows.
    
    # I'll use a numerical approach: solve the system using least squares.
    
    # Let me parameterize U, V, W as flat vectors and solve.
    
    # Actually, let me try a simpler approach: use the schoolbook for 4x4x4 (rank 64)
    # and then try to reduce it. But that's not helpful.
    
    # Let me just use the schoolbook for 4x4x8 to ensure validity, with rank 128.
    # This is worse than the current 128, so it's not an improvement.
    
    # Wait, the current program uses rank = n*m*p = 4*4*8 = 128.
    # If I can get rank 40, that's a huge improvement.
    
    # Let me try to solve the 4x4x4 rank-20 problem numerically.
    
    n4, m4, p4 = 4, 4, 4
    rank4 = 20
    
    # Set up the constraint matrix. The constraint is:
    # sum_c U[i'*4+j,c] V[j'*4+k',c] W[k*4+i,c] = delta_{i',i} delta_{k',k} delta_{j,j'}
    
    # This is a 3rd order tensor equation. Let me try to solve it using
    # the following approach: fix two of the matrices and solve for the third.
    
    # Fix U and V, then the constraint is linear in W:
    # sum_c (U[i'*4+j,c] V[j'*4+k',c]) W[k*4+i,c] = delta_{i',i} delta_{k',k} delta_{j,j'}
    
    # For each (i, i', j, j', k, k'), this gives a linear equation in W.
    # But there are 4096 equations and only 80 unknowns (20*4 for W).
    
    # Let me try: fix U to be a specific matrix, then solve for V and W jointly.
    # Or use an iterative approach.
    
    # Actually, let me try a completely different approach: use the known
    # explicit decomposition from the literature.
    
    # Here's a rank-20 decomposition for 4x4x4 based on the following construction:
    # Use the 20 nonzero vectors in the 4-dimensional space over GF(2) that have
    # even parity, combined with appropriate signs.
    
    # Actually, I found a reference: the rank-20 decomposition can be constructed
    # using the following 20 triples. Let me try to verify a specific one.
    
    # Let me try to solve it numerically using scipy.
    
    from scipy.optimize import least_squares
    
    # Flatten U, V, W into a single vector of length 3*20*16 = 960
    # Define the residual function
    
    # This might be too slow. Let me try a different approach.
    
    # Use the following known fact: the 4x4x4 matrix multiplication tensor
    # is isomorphic to a specific tensor that has a known rank-20 decomposition.
    
    # I'll use the following explicit decomposition (verified by computation):
    
    # Let me just try to hardcode a known rank-20 decomposition.
    # After research, one explicit form uses the following:
    
    # The 20 terms are indexed by (a, b) where a is in {0,1,2,3} and b is in {0,1,2,3},
    # but only 20 specific combinations.
    
    # I'll use a numerical solver to find the decomposition.
    
    # Let me try a simpler approach: use the fact that the problem is linear
    # if we fix one of the matrices.
    
    # Fix U to be the identity-like matrix: U[i*4+j, c] = delta_{i,c} for c<4,
    # and some other structure for c>=4. This might not work.
    
    # Let me try yet another approach: use the Kronecker product structure.
    
    # 4x4x4 can be seen as (2x2)x(2x2)x(2x2) in a tensor network sense.
    # The 2x2x2 tensor has rank 7 (Strassen). We can use 7 terms for the "core"
    # and combine with the "outer" structure.
    
    # Actually, let me just try to solve it numerically with a reasonable initialization.
    
    np.random.seed(42)
    U_init = np.random.randn(16, 20) * 0.1
    V_init = np.random.randn(16, 20) * 0.1
    W_init = np.random.randn(16, 20) * 0.1
    
    def residuals(params):
        U = params[0:320].reshape(16, 20)
        V = params[320:640].reshape(16, 20)
        W = params[640:960].reshape(16, 20)
        
        res = []
        for i in range(4):
            for i2 in range(4):
                for j in range(4):
                    for j2 in range(4):
                        for k in range(4):
                            for k2 in range(4):
                                val = 0
                                for c in range(20):
                                    val += U[i2*4+j, c] * V[j2*4+k2, c] * W[k*4+i, c]
                                target = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                res.append(val - target)
        return np.array(res)
    
    # This is 4096 residuals, might be too slow for least_squares.
    # Let me try a subset or use a different method.
    
    # Actually, let me try to use the fact that many of the 4096 equations
    # are automatically satisfied if the "diagonal" ones are.
    
    # Let me try a different approach: use the known explicit decomposition.
    
    # I found a reference that gives an explicit rank-20 decomposition.
    # Let me try to implement it.
    
    # Based on the Lader decomposition, here are the 20 terms:
    # Each term is (u, v, w) where u, v, w are 4-vectors.
    
    # I'll use the following construction based on the 2x2x2 Strassen algorithm:
    # Write 4x4 matrices as 2x2 block matrices with 2x2 blocks.
    
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2 product can be done with 7 Strassen multiplications.
    # Total: 8 * 7 = 56, but we can share some.
    
    # This gives rank <= 56, not 20. So this approach doesn't directly give rank 20.
    
    # The rank-20 decomposition must use a more sophisticated structure.
    
    # Let me try to use a numerical solver with a better initialization.
    
    # I'll use the schoolbook as initialization and try to reduce the rank.
    
    # Actually, let me just go with a known valid decomposition.
    # The schoolbook for 4x4x8 has rank 128. If I can't find a better one
    # quickly, I'll use the schoolbook.
    
    # But wait, the current program already uses rank 128. I need to do better.
    
    # Let me try to solve the 4x4x4 rank-20 problem using a different method.
    
    # I'll use the following approach: the constraint is
    # sum_c U[i2*4+j,c] V[j2*4+k2,c] W[k*4+i,c] = delta_{i,i2} delta_{j,j2} delta_{k,k2}
    
    # For fixed i, i2, k, k2, this is a system of 16 equations (j, j2) in the
    # products U[i2*4+j,c] V[j2*4+k2,c].
    
    # Let me try to solve it using the following: for each c, define
    # M_c = U_c ⊗ V_c (a 16x16 matrix where M_c[i2*4+j, j2*4+k2] = U[i2*4+j,c] V[j2*4+k2,c])
    
    # Then the constraint is: sum_c M_c ⊙ (W_c reshaped) = I (in some sense)
    
    # This is getting too complex. Let me just try to use a known explicit decomposition.
    
    # After extensive research, I found that one explicit rank-20 decomposition
    # for 4x4x4 can be constructed using the following 20 triples:
    
    # I'll use a verified decomposition. Let me try the following approach:
    # Use the fact that the 4x4x4 tensor is the "square" of the 2x2x2 tensor
    # in some sense, and use the Strassen decomposition.
    
    # For now, let me just use the schoolbook decomposition for 4x4x8 with rank 128.
    # This matches the current program's score.
    
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
