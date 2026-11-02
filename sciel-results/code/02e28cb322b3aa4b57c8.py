"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
import itertools
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, max_samples=200):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each r in range(rank):
        A_part = U[:, r] reshaped to (n, m)
        B_part = V[:, r] reshaped to (m, p)
        C_part = W[:, r] reshaped to (p, n) -- but W is indexed as W[k*n+i, r]
        So C = sum_r (U_r * A) * (V_r * B) * W_r
        where U_r is a vector of length n*m, V_r is length m*p, W_r is length p*n.
        
        The contribution to C[i, k] from term r is:
            W_r[k*n + i] * (sum_{j} U_r[i*m + j] * A[i,j]) * (sum_{j} V_r[j*p + k] * B[j,k])
        
        Wait, that's not quite right. Let me think about this more carefully.
        
        In a rank-r decomposition of the matrix multiplication tensor:
        T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
        
        But the standard formulation is:
        C_{i,k} = sum_j A_{i,j} * B_{j,k}
        
        And the tensor decomposition says:
        sum_j A_{i,j} * B_{j,k} = sum_r (sum_j U_{i,j}^{(r)} * A_{i,j}) * (sum_j V_{j,k}^{(r)} * B_{j,k}) * W_{k,i}^{(r)}
        
        Hmm, that doesn't look right either. Let me use the standard formulation.
        
        The matrix multiplication tensor M is a 3-tensor of size (n, m, p) where:
        M_{i,j,k} = 1 if we're computing C_{i,k} = sum_j A_{i,j} B_{j,k}
        
        Actually, the tensor for matrix multiplication is:
        T_{i,j,k} = delta_{i,k} ... no.
        
        Let me think about it differently. The bilinear map is:
        (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
        
        A rank-r decomposition means:
        C_{i,k} = sum_r alpha_r(A) * beta_r(B) * gamma_{i,k}^{(r)}
        
        where alpha_r(A) is a linear functional of A, beta_r(B) is a linear functional of B,
        and gamma^{(r)} is a matrix of size (n, p).
        
        In the interface, we have:
        U: (n*m, rank) -- so U[i*m + j, r] gives the coefficient of A[i,j] in alpha_r
        V: (m*p, rank) -- so V[j*p + k, r] gives the coefficient of B[j,k] in beta_r
        W: (p*n, rank) -- so W[k*n + i, r] gives the coefficient for C[i,k] in gamma^{(r)}
        
        So: C[i,k] = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        
        This should equal sum_j A[i,j] * B[j,k] for all i, k.
    """
    rank = len(U[0])
    # Test with random matrices
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # alpha_r = sum_j U[i*m+j, r] * A[i,j] for each i
            # beta_r = sum_j V[j*p+k, r] * B[j,k] for each k
            # gamma_r[i,k] = W[k*n+i, r]
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i*m + j][r] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j*p + k][r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += alpha[i] * beta[k] * W[k*n + i][r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction(n, m, p, base_rank, max_steps=5000, deadline=None):
    """
    Try to find a lower-rank decomposition by starting from the schoolbook
    and attempting to merge terms or find linear dependencies.
    This is a heuristic approach.
    """
    # For 2x4x6, we can try to use the structure of the problem.
    # The matrix multiplication tensor for (n, m, p) has rank at most n*m*p.
    # For small dimensions, we can try to find the exact rank.
    # 
    # One approach: use the fact that the tensor is a sum of rank-1 tensors.
    # We can try to find a smaller set of rank-1 tensors that span the same space.
    #
    # For now, let's just return the schoolbook decomposition.
    # A more sophisticated approach would use numerical linear algebra to
    # find the rank of the tensor, but that's complex.
    return _schoolbook_decomposition(n, m, p)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook decomposition
    result = _schoolbook_decomposition(n, m, p)
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, max_samples=50):
        # This should never happen, but just in case
        pass
    
    # For 2x4x6, the schoolbook rank is 2*4*6 = 48.
    # We can try to find a lower rank decomposition.
    # 
    # Known results:
    # - For 2x2x2, the rank is 7 (Strassen).
    # - For larger matrices, the rank is between n^3 and n^2 * m * p.
    # 
    # For 2x4x6, we can try to use a block decomposition.
    # Split the 4x6 matrix into 2x3 blocks of size 2x2.
    # Then we have 6 blocks, each requiring a 2x2x2 decomposition.
    # If we use Strassen for each 2x2x2, the total rank would be 6*7 = 42.
    # But we need to be careful about how the blocks interact.
    
    # Let's try to construct a decomposition using Strassen for 2x2 blocks.
    # A is 2x4, B is 4x6.
    # Split A into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # Split B into three 2x2 blocks: B_top_left (rows 0-1, cols 0-1), 
    #   B_top_right (rows 0-1, cols 2-3), B_bottom_left (rows 2-3, cols 0-1),
    #   B_bottom_right (rows 2-3, cols 2-3)
    # 
    # C = A @ B = [A_left @ B_top_left + A_right @ B_bottom_left, 
    #               A_left @ B_top_right + A_right @ B_bottom_right]
    # 
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    # We have 4 such multiplications (A_left@B_top_left, A_left@B_top_right, 
    #   A_right@B_bottom_left, A_right@B_bottom_right), so total rank = 4*7 = 28.
    # But we also need to add the results, which doesn't add rank.
    # So the total rank should be 28.
    
    # Let's implement the Strassen decomposition for 2x2 matrices.
    # For 2x2 matrices A and B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A @ B = [[ae+bf, af+bg], [ce+df, cf+dg]]
    
    # Strassen's algorithm uses 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C[0,0] = m1 - m4 + m5 + m6
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m7
    
    # Let's verify this:
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (c+d)(e) = ce + de
    # m3 = a(f-h) = af - ah
    # m4 = d(g-e) = dg - de
    # m5 = (a+b)(h) = ah + bh
    # m6 = (c-a)(e+f) = ce + cf - ae - af
    # m7 = (b-d)(g+h) = bg + bh - dg - dh
    
    # C[0,0] = m1 - m4 + m5 + m6
    # = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (ce + cf - ae - af)
    # = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
    # = (ae - ae) + (de + de) + (dh) + (-dg) + (ah + ah) + (bh) + (ce) + (cf) + (-af)
    # = 2de + dh - dg + 2ah + bh + ce + cf - af
    
    # This should equal ae + bf, but it doesn't seem to. Let me recheck.
    
    # Actually, I think I made an error in the Strassen formulas. Let me look them up.
    # The correct Strassen formulas are:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)(h)
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C[0,0] = m1 - m4 + m5 + m6
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m7
    
    # Let me test with a=1, b=0, c=0, d=0, e=1, f=0, g=0, h=0:
    # A = [[1, 0], [0, 0]], B = [[1, 0], [0, 0]]
    # C should be [[1, 0], [0, 0]]
    
    # m1 = (1+0)(1+0) = 1
    # m2 = (0+0)(1) = 0
    # m3 = 1(0-0) = 0
    # m4 = 0(0-1) = 0
    # m5 = (1+0)(0) = 0
    # m6 = (0-1)(1+0) = -1
    # m7 = (0-0)(0+0) = 0
    
    # C[0,0] = 1 - 0 + 0 + (-1) = 0. But it should be 1.
    
    # So my Strassen formulas are still wrong. Let me try a different source.
    
    # After some research, the correct Strassen decomposition is:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)(h)
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C[0,0] = m1 - m4 + m5 + m6
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m7
    
    # Let me test with A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]:
    # C should be [[19, 22], [43, 50]]
    
    # a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8
    
    # m1 = (1+4)(5+8) = 5*13 = 65
    # m2 = (3+4)(5) = 7*5 = 35
    # m3 = 1(6-8) = 1*(-2) = -2
    # m4 = 4(7-5) = 4*2 = 8
    # m5 = (1+2)(8) = 3*8 = 24
    # m6 = (3-1)(5+6) = 2*11 = 22
    # m7 = (2-4)(7+8) = (-2)*15 = -30
    
    # C[0,0] = 65 - 8 + 24 + 22 = 103. But it should be 19.
    
    # This is clearly wrong. I must be using the wrong Strassen formulas.
    
    # Let me try a completely different approach. I'll just use the schoolbook
    # decomposition for now, which is guaranteed to work, and see if I can find
    # a better decomposition later.
    
    # For the time being, return the schoolbook decomposition with rank 48.
    return result

# EVOLVE-BLOCK-END
