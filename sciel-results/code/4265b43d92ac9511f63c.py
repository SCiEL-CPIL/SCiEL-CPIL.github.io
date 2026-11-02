"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is a tuple (U_row, V_row, W_row) where:
      U_row: indices (i, j) in A for the linear functional
      V_row: indices (j, k) in B for the linear functional  
      W_row: indices (k, i) in C for the output coefficient
    
    For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A@B:
    C[0,0] = ae + bf
    C[0,1] = af + bg
    C[1,0] = ce + df
    C[1,1] = cf + dg
    
    Strassen's 7 multiplications:
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)(h)
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    And:
    C[0,0] = m1 - m4 + m5 + m6
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m3 + m7
    """
    # For each of the 7 terms, we define:
    # U: which entries of A are summed (with coefficients)
    # V: which entries of B are summed (with coefficients)
    # W: which entry of C this contributes to (with coefficient)
    
    # Term 1: m1 = (a+d)(e+h)
    # U: a=1, d=1 (indices (0,0) and (1,1))
    # V: e=1, h=1 (indices (0,0) and (1,1))
    # W: contributes to C[0,0] with +1, C[1,1] with +1
    # But wait, in our formulation, each term r has:
    #   alpha_r = sum U[i,j,r] * A[i,j]  (scalar)
    #   beta_r = sum V[j,k,r] * B[j,k]  (scalar)
    #   C[i,k] += alpha_r * beta_r * W[k,i,r]
    
    # So for m1:
    # U[(0,0)] = 1, U[(1,1)] = 1
    # V[(0,0)] = 1, V[(1,1)] = 1
    # W[(0,0)] = 1, W[(1,1)] = 1  (i.e., W[0*n+0] and W[1*n+1] for n=2)
    
    # But this doesn't work directly because each term produces ONE scalar product
    # that gets distributed to multiple C entries.
    
    # Let me reformulate. In the standard tensor decomposition:
    # C[i,k] = sum_r alpha_r(A) * beta_r(B) * gamma_r[i,k]
    
    # where alpha_r(A) = sum_{i',j'} U[i',j',r] * A[i',j']
    #       beta_r(B) = sum_{j',k'} V[j',k',r] * B[j',k']
    #       gamma_r[i,k] = W[k*n+i, r]
    
    # For Strassen term 1: m1 = (a+d)(e+h)
    # alpha_1 = A[0,0] + A[1,1]
    # beta_1 = B[0,0] + B[1,1]
    # gamma_1[0,0] = 1, gamma_1[1,1] = 1, others = 0
    
    # But that's not right either, because m1 contributes to both C[0,0] and C[1,1].
    # In the bilinear decomposition, each term r contributes alpha_r * beta_r * gamma_r[i,k]
    # to C[i,k]. So if gamma_1[0,0]=1 and gamma_1[1,1]=1, then:
    # C[0,0] += alpha_1 * beta_1 * 1
    # C[1,1] += alpha_1 * beta_1 * 1
    # That's correct!
    
    terms = []
    
    # Term 1: m1 = (a+d)(e+h)
    # U: A[0,0]=1, A[1,1]=1
    # V: B[0,0]=1, B[1,1]=1
    # W: C[0,0]=1, C[1,1]=1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Term 2: m2 = (c+d)(e)
    # U: A[1,0]=1, A[1,1]=1
    # V: B[0,0]=1
    # W: C[1,0]=1
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # Term 3: m3 = a(f-h)
    # U: A[0,0]=1
    # V: B[0,1]=1, B[1,1]=-1
    # W: C[0,1]=1
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1}
    })
    
    # Term 4: m4 = d(g-e)
    # U: A[1,1]=1
    # V: B[1,0]=1, B[0,0]=-1
    # W: C[0,0]=-1, C[1,0]=1
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # Term 5: m5 = (a+b)(h)
    # U: A[0,0]=1, A[0,1]=1
    # V: B[1,1]=1
    # W: C[0,0]=1, C[0,1]=1
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): 1, (0,1): 1}
    })
    
    # Term 6: m6 = (c-a)(e+f)
    # U: A[1,0]=1, A[0,0]=-1
    # V: B[0,0]=1, B[0,1]=1
    # W: C[0,0]=1
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1}
    })
    
    # Term 7: m7 = (b-d)(g+h)
    # U: A[0,1]=1, A[1,1]=-1
    # V: B[1,0]=1, B[1,1]=1
    # W: C[1,1]=1
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(1,1): 1}
    })
    
    return terms

def _verify_strassen_2x2():
    """Verify that the Strassen terms correctly compute 2x2 matrix multiplication."""
    terms = _strassen_2x2_terms()
    n = m = p = 2
    
    # Test with random matrices
    for _ in range(100):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for term in terms:
            alpha = 0
            for (i, j), coeff in term['U'].items():
                alpha += coeff * A[i, j]
            
            beta = 0
            for (j, k), coeff in term['V'].items():
                beta += coeff * B[j, k]
            
            for (i, k), coeff in term['W'].items():
                C_computed[i, k] += alpha * beta * coeff
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _construct_block_decomposition(n, m, p, deadline):
    """
    Construct a decomposition using Strassen for 2x2 blocks.
    
    For n=2, m=4, p=6:
    Split A (2x4) into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    Split B (4x6) into four 2x2 blocks:
      B00 (rows 0-1, cols 0-1)
      B01 (rows 0-1, cols 2-3)
      B10 (rows 2-3, cols 0-1)
      B11 (rows 2-3, cols 2-3)
    
    C = A @ B
    C[0:2, 0:2] = A0 @ B00 + A1 @ B10
    C[0:2, 2:4] = A0 @ B01 + A1 @ B11
    C[2:4, 0:2] = ... wait, C is 2x6, so:
    
    Actually, A is 2x4 and B is 4x6, so C is 2x6.
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    Split columns of A: A[:, 0:2] and A[:, 2:4]
    Split rows of B: B[0:2, :] and B[2:4, :]
    
    C[i, k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
             = (A[:, 0:2] @ B[0:2, :])[i, k] + (A[:, 2:4] @ B[2:4, :])[i, k]
    
    So we have two 2x4 matrix multiplications, each of which can be viewed as
    a 2x2 block matrix times a 2x(2x3) block matrix.
    
    Alternatively, split both:
    A = [A0 | A1] where A0, A1 are 2x2
    B = [B0; B1] where B0, B1 are 2x6
    
    C = A0 @ B0 + A1 @ B1
    
    Now A0 @ B0 is 2x2 @ 2x6. We can split B0 into three 2x2 blocks:
    B0 = [B00 | B01 | B02]
    A0 @ B0 = [A0@B00 | A0@B01 | A0@B02]
    
    Each A0@B0x is a 2x2 @ 2x2 multiplication, which can be done with rank 7.
    
    Similarly for A1 @ B1.
    
    Total: 6 blocks * 7 = 42 rank. But we can do better.
    
    Actually, let's think differently. We have:
    C = A0 @ B0 + A1 @ B1
    
    where A0, A1 are 2x2, B0, B1 are 2x6.
    
    We can use the fact that matrix multiplication is bilinear. Let's define:
    - Use Strassen to compute A0 @ B0 and A1 @ B1 separately.
    
    But A0 @ B0 is 2x2 @ 2x6. We can split B0 into 3 blocks of 2x2:
    B0 = [B0a | B0b | B0c]
    A0 @ B0 = [A0@B0a | A0@B0b | A0@B0c]
    
    Each A0@B0x is 2x2 @ 2x2, rank 7.
    So A0 @ B0 takes 3*7 = 21 rank.
    Similarly A1 @ B1 takes 3*7 = 21 rank.
    Total: 42 rank.
    
    But wait, we can share some Strassen terms between A0@B0 and A1@B1?
    No, because they involve different matrices.
    
    Actually, let me reconsider. The total number of 2x2 @ 2x2 multiplications needed:
    - A0 @ B0a, A0 @ B0b, A0 @ B0c
    - A1 @ B1a, A1 @ B1b, A1 @ B1c
    
    That's 6 multiplications, each of rank 7, total 42.
    
    Can we do better? What if we use a different block structure?
    
    Let me try: split A into 2 blocks of 2x2, and B into 3 blocks of 4x2.
    Then C = A0 @ B0 + A1 @ B1 where B0, B1 are 4x2.
    
    A0 @ B0 is 2x2 @ 4x2. Split B0 into two 2x2 blocks:
    B0 = [B0a; B0b] (stacked vertically)
    A0 @ B0 = A0 @ [B0a; B0b] = [A0@B0a | A0@B0b]... no, that's not right.
    
    A0 @ B0 where A0 is 2x2 and B0 is 4x2:
    C[i,k] = sum_j A0[i,j] * B0[j,k] for j=0..3
    = sum_{j=0}^{1} A0[i,j] * B0[j,k] + sum_{j=2}^{3} A0[i,j] * B0[j,k]
    
    This doesn't factor nicely into 2x2 multiplications.
    
    Let me stick with the 42-rank construction and see if we can reduce it.
    
    Actually, I realize there might be a simpler approach. Let me just construct
    the 42-rank decomposition directly.
    """
    n, m, p = 2, 4, 6
    
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into A0 (cols 0-1) and A1 (cols 2-3), each 2x2
    # Split B into 6 blocks of 2x2:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
    
    # C[0:2, 0:2] = A0 @ B[0:2,0:2] + A1 @ B[2:4,0:2]
    # C[0:2, 2:4] = A0 @ B[0:2,2:4] + A1 @ B[2:4,2:4]
    # C[0:2, 4:6] = A0 @ B[0:2,4:6] + A1 @ B[2:4,4:6]
    
    # For each 2x2 @ 2x2 multiplication, use Strassen (rank 7)
    # Total: 6 * 7 = 42
    
    strassen_terms = _strassen_2x2_terms()
    
    # Build U, V, W matrices
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each of the 6 block multiplications, add 7 Strassen terms
    # Block (a_block, b_block) where a_block in {0, 1} (which 2x2 block of A)
    # and b_block in {0, 1, 2} (which 2x2 block of B)
    
    for a_block in range(2):
        for b_block in range(3):
            for t_idx, term in enumerate(strassen_terms):
                r = a_block * 21 + b_block * 7 + t_idx
                
                # U: which entries of A are used
                # A is 2x4. A0 = A[:, 0:2], A1 = A[:, 2:4]
                # For a_block=0: A entries are A[i, j] for i=0,1; j=0,1
                # For a_block=1: A entries are A[i, j] for i=0,1; j=2,3
                for (i, j_local), coeff in term['U'].items():
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global][r] += coeff
                
                # V: which entries of B are used
                # B is 4x6. For b_block=0: B[0:2, 0:2]
                # For b_block=1: B[0:2, 2:4]
                # For b_block=2: B[0:2, 4:6]
                # Wait, we also have A1 @ B[2:4, :]
                # Let me redefine:
                # a_block=0: A[:, 0:2], b_block: B[0:2, b_block*2:(b_block+1)*2]
                # a_block=1: A[:, 2:4], b_block: B[2:4, b_block*2:(b_block+1)*2]
                
                # Actually, I need to be more careful.
                # C[i, k] = sum_j A[i,j] * B[j,k]
                # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
                
                # For the first sum, j=0,1, so B[j,k] for j=0,1
                # For the second sum, j=2,3, so B[j,k] for j=2,3
                
                # So:
                # a_block=0: A[i, j] for j=0,1; B[j, k] for j=0,1
                # a_block=1: A[i, j] for j=2,3; B[j, k] for j=2,3
                
                # For b_block, it determines which columns of B:
                # b_block=0: k=0,1
                # b_block=1: k=2,3
                # b_block=2: k=4,5
                
                # So for term (a_block, b_block, t_idx):
                # A entries: i in {0,1}, j in {a_block*2, a_block*2+1}
                # B entries: j in {a_block*2, a_block*2+1}, k in {b_block*2, b_block*2+1}
                
                # Wait, that's not right. The j index in A and B must match.
                # For a_block=0: A uses j=0,1 and B uses j=0,1
                # For a_block=1: A uses j=2,3 and B uses j=2,3
                
                # So B's row index is determined by a_block, and B's column index by b_block.
                
                for (j_local, k_local), coeff in term['V'].items():
                    j_global = a_block * 2 + j_local
                    k_global = b_block * 2 + k_local
                    V[j_global * p + k_global][r] += coeff
                
                # W: which entry of C
                # C[i, k] where i in {0,1} and k in {b_block*2, b_block*2+1}
                for (i, k_local), coeff in term['W'].items():
                    k_global = b_block * 2 + k_local
                    W[k_global * n + i][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, max_samples=100):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i * m + j][r] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j * p + k][r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += alpha[i] * beta[k] * W[k * n + i][r]
        
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try the block Strassen decomposition (rank 42)
    result = _construct_block_decomposition(n, m, p, deadline)
    
    # Verify it
    if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, max_samples=50):
        return result
    
    # If verification fails, fall back to schoolbook (rank 48)
    result = _schoolbook_decomposition(n, m, p)
    return result

# EVOLVE-BLOCK-END
