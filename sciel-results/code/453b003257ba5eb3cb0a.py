import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # 2x3x6 = two independent 2x3x3 blocks
    # Need rank-8 decomposition for 2x3x3, use it twice for total rank 16
    
    # For 2x3x3: T(i,j,k) = 1 if indices consistent
    # U: (n*m, rank) = (6, 8)
    # V: (m*p, rank) = (9, 8)  
    # W: (p*n, rank) = (6, 8)
    # Condition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's find a rank-8 decomposition for 2x3x3
    # Using known results and algebraic identities
    
    # The 6 outputs are:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Rank-8 decomposition for 2x3x3:
    # We can use the following 8 terms (coefficients in {0, 1, -1, 1/2}):
    
    # Let me try to construct this using a systematic approach
    # with small coefficients.
    
    # Approach: use the fact that we can share multiplications
    # between the two rows of A.
    
    # Define 8 products:
    # P0 = A[0,0] * B[0,0]
    # P1 = A[0,1] * B[1,0]  
    # P2 = A[0,2] * B[2,0]
    # P3 = A[1,0] * B[0,1]
    # P4 = A[1,1] * B[1,1]
    # P5 = A[1,2] * B[2,1]
    # P6 = (A[0,0]+A[1,0]) * (B[0,2]+B[0,0])  -- combined term
    # P7 = (A[0,1]+A[1,2]) * (B[1,2]+B[2,0])  -- combined term
    
    # This is getting complex. Let me use a numerical search instead.
    
    if time.time() < deadline - 5.0:
        try:
            # Try to find rank-8 decomposition for 2x3x3 using numpy
            # We need U (6x8), V (9x8), W (6x8) such that
            # sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1 for all i,j,k
            # and 0 for inconsistent indices.
            
            # Let's use a parameterized ansatz with small integers
            # and solve the resulting system.
            
            # Actually, let me try a known decomposition from literature.
            # One rank-8 decomposition for 2x3x3:
            
            # Term structure (U_col, V_col, W_col):
            # Each term is a rank-1 tensor in the 6x9x6 space.
            
            # Let me try to use the following approach:
            # Split into 4 terms for "diagonal" and 4 for "off-diagonal"
            
            # Known rank-8 for 2x3x3 (from algebraic complexity theory):
            # Using the identity that allows sharing between rows
            
            # Let me just do a direct search with small coefficients
            max_coeff = 2
            coeffs = list(range(-max_coeff, max_coeff + 1))
            
            # This search space is too large for brute force.
            # Let me use a smarter approach: set up linear equations.
            
            # For rank 8, we have 8 * (6+9+6) = 168 unknowns.
            # We have constraints: 2*3*3 = 18 equations (for consistent indices)
            # Plus additional constraints for inconsistent indices.
            
            # Actually, the tensor T has dimension 6*9*6 = 324 entries.
            # But most are zero. The nonzero entries are at positions
            # (i*3+j, j*3+k, k*2+i) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
            # That's 18 nonzero entries.
            
            # The decomposition condition is:
            # For all (a,b,c) in {0..5}x{0..8}x{0..5}:
            #   sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # This is a nonlinear system (product of three variables).
            # Let me try a different approach: fix U and V, solve for W,
            # or use iterative methods.
            
            # Actually, let me try a specific known decomposition.
            # After research, I recall that for 2x3x3, one rank-8 decomposition is:
            
            # Let me try to construct it using the following insight:
            # We can write the multiplication using 8 bilinear forms.
            
            # Here's a concrete attempt:
            # The key is to use sums/differences to combine terms.
            
            # Let's define the 8 multiplications as:
            # m0 = A00 * B00
            # m1 = A01 * B10
            # m2 = A02 * B20
            # m3 = A10 * B01
            # m4 = A11 * B11
            # m5 = A12 * B21
            # m6 = (A00+A10) * (B02+B00)
            # m7 = (A01+A12) * (B12+B20)
            
            # Then:
            # C00 = m0 + m1 + m2
            # C10 = ? (need A10*B00, A11*B10, A12*B20)
            # C01 = ? (need A00*B01, A01*B11, A02*B21)
            # C11 = m4 + ? 
            # C02 = ? (need A00*B02, A01*B12, A02*B22)
            # C12 = ? (need A10*B02, A11*B12, A12*B22)
            
            # This doesn't quite work. Let me try a different combination.
            
            # Alternative: use 3x3 matrix multiplication rank 7 (Strassen)
            # But we need 2x3x3, not 3x3x3.
            
            # Let me try yet another approach: use the fact that
            # 2x3x3 can be embedded in 3x3x3 and use Strassen.
            
            # Actually, let me just try to find ANY valid decomposition
            # with rank < 36 using a computational search.
            
            # For rank 16 (two copies of rank 8):
            # Let me try to solve the 2x3x3 case using least squares
            # with a parameterized family.
            
            # Parameterize U, V, W with small integer entries
            # and check if the tensor condition is satisfied.
            
            # This is a constraint satisfaction problem.
            # Let me try a greedy approach: build up the decomposition
            # term by term, minimizing the residual.
            
            def residual(U, V, W, n_, m_, p_):
                """Compute the residual tensor."""
                T = np.zeros((n_*m_, m_*p_, p_*n_))
                for i in range(n_):
                    for j in range(m_):
                        for k in range(p_):
                            T[i*m_+j, j*p_+k, k*n_+i] = 1.0
                # Compute reconstructed tensor
                R = np.zeros_like(T)
                for r in range(U.shape[1]):
                    R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return T - R
            
            def check_valid(U, V, W, n_, m_, p_, tol=1e-9):
                """Check if the decomposition is valid."""
                T = np.zeros((n_*m_, m_*p_, p_*n_))
                for i in range(n_):
                    for j in range(m_):
                        for k in range(p_):
                            T[i*m_+j, j*p_+k, k*n_+i] = 1.0
                R = np.zeros_like(T)
                for r in range(U.shape[1]):
                    R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.allclose(R, T, atol=tol)
            
            # Try rank 16 with block structure
            # Block 0: k in {0,1,2}, Block 1: k in {3,4,5}
            
            # For each block, we need a 2x3x3 decomposition of rank 8.
            # Let me try to find one using a numerical method.
            
            # Ansatz: use U, V, W with entries in {0, 1, -1, 1/2, -1/2}
            # and search over a restricted set of patterns.
            
            # This is still a huge search space. Let me try a different tactic:
            # Use the known result that R(2,3,3) = 8 and try to find
            # a specific decomposition from the literature.
            
            # After careful consideration, here's a rank-8 decomposition for 2x3x3:
            # Using the following 8 terms (U_col, V_col, W_col):
            
            # I'll use a structured approach based on the following:
            # The multiplication can be done with 8 multiplications by
            # using the following bilinear forms:
            
            # f0 = A00*B00
            # f1 = A01*B10  
            # f2 = A02*B20
            # f3 = A10*B01
            # f4 = A11*B11
            # f5 = A12*B21
            # f6 = (A00+A11)*(B02+B10)
            # f7 = (A02+A10)*(B20+B02)
            
            # Let me verify if this works...
            # C00 = A00*B00 + A01*B10 + A02*B20 = f0 + f1 + f2
            # C10 = A10*B00 + A11*B10 + A12*B20
            # C01 = A00*B01 + A01*B11 + A02*B21
            # C11 = A10*B01 + A11*B11 + A12*B21 = f3 + f4 + f5
            # C02 = A00*B02 + A01*B12 + A02*B22
            # C12 = A10*B02 + A11*B12 + A12*B22
            
            # f6 = A00*B02 + A00*B10 + A11*B02 + A11*B10
            # f7 = A02*B20 + A02*B02 + A10*B20 + A10*B02
            
            # This doesn't directly give us the needed terms.
            
            # Let me try a completely different approach: use numpy to solve
            # for a rank-8 decomposition using optimization.
            
            # Actually, let me just return a known valid decomposition.
            # The schoolbook for 2x3x6 has rank 36.
            # A better approach: use Strassen's algorithm for 2x2 blocks.
            
            # Split the 3x6 matrix into 2x2 and 2x2 and 1x2 blocks?
            # Or split the 2x3 into 1x3 and 1x3?
            
            # Let me try: split A into A1 (1x3) and A2 (1x3)
            # Split B into B1 (3x3) and B2 (3x3)
            # Then C1 = A1*B1 (1x3), C2 = A2*B2 (1x3)
            # Each 1x3x3 multiplication has rank 3 (schoolbook).
            # Total rank: 3 + 3 = 6? No, that's not right.
            # 1x3x3: C[k] = sum_j A[j]*B[j,k], rank = 3 (one for each k)
            # Wait, 1x3x3 has n=1, m=3, p=3, schoolbook rank = 1*3*3 = 9.
            # But actually, it's just a vector-matrix product, which needs 3*3 = 9 mults.
            # Hmm, that's not helpful.
            
            # Let me try: split into 2x2x3 and 2x1x3
            # 2x2x3: schoolbook rank = 12, but R(2,2,3) might be less.
            # Actually R(2,2,2) = 7 (Strassen), so R(2,2,3) <= 7*2 + something.
            
            # This is getting complicated. Let me just try to find a valid
            # decomposition with rank < 36 using a numerical search.
            
            # For rank 16, let me try to solve the system using least squares.
            
            # Parameterize U, V, W with random initial values and use
            # gradient descent to minimize the residual.
            
            np.random.seed(42)
            rank_try = 16
            U_init = np.random.randn(n*m, rank_try) * 0.5
            V_init = np.random.randn(m*p, rank_try) * 0.5
            W_init = np.random.randn(p*n, rank_try) * 0.5
            
            # Use scipy to minimize
            from scipy.optimize import minimize
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(params):
                U = params[:n*m*rank_try].reshape(n*m, rank_try)
                V = params[n*m*rank_try:n*m*rank_try+m*p*rank_try].reshape(m*p, rank_try)
                W = params[n*m*rank_try+m*p*rank_try:].reshape(p*n, rank_try)
                return U, V, W
            
            def objective(params):
                U, V, W = unflatten_params(params)
                T = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i*m+j, j*p+k, k*n+i] = 1.0
                R = np.zeros_like(T)
                for r in range(rank_try):
                    R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.sum((R - T)**2)
            
            # This optimization is expensive and may not converge to an exact solution.
            # Let me try a different approach: use the known rank-8 for 2x3x3.
            
            # After more thought, here's a rank-8 decomposition for 2x3x3:
            # Using the following terms (each term is (u, v, w)):
            
            # I found a decomposition online/in memory:
            # For 2x3x3, one rank-8 decomposition is:
            
            U_233 = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],  # A00
                [0, 1, 0, 0, 0, 0, 0, 0],  # A01
                [0, 0, 1, 0, 0, 0, 0, 0],  # A02
                [0, 0, 0, 1, 0, 0, 0, 0],  # A10
                [0, 0, 0, 0, 1, 0, 0, 0],  # A11
                [0, 0, 0, 0, 0, 1, 0, 0],  # A12
            ])
            
            V_233 = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],  # B00
                [0, 1, 0, 0, 0, 0, 0, 0],  # B01
                [0, 0, 1, 0, 0, 0, 0, 0],  # B02
                [0, 0, 0, 1, 0, 0, 0, 0],  # B10
                [0, 0, 0, 0, 1, 0, 0, 0],  # B11
                [0, 0, 0, 0, 0, 1, 0, 0],  # B12
                [0, 0, 0, 0, 0, 0, 1, 0],  # B20
                [0, 0, 0, 0, 0, 0, 0, 1],  # B21
                [0, 0, 0, 0, 0, 0, 0, 0],  # B22
            ])
            
            W_233 = np.array([
                [1, 0, 0, 0, 0, 0],  # C00 (k=0,i=0)
                [0, 0, 0, 1, 0, 0],  # C01 (k=1,i=0)
                [0, 0, 0, 0, 0, 0],  # C02 (k=2,i=0)
                [0, 0, 0, 0, 1, 0],  # C10 (k=0,i=1)
                [0, 0, 0, 0, 0, 0],  # C11 (k=1,i=1)
                [0, 0, 0, 0, 0, 1],  # C12 (k=2,i=1)
            ])
            
            # This is just schoolbook with rank 6, but it's incomplete.
            # Let me just return the schoolbook for now and try to improve.
            
            pass
        except Exception as e:
            pass
    
    # Return schoolbook rank 36 as fallback
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
