import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 5, 5
    
    # We need to find U (10 x r), V (25 x r), W (10 x r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # i in [0,2), j in [0,5), k in [0,5)
    
    # The schoolbook solution has rank 50.
    # Let's try to find a lower rank solution.
    
    # Strategy: Use a structured decomposition based on partitioning the 5x5 matrix.
    # Partition B (5x5) into blocks: 2+2+1
    # Let's try to use Strassen-like tricks where possible.
    
    # For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications instead of 8.
    # For a 2x2 * 2x1 or 2x1 * 2x2 or 1x2 * 2x2, we need 4 multiplications (schoolbook) or can we do better?
    # Actually, for small rectangular cases, the rank is often nm or similar.
    
    # Let's manually construct a decomposition that exploits the structure.
    # We partition the columns of A (size 2x5) into 2+2+1.
    # We partition the rows of B (size 5x5) into 2+2+1.
    
    # Let A = [A1 | A2 | A3] where A1, A2 are 2x2 and A3 is 2x1.
    # Let B = [B1; B2; B3] where B1, B2 are 2x5 and B3 is 1x5.
    
    # C = A1*B1 + A2*B2 + A3*B3
    
    # Each term A1*B1 is a 2x2 * 2x5 multiplication.
    # Let's see if we can decompose these efficiently.
    
    # Actually, let's just try to find a solution by solving the linear system
    # for a given rank r, assuming the entries are from a small set like {-1, 0, 1}.
    # This is a constraint satisfaction problem.
    
    # Alternatively, we can use a known decomposition if available.
    # For MM(2,5,5), the rank is not known to be significantly less than 50 in the literature I can recall immediately.
    # However, we can try to improve upon the schoolbook by merging terms.
    
    # Let's try a randomized search for a lower rank solution.
    # We will try ranks from 45 down to 40.
    
    def try_random_search(r, max_iter=1000):
        # Initialize U, V, W with random integer values in {-1, 0, 1}
        # This is a heuristic approach.
        
        # For each term c, we have a rank-1 tensor u_c * v_c * w_c.
        # We want the sum to be the identity tensor.
        
        # This is equivalent to finding a set of r rank-1 tensors that sum to the target tensor.
        
        # Let's try a simple approach:
        # 1. Generate r random rank-1 tensors with integer entries in {-1, 0, 1}.
        # 2. Check if they sum to the target.
        # This is unlikely to work directly, but we can use a more structured approach.
        
        pass
    
    # Let's try a more deterministic approach.
    # We know the schoolbook solution. Can we combine some terms?
    
    # The schoolbook solution has 50 terms, each corresponding to a (i,j,k) triple.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k.
    
    # Can we find two or more schoolbook terms that can be combined into a single rank-1 term?
    # No, because each term has a unique (i,j,k) pattern in the support.
    
    # However, we might be able to find a decomposition that is not a subset of the schoolbook terms.
    
    # Let's try to use the fact that n=2 is small.
    # C[0, k] = sum_j A[0, j] * B[j, k]
    # C[1, k] = sum_j A[1, j] * B[j, k]
    
    # This is 2 independent matrix-vector multiplications (if we view each column of B separately).
    # Actually, it's 2 matrix-matrix multiplications of 2x5 by 5x5.
    
    # Let's try to find a decomposition by solving the system of equations.
    # For a given rank r, we have 45r variables and 50 equations.
    # We can set up the system and try to solve it using a numerical solver.
    
    # Let's try r=45, 46, 47, 48, 49.
    
    # For each r, we can try to find a solution with entries in {-1, 0, 1} or similar.
    
    # Let's use a simple iterative method:
    # 1. Initialize U, V, W with random values.
    # 2. Compute the current tensor T' = sum_c u_c * v_c * w_c.
    # 3. Compute the error E = T - T'.
    # 4. Update U, V, W to reduce the error.
    
    # This is similar to gradient descent, but for a trilinear form.
    
    # Let's try a simpler approach: use the known Strassen decomposition for the 2x2 blocks.
    
    # Partition A into A1 (2x2), A2 (2x2), A3 (2x1)
    # Partition B into B1 (2x5), B2 (2x5), B3 (1x5)
    
    # C = A1*B1 + A2*B2 + A3*B3
    
    # For A1*B1 (2x2 * 2x5):
    # Let A1 = [[a11, a12], [a21, a22]]
    # Let B1 = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]]
    
    # C1 = A1*B1 = [[a11*b11 + a12*b21, ..., a11*b15 + a12*b25],
    #                [a21*b11 + a22*b21, ..., a21*b15 + a22*b25]]
    
    # This is a 2x2 * 2x5 multiplication.
    # The rank of this is at most 4*5=20 (schoolbook).
    # Can we do better?
    
    # Actually, for a 2x2 * 2x5, we can use the fact that it's 5 columns of 2x2 * 2x1.
    # Each 2x2 * 2x1 multiplication has rank 4 (schoolbook) or possibly less.
    
    # Let's just try to find a solution by solving the system of equations for r=45.
    
    # We'll use a random initialization and then use a simple coordinate descent.
    
    def solve_system(r):
        # Initialize U, V, W with random values in {-1, 0, 1}
        np.random.seed(42)
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # For each column c, assign a random pattern
        for c in range(r):
            # Randomly choose a few non-zero entries
            num_nonzero_U = np.random.randint(1, 4)
            idx_U = np.random.choice(n*m, num_nonzero_U, replace=False)
            U[idx_U, c] = np.random.choice([-1, 1], num_nonzero_U)
            
            num_nonzero_V = np.random.randint(1, 4)
            idx_V = np.random.choice(m*p, num_nonzero_V, replace=False)
            V[idx_V, c] = np.random.choice([-1, 1], num_nonzero_V)
            
            num_nonzero_W = np.random.randint(1, 4)
            idx_W = np.random.choice(p*n, num_nonzero_W, replace=False)
            W[idx_W, c] = np.random.choice([-1, 1], num_nonzero_W)
        
        # Check if this is a valid solution
        # Compute the tensor
        T = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    if U[i*m+j, c] != 0:
                        for k in range(p):
                            if V[j*p+k, c] != 0 and W[k*n+i, c] != 0:
                                T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check if T is the identity tensor
        target = np.ones((n, m, p))
        if np.allclose(T, target):
            return U, V, W
        
        return None
    
    # Let's try a more systematic approach.
    # We'll use the fact that the problem is small and try to find a solution
    # by solving the system of equations directly.
    
    # For a given rank r, we have 45r variables and 50 equations.
    # We can use a numerical solver to find a solution.
    
    # Let's try r=45, 46, 47, 48, 49.
    
    # We'll use a simple approach: fix some variables and solve for the rest.
    
    # Actually, let's just return the schoolbook solution for now.
    # It's guaranteed to be correct, and we can try to improve it later.
    
    U_school = [[0]*50 for _ in range(10)]
    V_school = [[0]*50 for _ in range(25)]
    W_school = [[0]*50 for _ in range(10)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    return {'U': U_school, 'V': V_school, 'W': W_school}
