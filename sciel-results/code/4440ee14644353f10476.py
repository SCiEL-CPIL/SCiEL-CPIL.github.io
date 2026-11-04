"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (15, rank), (10, rank).
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k]
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # T[i, j, k] = 1 for all valid i,j,k (it's the identity-like tensor for the multiplication)
    # More precisely, the tensor is: T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # We want to decompose this tensor as sum of rank-1 tensors:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    
    # Wait, let me re-read the contract. The output factor follows W[k*n+i, r].
    # So the indices are:
    # U has shape (n*m, rank) - indexed by (i*m + j)
    # V has shape (m*p, rank) - indexed by (j*p + k)  
    # W has shape (p*n, rank) - indexed by (k*n + i)
    
    # The tensor equation is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    
    # This is the standard matrix multiplication tensor.
    
    # Start with schoolbook rank = n*m*p = 30
    # Try to find lower rank decompositions.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 2x3x3: rank ? 
    # - 2x3x5: rank ?
    
    # Let's try a structured approach. We can exploit the rectangular structure.
    
    # One approach: use the fact that we can decompose the 3x5 part.
    # A 2x3 matrix times a 3x5 matrix.
    
    # Let's try rank 28 first (as mentioned in the knowledge).
    
    # Actually, let me think about this more carefully.
    # The tensor T is 2x3x5.
    
    # We can try to find a decomposition using linear algebra.
    # The condition for rank r is that there exist U, V, W such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a system of equations. Let's try to solve it.
    
    # For a given rank r, we have:
    # - U: 6 x r
    # - V: 15 x r
    # - W: 10 x r
    # Total variables: 6r + 15r + 10r = 31r
    
    # Constraints: 2*3*5 = 30 equations (one for each (i,j,k))
    
    # So for r >= 1, we have more variables than constraints, but the system is nonlinear.
    
    # Let's try a greedy/approximate approach and then verify.
    
    # Actually, let's use a more systematic approach.
    # We'll try to find a decomposition of rank r by solving the nonlinear system.
    
    # Start with schoolbook and try to reduce rank.
    
    # Schoolbook: rank 30
    # Each term corresponds to one (i,j,k) triple.
    
    # Let's try to find a lower rank decomposition using numerical methods.
    
    # For a fixed rank r, we can set up the system and try to solve it.
    # The equations are:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a 3-linear system. We can try to solve it using iterative methods.
    
    # Let's try rank 28 first.
    
    # Actually, let me try a different approach. Let's use the structure of the problem.
    
    # The 2x3x5 tensor can be seen as:
    # For each j (0, 1, 2), we have a 2x5 sub-tensor (from A[:,j] * B[j,:])
    
    # Each 2x5 outer product has rank 10 (schoolbook) or potentially less.
    # But these are not independent because they share the same j.
    
    # Let's try to find a decomposition by solving the system numerically.
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a decomposition of rank r
    def find_decomposition(r, max_iter=1000, seed=42):
        """Try to find a rank-r decomposition."""
        n, m, p = DIMENSIONS
        rng = random.Random(seed)
        
        # Initialize with random values
        U = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
        V = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
        W = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
        
        # Try to solve using iterative refinement
        # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # We can use a simple iterative method:
        # 1. Fix V and W, solve for U (linear in each U entry)
        # 2. Fix U and W, solve for V
        # 3. Fix U and V, solve for W
        # Repeat.
        
        # Actually, this is tricky because the equations are 3-linear.
        # Let's try a different approach: use the fact that we can scale.
        
        # Alternative: use a tensor decomposition algorithm.
        # For now, let's try a simple approach: start with schoolbook and try to merge terms.
        
        # Schoolbook has 30 terms. Let's try to see if we can merge some.
        
        # Actually, let's just return the schoolbook for now and try to improve.
        return None
    
    # Let's try a more direct approach.
    # We'll set up the system and try to solve it.
    
    # For rank r, we have 31r variables and 30 equations.
    # The system is underdetermined for r >= 1, but it's nonlinear.
    
    # Let's try to find a decomposition by solving a sequence of linear systems.
    
    # Idea: fix two of the matrices and solve for the third.
    
    # For a fixed rank r, let's try:
    # 1. Initialize V and W randomly.
    # 2. For each (i,j,k), the equation is: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
    #    This is linear in U[i*m+j,c].
    # 3. Solve for U.
    # 4. Similarly solve for V and W.
    # 5. Iterate.
    
    # Let's implement this.
    
    def solve_decomposition(r, max_iter=500, tol=1e-12, seed=42):
        """Solve for a rank-r decomposition using iterative linear solves."""
        n, m, p = DIMENSIONS
        rng = random.Random(seed)
        
        # Initialize
        U = [[rng.uniform(0.1, 1.0) for _ in range(r)] for _ in range(n*m)]
        V = [[rng.uniform(0.1, 1.0) for _ in range(r)] for _ in range(m*p)]
        W = [[rng.uniform(0.1, 1.0) for _ in range(r)] for _ in range(p*n)]
        
        for it in range(max_iter):
            # Solve for U: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by (i,j): for fixed (i,j), the equations over k are:
            # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for k=0..p-1
            # This is a system of p equations in r unknowns (U[i*m+j,0..r-1]).
            
            new_U = [[0.0]*r for _ in range(n*m)]
            for i in range(n):
                for j in range(m):
                    # Build the system: A * x = b
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # b[k] = 1
                    A = [[V[j*p+k][c] * W[k*n+i][c] for c in range(r)] for k in range(p)]
                    b = [1.0] * p
                    # Solve A * x = b using least squares (if p < r) or direct solve
                    # Use numpy
                    import numpy as np
                    A_np = np.array(A)
                    b_np = np.array(b)
                    if p <= r:
                        # Overdetermined or square, use lstsq
                        x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                    else:
                        # Underdetermined, use lstsq (minimum norm)
                        x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                    new_U[i*m+j] = list(x)
            
            U = new_U
            
            # Solve for V: for each (j,k), sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for i=0..n-1
            new_V = [[0.0]*r for _ in range(m*p)]
            for j in range(m):
                for k in range(p):
                    A = [[U[i*m+j][c] * W[k*n+i][c] for c in range(r)] for i in range(n)]
                    b = [1.0] * n
                    A_np = np.array(A)
                    b_np = np.array(b)
                    x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                    new_V[j*p+k] = list(x)
            
            V = new_V
            
            # Solve for W: for each (k,i), sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for j=0..m-1
            new_W = [[0.0]*r for _ in range(p*n)]
            for k in range(p):
                for i in range(n):
                    A = [[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)]
                    b = [1.0] * m
                    A_np = np.array(A)
                    b_np = np.array(b)
                    x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                    new_W[k*n+i] = list(x)
            
            W = new_W
            
            # Check convergence
            if it % 50 == 0 or it == max_iter - 1:
                # Compute residual
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                            max_res = max(max_res, abs(val - 1.0))
                if max_res < tol:
                    break
        
        # Final verification
        if verify(U, V, W, tol=1e-10):
            return U, V, W
        return None
    
    # Try ranks from low to high
    # Known: schoolbook is 30. Try to find lower.
    # Let's try 28, 27, 26, ...
    
    # But this might be slow. Let's try a few ranks.
    
    # Actually, let's first return the schoolbook to ensure correctness, then try to improve.
    
    # Schoolbook decomposition
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
    
    # Try to find a lower rank decomposition
    # Try ranks 28, 27, 26, ...
    for r in range(28, 15, -1):
        result = solve_decomposition(r, max_iter=200, tol=1e-12, seed=42)
        if result is not None:
            U, V, W = result
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
