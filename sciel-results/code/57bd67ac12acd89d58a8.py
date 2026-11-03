"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _validate_333(U3, V3, W3):
    """Check 3x3x3 decomposition: sum_r U3[i,j,r]*V3[j,k,r]*W3[i,k,r] = 1"""
    R = len(U3[0]) if U3 else 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(R):
                    s += U3[i*3+j][r] * V3[j*3+k][r] * W3[i*3+k][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 3, 3, 4

    # Schoolbook fallback: rank 36
    def schoolbook():
        R = n * m * p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    # Strategy: Use Winograd 3x3x3 (rank 21) + 9 schoolbook for k=3
    # Total: 30
    
    # Construct Winograd 3x3x3 decomposition using 5-point DFT
    # The key: embed 3x3 into 5x5, use DFT-5 to diagonalize
    
    def construct_winograd_333():
        """Construct rank-21 Winograd decomposition for 3x3x3 matmul."""
        # 5-point DFT matrix (real part for rational coefficients)
        # We use the fact that the 5th cyclotomic polynomial gives
        # a basis over Q(sqrt(5))
        
        # For the tensor decomposition, we need explicit rank-1 terms.
        # The Winograd 3x3x3 uses 21 multiplications.
        
        # Let me use a known explicit decomposition.
        # The Bini-Capovani-Lotti algorithm for 3x3x3 has rank 21.
        
        # Here's the explicit rank-21 decomposition:
        # Each rank-1 term is (u_r, v_r, w_r) where each is a 3x3 matrix.
        
        # I'll use the following approach: try to find a decomposition
        # using the structure of cyclic convolutions.
        
        # Actually, let me use a numerical approach with random initialization
        # and iterative refinement.
        
        R = 21
        np.random.seed(42)
        
        # Initialize U, V, W as 3x3 x R arrays
        U3 = np.zeros((9, R))
        V3 = np.zeros((9, R))
        W3 = np.zeros((9, R))
        
        # Random initialization
        U3 = np.random.randn(9, R)
        V3 = np.random.randn(9, R)
        W3 = np.random.randn(9, R)
        
        # The constraint is: for all i,j,k in {0,1,2}:
        # sum_r U3[i,j,r] * V3[j,k,r] * W3[i,k,r] = 1
        
        # This is a system of 27 equations in 3*9*21 = 567 unknowns.
        # We can use iterative methods.
        
        # Simple approach: use the fact that we can fix one matrix and solve
        # for the others, then iterate.
        
        # Let me try a different approach: use the known Winograd structure.
        
        # The Winograd 3x3x3 decomposition based on DFT-5:
        # We need the 5th roots of unity. Let's work over the reals
        # using the minimal polynomial of cos(2*pi/5).
        
        # cos(2*pi/5) = (sqrt(5)-1)/4
        # Let a = cos(2*pi/5) = (sqrt(5)-1)/4
        # Let b = sin(2*pi/5) = sqrt(10+2*sqrt(5))/4
        
        # For a real decomposition, we can use the real and imaginary parts.
        
        # Actually, let me just try to find any valid rank-21 decomposition
        # using a more systematic approach.
        
        # The key insight: 3x3 matmul can be viewed as computing
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Using the Winograd scheme with 5-point transforms:
        # We pad A, B, C to 5x5 and use DFT-5.
        
        # The DFT-5 matrix F has entries F[a,b] = exp(-2*pi*i*a*b/5)
        
        # For the tensor decomposition, each rank-1 term corresponds
        # to a frequency component.
        
        # Let me construct it explicitly using the algebraic structure.
        
        # The 5th cyclotomic polynomial is x^4 + x^3 + x^2 + x + 1
        # The roots are exp(2*pi*i*k/5) for k=1,2,3,4
        
        # For a real decomposition, we can use:
        # - The trivial component (k=0): all 1s
        # - Pairs of conjugate components (k=1,4) and (k=2,3)
        
        # This gives 1 + 2 + 2 = 5 frequency components, but each
        # 2D frequency component gives multiple rank-1 terms.
        
        # Actually, for 2D DFT-5, we have 5x5 = 25 frequency components.
        # Due to conjugate symmetry, we can reduce to 13 real components.
        # But we need exactly 21, so let me think differently.
        
        # Let me just use a known explicit decomposition from the literature.
        # The following is a rank-21 decomposition for 3x3x3:
        
        # I'll use the approach from "Fast Matrix Multiplication" by
        # Winograd, adapted for the tensor decomposition format.
        
        # For now, let me try a simpler approach: use random search
        # with constraint satisfaction.
        
        # Actually, let me just return None and use the Strassen approach
        # which I know works.
        
        return None
    
    # Let me try a different approach: use multiple Strassen blocks
    # to get a better rank than 35.
    
    # Approach: Use Strassen on multiple 2x2x2 blocks.
    # We have a 3x3x4 tensor. We can cover it with overlapping Strassen blocks.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> rank 7
    # Block 2: i in {1,2}, j in {0,1}, k in {0,1} -> rank 7 (but overlaps with Block 1)
    
    # The overlap makes this tricky. Let me try a different partition.
    
    # Alternative: Use the fact that 3 = 2+1, so we can write:
    # C = A*B where A is 3x3, B is 3x4
    # Split A into [A1; A2] where A1 is 2x3, A2 is 1x3
    # Split B into [B1 B2] where B1 is 3x3, B2 is 3x1
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # = (2x3 x 3x3) + (2x3 x 3x1) + (1x3 x 3x3) + (1x3 x 3x1)
    
    # Rank of 2x3 x 3x3: can be done with rank 2*3*3=18 schoolbook,
    # but potentially less with Strassen-like methods.
    
    # This is getting complex. Let me stick with the known rank-35 solution
    # and try to optimize it.
    
    # For now, return the rank-35 solution.
    total_rank = 35
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_U = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]],
    ]
    strassen_V = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,0],[1,-1]],
        [[0,1],[-1,0]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[0,1]],
    ]
    strassen_W = [
        [[1,1],[1,1]],
        [[0,0],[1,0]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[0,1],[0,0]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
    ]
    
    # Verify Strassen
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += strassen_U[r][i][j] * strassen_V[r][j][k] * strassen_W[r][i][k]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = strassen_U[r][i][j]
                for k in range(2):
                    W[k*n + i][col] = strassen_W[r][i][k]
            for j in range(2):
                for k in range(2):
                    V[j*p + k][col] = strassen_V[r][j][k]
            col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return best

# EVOLVE-BLOCK-END
