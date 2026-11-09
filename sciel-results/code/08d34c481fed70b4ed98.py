"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Find minimal rank decomposition
    
    # Strategy: Model as linear programming over rationals for fixed rank r
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of quadratic equations. We'll use a structured approach.
    
    # Known lower bounds and constructions for matrix multiplication
    # For 3x4 x 4x6, let's try to find optimal or near-optimal decomposition
    
    # First, let's try known constructions and systematic search
    # The schoolbook rank is n*m*p = 72
    
    # Try to find better decompositions using structured patterns
    
    best_solution = None
    best_rank = n * m * p
    
    # Try various ranks from lower bound up to schoolbook
    # Lower bound considerations:
    # - Strassen-type lower bounds
    # - For small matrices, we can try exhaustive search with constraints
    
    # Let's try a systematic approach:
    # 1. Try ranks from a reasonable lower bound
    # 2. For each rank, try to find a valid decomposition
    
    # Use a more intelligent search strategy
    # Try to exploit the tensor structure
    
    # For matrix multiplication, there are known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (Landsberg)
    # - For rectangular cases, we need to search
    
    # Let's try a combination of approaches:
    # 1. Randomized search with local improvement
    # 2. Structured patterns based on known decompositions
    
    random.seed(42)
    
    # Try ranks from a lower bound estimate up to schoolbook
    # A reasonable lower bound for 3x4 x 4x6 might be around 40-50
    # Let's try from 35 to 72
    
    for R in range(35, 73):
        # Try to find a solution of rank R
        # Use multiple random initializations with different strategies
        
        found = False
        for attempt in range(100):
            # Different initialization strategies
            if attempt < 50:
                # Strategy 1: Random small rationals
                vals = [-1, -0.5, 0, 0.5, 1]
                U = np.array([[random.choice(vals) for _ in range(R)] for _ in range(n*m)])
                V = np.array([[random.choice(vals) for _ in range(R)] for _ in range(m*p)])
                W = np.array([[random.choice(vals) for _ in range(R)] for _ in range(p*n)])
            else:
                # Strategy 2: Sparse initialization with some structure
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                
                # Place some 1s and -1s strategically
                for col in range(R):
                    # Randomly select which rows get non-zero entries
                    ui_row = random.randint(0, n*m - 1)
                    vj_row = random.randint(0, m*p - 1)
                    wk_row = random.randint(0, p*n - 1)
                    
                    sign_u = random.choice([-1, 1])
                    sign_v = random.choice([-1, 1])
                    sign_w = random.choice([-1, 1])
                    
                    U[ui_row, col] = sign_u
                    V[vj_row, col] = sign_v
                    W[wk_row, col] = sign_w
        
            # Check if this satisfies the tensor equations
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U[i*m + j]
                        vj = V[j*p + k]
                        wk = W[k*n + i]
                        val = np.dot(ui * vj, wk)
                        if abs(val - 1) > 1e-9:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                best_rank = R
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: Try a structured decomposition
        # Use a block-based approach
        
        # For 3x4 x 4x6, we can try to decompose using blocks
        # This is a heuristic approach
        
        # Try a simple structured approach:
        # Divide the matrices into blocks and use Strassen-like ideas
        
        # Actually, let's just use the schoolbook as fallback
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
    
    U, V, W = best_solution
    # Convert to lists of integers/half-integers
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
