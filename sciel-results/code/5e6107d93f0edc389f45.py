"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12

    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Strategy: Use the known rank-12 decomposition for 3x4x4 matrix multiplication
    # This is based on the fact that the tensor can be decomposed using
    # the structure of the multiplication map.
    
    # Let's try to find a rank-12 decomposition using a systematic approach.
    # 
    # Key insight: For matrix multiplication C = A*B where A is n×m and B is m×p,
    # the tensor has a special structure. We can use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A rank-1 term in the decomposition corresponds to a "bilinear form" that
    # contributes to all entries simultaneously.
    
    # Let's construct a decomposition with rank = n * p = 12 (if possible)
    # or use a known low-rank decomposition.
    
    # Approach: Use the "outer product" structure
    # For each (i,k) pair, we want contributions that sum to 1 when combined with A[i,j]*B[j,k]
    
    # Actually, let's use a different approach:
    # The tensor T can be viewed as a bilinear map. We can use the fact that
    # the space of bilinear maps from R^{n*m} × R^{m*p} to R^{p*n} has dimension n*m*p = 48.
    # The rank of the tensor is the minimum number of rank-1 bilinear maps that sum to T.
    
    # Let's try a constructive approach:
    # We'll use rank = 12 and see if we can find a valid decomposition.
    
    # For rank-12, we have 12 slices. Each slice (a_c, b_c, c_c) where:
    # a_c in R^{12}, b_c in R^{16}, c_c in R^{12}
    
    # The constraint is: for all i,j,k:
    # sum_c a_c[i*m+j] * b_c[j*p+k] * c_c[k*n+i] = 1
    
    # Let's try a specific structure:
    # Set a_c[i*m+j] = alpha_{c,i} for all j (i.e., a_c is constant across j for fixed i)
    # Set b_c[j*p+k] = beta_{c,k} for all j (i.e., b_c is constant across j for fixed k)
    # Then: sum_c alpha_{c,i} * beta_{c,k} * c_c[k*n+i] = 1 for all i,k
    
    # This gives us: for each (i,k), sum_c alpha_{c,i} * beta_{c,k} * c_c[k*n+i] = 1
    # If we set c_c[k*n+i] = 1 for all c, i, k, then we need:
    # sum_c alpha_{c,i} * beta_{c,k} = 1 for all i,k
    # This is a matrix equation: Alpha * Beta^T = J (all-ones matrix)
    # where Alpha is n×rank and Beta is p×rank.
    # The rank of J is 1, so we need rank >= 1. But we also need the full constraint
    # to hold for all j, not just in aggregate.
    
    # Let's try a more general approach: use random initialization and solve.
    
    # Actually, let's use a known result: the tensor rank of 3x4x4 matrix multiplication
    # is at most 12 (since n*p = 12). Let's verify if rank 12 works.
    
    # For rank 12, let's try:
    # c_c[k*n+i] = 1 for all c, i, k (i.e., c_c is all-ones)
    # a_c[i*m+j] = u_{c,i} * v_{c,j} (outer product structure)
    # b_c[j*p+k] = w_{c,j} * x_{c,k} (outer product structure)
    
    # Then the constraint becomes:
    # sum_c u_{c,i} * v_{c,j} * w_{c,j} * x_{c,k} = 1 for all i,j,k
    # sum_c u_{c,i} * x_{c,k} * (v_{c,j} * w_{c,j}) = 1 for all i,j,k
    
    # For this to hold for all j, we need v_{c,j} * w_{c,j} = 1 for all c,j (or some constant)
    # Let's set v_{c,j} = 1/sqrt(m) and w_{c,j} = sqrt(m) so that v*w = 1.
    # Then: sum_c u_{c,i} * x_{c,k} = 1 for all i,k
    # This is again Alpha * X^T = J, which has rank 1 solution.
    
    # But this doesn't capture the full structure. Let me try a different approach.
    
    # Use the fact that we can decompose using the basis of the space.
    # The tensor T has a Kruskal-Cantor expansion. Let's find a low-rank approximation.
    
    # Let's try rank = 16 (m*p) or rank = 12 (n*p) and solve the nonlinear system.
    
    # For now, let's use a fallback to the schoolbook decomposition with rank = n*m*p = 48,
    # but try to reduce it using a greedy approach.
    
    # Actually, let's just use the schoolbook decomposition as a safe fallback,
    # and try to find a lower rank using random search.
    
    # Schoolbook: rank = n*m*p = 48
    rank = n * m * p
    U = [[Fraction(0, 1) for _ in range(rank)] for _ in range(dim_U)]
    V = [[Fraction(0, 1) for _ in range(rank)] for _ in range(dim_V)]
    W = [[Fraction(0, 1) for _ in range(rank)] for _ in range(dim_W)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = Fraction(1, 1)
                V[j*p+k][col] = Fraction(1, 1)
                W[k*n+i][col] = Fraction(1, 1)
                col += 1
    
    # Now try to find a lower rank decomposition using random search
    # We'll try ranks from 12 to 47 and see if we can find a valid decomposition.
    
    def check_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0, 1)
                    for c in range(len(U_mat[0])):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if total != Fraction(1, 1):
                        return False
        return True
    
    # Try to find a lower rank decomposition
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try rank 12 first (n*p)
    for target_rank in [12, 13, 14, 15, 16]:
        found = False
        for trial in range(100):
            np.random.seed(42 + target_rank * 1000 + trial)
            
            # Initialize with random values
            U_rand = np.random.randn(dim_U, target_rank)
            V_rand = np.random.randn(dim_V, target_rank)
            W_rand = np.random.randn(dim_W, target_rank)
            
            # Normalize
            for c in range(target_rank):
                norm_u = np.linalg.norm(U_rand[:, c])
                norm_v = np.linalg.norm(V_rand[:, c])
                norm_w = np.linalg.norm(W_rand[:, c])
                if norm_u > 1e-10:
                    U_rand[:, c] /= norm_u
                if norm_v > 1e-10:
                    V_rand[:, c] /= norm_v
                if norm_w > 1e-10:
                    W_rand[:, c] /= norm_w
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(target_rank):
                            total += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        if abs(total - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to fractions
                U_frac = [[Fraction(0, 1) for _ in range(target_rank)] for _ in range(dim_U)]
                V_frac = [[Fraction(0, 1) for _ in range(target_rank)] for _ in range(dim_V)]
                W_frac = [[Fraction(0, 1) for _ in range(target_rank)] for _ in range(dim_W)]
                
                for idx in range(dim_U):
                    for c in range(target_rank):
                        U_frac[idx][c] = Fraction(U_rand[idx, c]).limit_denominator(10**6)
                for idx in range(dim_V):
                    for c in range(target_rank):
                        V_frac[idx][c] = Fraction(V_rand[idx, c]).limit_denominator(10**6)
                for idx in range(dim_W):
                    for c in range(target_rank):
                        W_frac[idx][c] = Fraction(W_rand[idx, c]).limit_denominator(10**6)
                
                # Verify with fractions
                if check_decomposition(U_frac, V_frac, W_frac, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_frac, V_frac, W_frac
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
