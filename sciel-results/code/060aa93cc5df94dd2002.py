"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    # Build the multiplication tensor T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is of shape (n*p, m) essentially, but let's think of it as T[i,j,k]
    # where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 (it's the structure tensor)
    
    # The decomposition: C[i,k] = sum_r U[i, r] * V[r, k] where we flatten appropriately
    # Actually, let's use the standard form:
    # C[i,k] = sum_r A[i, j_r] * B[j_r, k] for each r, but with combinations
    
    # Let's use the approach: we want to find matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: the tensor equation holds.
    
    # The tensor equation: sum_r U[i*m + j, r] * V[j*p + k, r] = W[k*n + i, r] ... no that's not right.
    
    # Let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition should satisfy: for the matrix multiplication tensor,
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no, that's for 3-way.
    
    # Actually for matrix multiplication, the tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T_{i,j,k} where the multiplication tensor has T_{i,j,k} = 1
    # and the decomposition is: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)}
    # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # I think the convention is:
    # The product C[i,k] is computed as sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... no.
    
    # Let me think differently. The standard tensor decomposition for matrix multiplication:
    # We want to express the bilinear map (A,B) -> C where C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-1 term in the tensor decomposition corresponds to:
    # C[i,k] += alpha * (sum_j a_{i,j} * A[i,j]) * (sum_k b_{j,k} * B[j,k])
    # But this doesn't quite work because the indices are coupled.
    
    # The correct tensor decomposition:
    # The multiplication tensor T has components T[i,j,k] = 1 for all i,j,k
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r]
    # where U is (n,m,R) and V is (m,p,R)
    
    # But the interface uses flattened indices:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]  
    # W: (p*n, rank) -> W[k*n+i, r]
    
    # I believe the convention is that the tensor is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But that's a 3-way tensor product, which isn't standard for matrix multiplication.
    
    # Actually, I think the W matrix is used differently. Let me reconsider.
    
    # Perhaps the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    # No, that's not dimensionally consistent.
    
    # Let me just use the schoolbook and try to improve it by finding linear dependencies.
    
    # Schoolbook: for each (i,j,k), we have a term:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for the matching (i,j,k)
    
    # Wait, I think the tensor equation is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
    
    # No, that can't be right because then the rank would be at least n*m*p.
    
    # Let me look at this more carefully. For matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The tensor T is defined such that C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k']
    # So T[i,j,k] = 1 if the indices match appropriately, specifically T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    
    # A rank-1 decomposition of the tensor: T[i,j,k] = U[i,j] * V[j,k] (for a single term)
    # But we need: sum_r U_r[i,j] * V_r[j,k] = 1 for all i,j,k
    # This is impossible with rank 1 unless n=m=p=1.
    
    # I think the correct interpretation is:
    # The tensor T[i,j,k] = 1 (for all i,j,k in the multiplication)
    # And we decompose: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # where W is not part of the tensor equation but is used for something else?
    
    # Given the confusion, let me just keep the schoolbook which is guaranteed correct,
    # and try to find improvements by random search.
    
    # Actually, let me try a different approach: use known results.
    # For 4x4x4 matrix multiplication, the rank is 23 (Laser method / Strassen-like).
    # For 4x4x7, we can pad or use a direct construction.
    
    # Let me try to use the fact that we can decompose 4x4x4 with rank 23,
    # then extend to 4x4x7 by adding 3 more columns.
    
    # But I don't have a concrete rank-23 decomposition for 4x4x4 readily available.
    
    # Let me use a practical random search approach:
    # 1. Start with schoolbook (rank 112)
    # 2. Try to find a lower rank by random linear combinations
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The multiplication tensor T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook solution
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # This is a 3-way tensor equation.
    
    # Let's try random linear combinations to reduce rank.
    # Idea: if we can find coefficients c_r such that sum_r c_r * (U_r, V_r, W_r) = 0
    # in the tensor sense, we can eliminate that term.
    
    # Simpler: try to find a subset of terms that can be combined.
    
    # Given time constraints, let's just return the schoolbook if we can't find better.
    # But let's try a quick random search.
    
    random.seed(42)
    num_attempts = 200
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Try to reduce rank by 1: find a linear combination that cancels
        # Pick a random target rank
        target_rank = best_rank - 1
        if target_rank < 1:
            break
        
        # Random projection approach
        # This is heuristic and may not work, but let's try
        
        # Generate random matrices and check
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Check if sum_r U_rand[i,j,r] * V_rand[j,k,r] * W_rand[k,i,r] = 1 for all i,j,k
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                    if abs(s - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_U = [[U_rand[i, r] for r in range(target_rank)] for i in range(n*m)]
            best_V = [[V_rand[i, r] for r in range(target_rank)] for i in range(m*p)]
            best_W = [[W_rand[i, r] for r in range(target_rank)] for i in range(p*n)]
    
    # Convert back to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(len(arr[0]))] for i in range(len(arr))]
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
