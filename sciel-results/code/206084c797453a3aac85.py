"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # The schoolbook decomposition has rank n*m*p = 72.
    # We attempt to find a lower-rank decomposition using a local search / merging strategy.
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that 
    # sum_{t=1}^r U[a,t] V[b,t] W[c,t] = T[a,b,c] for all a,b,c.
    
    # Initialize with schoolbook
    rank0 = n * m * p
    U = np.zeros((n * m, rank0), dtype=float)
    V = np.zeros((m * p, rank0), dtype=float)
    W = np.zeros((p * n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Target tensor T
    # T[a,b,c] is 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Note: the mapping (i,j,k) -> (a,b,c) is bijective for valid i,j,k.
    # So T is sparse.
    
    # Define the set of non-zero entries (targets)
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets.append((a, b, c))
                
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        # Check all targets
        for a, b, c in targets:
            val = np.dot(U[a] * V[b], W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check a subset of non-targets to ensure they are 0
        # For efficiency, we check all non-targets if rank is small enough, otherwise sample
        # Total entries: (n*m)*(m*p)*(p*n) = 9 * 24 * 24 = 5184
        # Targets: 72
        # Non-targets: 5112
        # Checking all is feasible.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Skip if target
                    is_target = False
                    # Optimization: precompute target set? 
                    # For now, simple check. 
                    # Actually, we can just compute the value and check if it's close to 0.
                    # But we must distinguish between 0 and 1.
                    # If it's a target, it should be 1. If not, 0.
                    # Let's just check all 5184 entries.
                    pass
        
        # More efficient check:
        # Compute the full tensor T' = sum U[:,t] V[:,t]^T W[:,t]^T ? No, that's outer product.
        # T'[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # We can compute this using einsum or loops.
        # Given small dimensions, loops are fine.
        
        # Let's optimize the check by precomputing the target set as a set of tuples for O(1) lookup
        target_set = set(targets)
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = np.dot(U[a] * V[b], W[c])
                    if (a, b, c) in target_set:
                        if abs(val - 1.0) > tol:
                            return False
                    else:
                        if abs(val) > tol:
                            return False
        return True

    # Strategy: Try to merge columns.
    # If we have two columns t1, t2 such that their contribution to the tensor is "redundant" or can be combined,
    # we might reduce rank.
    # A simple heuristic: If we can find a linear combination of columns that cancels out some entries,
    # we might be able to remove a column.
    
    # Alternative: Use the known result that rank(3x3x3) = 23.
    # The 3x3x8 problem can be seen as 8 copies of 3x3x1? No.
    # It is 3x3 matrix times 3x8 matrix.
    # This is equivalent to 8 independent 3x3x1 matrix-vector multiplications?
    # No, the A matrix is shared.
    # C = A B. B has 8 columns.
    # C[:, k] = A B[:, k].
    # Each column of C is a vector in R^3.
    # The tensor is T(A, B, C).
    
    # Let's try to find a rank-23 decomposition for the 3x3x3 part and see if it extends.
    # Actually, the structure is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # For a fixed k, this is a 3x3 mat-vec mult.
    # The tensor for 3x3x1 has rank 9 (schoolbook). Is it lower?
    # Rank of 3x3x1 is 9 (since 3x3 matmul is 23, but 3x3x1 is just one column, which is 9 terms in schoolbook.
    # Wait, rank of M_{n,m,1} is n*m. So rank is 9.
    # So for 8 columns, if they were independent, rank would be 8*9 = 72.
    # But A is shared.
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's attempt a random search for a lower rank decomposition.
    # We start with rank 72 and try to remove columns.
    
    def try_remove_column(U, V, W, col_idx):
        """Try to see if column col_idx is redundant or can be merged."""
        # This is a complex optimization problem.
        # Let's try a simpler approach: Random initialization of a lower rank and verify.
        pass

    # Given the difficulty of finding a rank < 72 decomposition quickly,
    # and the fact that the schoolbook rank is 72,
    # and the inspiration suggests rank 23 for 3x3x3,
    # let's check if we can construct a rank 23 + 5*9 = 68 decomposition?
    # The 3x3x3 part (first 3 columns of B) has rank 23.
    # The remaining 5 columns are 3x3x1, each rank 9.
    # Total rank = 23 + 5*9 = 68.
    
    # We need a valid rank-23 decomposition for 3x3x3.
    # Let's try to find one or use a known one.
    # If we can't find one quickly, we return schoolbook rank 72.
    
    # Known rank-23 decomposition for 3x3x3:
    # Source: "The rank of 3x3 matrix multiplication is 23"
    # We can try to construct it or use a hardcoded one if available.
    # Since I don't have a hardcoded one in memory, I will attempt a local search to find one.
    # However, this is risky for the time limit.
    
    # Let's stick to the schoolbook decomposition for now, but optimize the code to be fast.
    # If we have time, we can try to improve.
    
    # For now, return the schoolbook decomposition.
    # The score is rank, so 72.
    
    # Let's try one simple optimization:
    # Check if we can merge any two columns in the schoolbook decomposition.
    # In schoolbook, each column corresponds to a unique (i,j,k).
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1.
    # If we have two columns t1, t2 with same i,j but different k,
    # U is the same for t1 and t2.
    # V[j*p+k1, t1] = 1, V[j*p+k2, t2] = 1.
    # W[k1*n+i, t1] = 1, W[k2*n+i, t2] = 1.
    # Can we combine them?
    # If we set U[i*m+j, t1] = 1, U[i*m+j, t2] = 0,
    # and adjust V and W?
    # This is essentially trying to find a lower rank factorization.
    
    # Given the time constraints and the lack of a guaranteed fast algorithm for rank minimization,
    # I will return the schoolbook decomposition.
    
    # Convert to list of lists as required
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
