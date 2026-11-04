<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Schoolbook baseline
    rank_school = n * m * p  # 96
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Verify schoolbook
    # T_recon = np.einsum('ir,jr,kr->ijk', U_school, V_school, W_school)
    # if np.array_equal(T_recon, T):
    #     valid = True
    # else:
    #     valid = False
    # We trust the construction.
    
    # Attempt to find a lower rank decomposition
    # Strategy: Check if any term is redundant.
    # Given the disjoint support property, terms are linearly independent in the ambient space.
    # However, tensor rank might be lower due to algebraic identities.
    # For 3x4x8, no known algebraic identities reduce the rank below 96 in standard literature 
    # for small dimensions without complex group theory.
    # We will return the schoolbook decomposition as it is the safest and fastest valid solution.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Schoolbook baseline
    rank_school = n * m * p  # 96
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, T_target, tol=1e-10):
        """Verify that U, V, W exactly decompose T_target."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Reconstruct tensor
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol)
    
    def try_block_decomposition():
        """
        Try to decompose the tensor by exploiting block structure.
        The tensor T[i*m+j, j*p+k, k*n+i] = 1 can be seen as:
        For each j, we have a sub-tensor T_j[i, k] = 1 when indices align.
        
        Actually, let's think differently. The matrix multiplication C = A*B where
        A is n×m, B is m×p, C is n×p.
        
        T_{i,j,k} = 1 if the term contributes to C[i,k] via A[i,j]*B[j,k].
        
        In our indexing:
        - First index: i*m + j (row of A, col of A)
        - Second index: j*p + k (row of B, col of B)  
        - Third index: k*n + i (row of C, col of C)
        
        So T[i*m+j, j*p+k, k*n+i] = 1
        
        This is the standard matrix multiplication tensor.
        
        Strategy: Use known low-rank decompositions for small matrix multiplications.
        For 3×4×8, we can try:
        1. Split into blocks and use rank-1 decompositions for each block
        2. Use Strassen-like techniques
        
        Let's try a structured approach:
        - The tensor has support only when the middle indices match (j in first and second)
        - We can try to find a decomposition with rank < 96
        """
        if time.time() > deadline - 30:
            return None
        
        # Try to find a decomposition using the fact that the tensor is a sum of
        # rank-1 tensors corresponding to each (i,j,k) triple.
        # But we want to see if we can combine them.
        
        # For now, let's try a greedy approach: check if we can remove any rank-1 term
        # while maintaining the decomposition.
        
        # Actually, a better approach: use the fact that for matrix multiplication,
        # there are known lower bounds and constructions.
        
        # For 3×4×8, let's try to see if rank 95 is possible by checking if
        # any two rank-1 terms can be merged.
        
        # This is complex, so let's try a different approach:
        # Use the block structure. The tensor can be written as:
        # T = sum_j T_j where T_j is the sub-tensor for fixed j.
        
        # T_j has support at (i, k) positions for all i in [0,n), k in [0,p)
        # Specifically, T_j[i*m+j, j*p+k, k*n+i] = 1
        
        # Each T_j is itself a matrix multiplication tensor for n×1 and 1×p,
        # which has rank n*p = 24.
        
        # But we can potentially share terms across different j values.
        
        # Let's try to construct a decomposition with rank < 96.
        
        # One idea: use the fact that 4 = 2*2 and try to use Strassen's algorithm
        # for the middle dimension.
        
        # Actually, let's just verify the schoolbook and return it if we can't
        # find something better quickly.
        
        return None
    
    # Try to find a better decomposition
    result = try_block_decomposition()
    if result is not None:
        best_U, best_V, best_W, best_rank = result
    
    # If time permits, try more sophisticated searches
    if time.time() < deadline - 60:
        # Try random perturbations and rank reduction
        try_random_reduction()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I realize I need to actually implement a working search. Let me provide a more complete solution:

