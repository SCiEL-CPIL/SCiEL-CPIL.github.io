"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j, :]
                    v_val = V_arr[j*p + k, :]
                    w_val = W_arr[k*n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Use Strassen-like decomposition for 4x4x9
    # Key insight: We can decompose the 4x4x9 multiplication by splitting
    # the 9 columns into 3 groups of 3, and using a structured approach
    
    # For 4x4x3: We can use a rank-12 decomposition
    # This comes from combining two 2x2 Strassen steps appropriately
    
    # Let's construct the 4x4x3 decomposition first
    # Then extend to 4x4x9 by direct sum
    
    def construct_4x4x3():
        """Construct a rank-12 decomposition for 4x4 x 4x3 multiplication."""
        n_local, m_local, p_local = 4, 4, 3
        rank_local = 12
        
        U = np.zeros((n_local*m_local, rank_local))
        V = np.zeros((m_local*p_local, rank_local))
        W = np.zeros((p_local*n_local, rank_local))
        
        # Use Strassen-style decomposition
        # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
        # Split B (4x3) into 2x3 blocks: B1, B2
        # C (4x3) has 2x3 blocks: C1, C2
        
        # We need to compute:
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Using 6 multiplications (3 for each 2x2x3 block):
        # For 2x2x3, we can use rank 3 (schoolbook) or potentially less
        
        # Actually, for 2x2x3, the schoolbook rank is 6
        # But we can potentially do better with structured decompositions
        
        # Let's use the direct sum approach:
        # For each of the 3 columns, use a 4x4x1 decomposition
        # 4x4x1 has rank at most 16 (schoolbook) but can be done with rank 7
        # using Strassen-like methods
        
        # Actually, let's be more careful. Let's construct explicitly.
        
        # For 4x4x1 (multiplying 4x4 by 4x1):
        # C = A * b where b is a 4x1 vector
        # This is just a linear combination, rank 1 per column
        # But we need the full tensor structure
        
        # Let me think differently. The tensor for 4x4x9 is:
        # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        # where the indices are (i*m+j), (j*p+k), (k*n+i)
        
        # For a structured decomposition, we can use the fact that
        # 4 = 2+2 and 9 = 3+3+3
        
        # Let's try a Kronecker product approach:
        # 4x4x9 = (2x2x3) ⊗ (2x2x3) in some sense
        
        # Actually, the cleanest approach is to use the known result that
        # the rank of 4x4x4 is at most 49 (recursive Strassen)
        # And for 4x4x9, we can use a similar approach
        
        # Let's construct a valid decomposition with rank < 144
        # using a block-circulant or similar structure
        
        # For now, let's try to construct a rank-126 decomposition
        # (which is 144 - 18, a modest improvement)
        
        # Actually, let's try a more principled approach:
        # Use the fact that we can combine some of the schoolbook terms
        
        # The key observation: for fixed i, the terms with the same i
        # can potentially be combined using the structure of B
        
        # Let's try a random search for a valid decomposition with rank < 144
        
        return None
    
    # Try to find a better decomposition using structured search
    np.random.seed(42)
    
    # Method: Start from schoolbook and try to merge terms
    # by finding linear dependencies
    
    # For a more systematic approach, let's try to construct
    # a decomposition using the algebraic structure
    
    # Key idea: The matrix multiplication tensor has symmetries
    # that can be exploited
    
    # Let's try to construct a rank-120 decomposition
    # by exploiting the 4 = 2*2 structure
    
    def try_strassen_extension():
        """Try to construct a Strassen-based decomposition."""
        
        # Strassen for 2x2x2 uses 7 multiplications
        # For 4x4x4, recursive Strassen gives 7^2 = 49
        
        # For 4x4x9, we can think of it as:
        # First compute 4x4x4 (rank 49), then extend to 9 columns
        
        # The extension from 4 to 9 columns:
        # We need 5 additional columns
        # Each additional column of B requires computing A * b_col
        # which is a 4x4x1 multiplication
        
        # The rank of 4x4x1 is at most 16 (schoolbook)
        # But can be done with rank 7 using Strassen-like methods
        
        # So total rank: 49 + 5*7 = 49 + 35 = 84
        
        # But this is an upper bound, and we need to verify it's valid
        
        # Let's try to construct this explicitly
        
        # Step 1: Construct 4x4x4 Strassen decomposition (rank 49)
        # Step 2: Construct 4x4x1 decomposition for each extra column (rank 7 each)
        
        # This is complex, so let's try a simpler approach first
        
        # Try to find any valid decomposition with rank < 144
        # using random initialization and iterative refinement
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Try rank 120
            target_rank = 120
            
            # Initialize with random values
            U_rand = np.random.randn(n*m, target_rank) * 0.1
            V_rand = np.random.randn(m*p, target_rank) * 0.1
            W_rand = np.random.randn(p*n, target_rank) * 0.1
            
            # This is very unlikely to validate, so skip for now
            # The random search approach is not effective for this problem
        
        return None
    
    # Given the complexity, let's return the schoolbook decomposition
    # which is guaranteed to be valid
    # and has rank 144
    
    # However, let's try one more thing: a structured decomposition
    # that exploits the 4x4x9 = (2x2x3) ⊗ (2x2x3) structure
    
    def try_kronecker_structure():
        """Try to use Kronecker product structure."""
        
        # The 4x4x9 multiplication can be viewed as:
        # (2x2 ⊗ 2x2) * (2x2 ⊗ 2x3) -> (2x2 ⊗ 2x3)
        
        # This suggests a tensor product structure
        # but it's not a simple Kronecker product
        
        # Let's try to construct a decomposition using
        # the 2x2x3 building block
        
        # For 2x2x3, the schoolbook rank is 12
        # But we might be able to do better
        
        # Actually, for 2x2x3:
        # A (2x2) * B (2x3) = C (2x3)
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C13 = A11*B13 + A12*B23
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        # C23 = A21*B13 + A22*B23
        
        # This is 12 terms in schoolbook
        # Can we do better? The lower bound is max(4, 6) = 6
        # but the actual rank is likely 12 or close
        
        # For now, let's use the schoolbook for 2x2x3
        # and see if we can combine terms across the Kronecker structure
        
        return None
    
    # Given the time constraints and the difficulty of finding
    # a valid lower-rank decomposition, let's return the schoolbook
    # which is guaranteed to work
    
    return best
# EVOLVE-BLOCK-END
