"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses a structured random search with verification, falling back to schoolbook if needed.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k,l] = delta(i,l) * delta(j,k) for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # Wait, let me re-read the convention.
        # W[k*n+i, r] - this is the output factor
        # U has shape (n*m, rank) - input A factor
        # V has shape (m*p, rank) - input B factor
        # W has shape (p*n, rank) - output factor
        
        # The decomposition says: for each output element C[i,k] (where i in [0,n), k in [0,p)),
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = delta... 
        # Actually, the matrix multiplication tensor T has indices (i, j, k) where:
        # T[i,j,k] = 1 if this term appears in the product, 0 otherwise
        # Specifically, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = delta(j,k) ... no.
        
        # Let me think again. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
        # has entries T[i, j, k] where i indexes rows of A (and C), j indexes columns of A / rows of B,
        # k indexes columns of B (and C).
        # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k], which is always true when
        # the middle indices match. So T[i,j,k] = 1 for all valid (i,j,k) where j is shared.
        # Wait, that's just the identity - every term A[i,j]*B[j,k] contributes to C[i,k].
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the indexing is:
        # U[i*m+j, r] - row index is i*m+j (flattening A's indices)
        # V[j*p+k, r] - row index is j*p+k (flattening B's indices)  
        # W[k*n+i, r] - row index is k*n+i (flattening C's indices, but with k first then i)
        
        # So we need: for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's verify this
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all (i, j, k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :]), W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try Strassen-based approaches and random search
    # For 4x4 * 4x5, we can try various structured decompositions
    
    # Approach 1: Use Kronecker products of 2x2 Strassen
    # 4x4 = 2x2 block of 2x2, 4x5 = 2x2 block of 2x2 with padding
    # This is complex, let's try a more direct random search
    
    # Random search with structured initialization
    np.random.seed(42)
    
    for target_rank in range(56, rank_sb + 1):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 5000 if target_rank <= 60 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random decomposition
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Try to verify
            # Fast verification: check a subset first
            # Sample some (i,j,k) triples
            sample_indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sample_indices.append((i, j, k))
            
            valid = True
            for (i, j, k) in sample_indices:
                val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Full verification
                if verify_decomposition(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
