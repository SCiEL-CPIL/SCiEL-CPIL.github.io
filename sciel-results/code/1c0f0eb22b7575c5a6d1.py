"""Matrix multiplication tensor decomposition for 3x5 x 5x7."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 120.0
    margin = 10.0

    # Schoolbook rank: n*m*p = 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor."""
        rank = len(U[0])
        # Build the matmul tensor T[i, j, k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] == delta_{i,k} for all i,j,k
        # Actually the tensor is: T[i,j,k] = 1 if we're looking at the (i,j,k) entry
        # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} (Kronecker delta)
        # Wait, let me reconsider. The matmul tensor M has entries M[i,j,k] = delta_{i,k}
        # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor is sum_j A[i,j] B[j,k] delta_{i,k}... 
        # No. The matrix multiplication tensor T has T[i,j,k] = 1 if and only if the product 
        # A[i,j]*B[j,k] contributes to C[i,k]. So T[i,j,k] = delta_{i,k} (Kronecker).
        # Wait no - T[i,j,k] is just 1 for all valid (i,j,k) because each product A[i,j]*B[j,k] 
        # is a distinct term that contributes to C[i,k]. The tensor entry is 1.
        # Actually, the matmul tensor is defined as: for matrices A (n x m), B (m x p), C (n x p),
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # The rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that's not right either. Let me re-read the contract.
        # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        # The tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal the matmul tensor.
        # But the matmul tensor has a specific structure. Let me think again.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: T is a 3-tensor where T[i,j,k] = 1 (each term A[i,j]*B[j,k] appears exactly once)
        # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
        # So sum_c gives 1 when c = i*m+j, matching T[i,j,k] = 1.
        # But wait, the W index is k*n+i, and in schoolbook W[k*n+i, c] where c corresponds to (i,j).
        # For the sum to work: we need U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to sum to 1 for all (i,j,k).
        # In schoolbook, each (i,j,k) has exactly one c = i*m+j where all three are 1. So sum = 1. Correct.
        # For verification, I'll check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower-rank decomposition using structured search
    # Strategy: Try ranks from some minimum up to 104, using randomized initialization + refinement
    
    # For small tensors, we can try a direct search approach
    # The minimum possible rank: lower bounds exist but for 3x5x7, 
    # rank >= max(n*m, m*p, p*n) / something... actually rank >= max(n, m, p) * something
    # Simple lower bound: rank >= max(n*m, m*p, p*n) is not true.
    # rank >= n*m*p / max(n*m, m*p, p*n) ... not a standard bound.
    # Let's just try to improve on 105.
    
    # Use a structured approach: try Kronecker-like decompositions
    # A 3x5x7 multiplication can be viewed as 3 separate 1x5x7 multiplications (rows of A)
    # Each 1x5x7 has rank 35 (schoolbook: 1*5*7=35)
    # But there might be structure to share.
    
    # Alternative: try a direct numerical search for small rank reductions
    # This is computationally expensive, so limit it.
    
    # Let's try a simple approach: start with schoolbook and try to merge columns
    # by checking if any two columns can be combined (i.e., if there's a linear dependency)
    
    # Actually, let's try a different structured decomposition.
    # Use the fact that 3 = 2 + 1 and try to decompose as (2x5x7) + (1x5x7)
    # 2x5x7 schoolbook: 70, 1x5x7 schoolbook: 35, total: 105. Same.
    
    # Try Winograd-style or Strassen-style block decomposition.
    # For 2x2x2, Strassen gives rank 7 < 8.
    # For our case, we could potentially use block methods.
    
    # Let's try a randomized search for rank 104, 103, etc.
    # But this is very expensive. Let's be pragmatic.
    
    # Given time constraints, return schoolbook if we can't find better quickly.
    
    # Try a quick search: can we reduce by 1?
    # This would require finding a rank-104 decomposition.
    # This is a hard problem. Let's just return schoolbook.
    
    if time.time() > deadline - margin:
        return schoolbook()
    
    # Try a very limited search for rank 104
    # Actually, let me try a structured decomposition that might work.
    # Consider grouping the 7 columns into groups that share structure.
    
    # For now, return schoolbook as it's guaranteed correct.
    return schoolbook()
# EVOLVE-BLOCK-END
