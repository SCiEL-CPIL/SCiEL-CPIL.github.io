"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Trivial decomposition has rank n*m*p = 72
    # Lower bound is max(n*m, m*p, p*n) = 24
    
    # Strategy: Use structured decomposition based on the algebraic structure
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T has T[i,j,k] = 1 for all (i,j,k)
    # A rank-r decomposition gives:
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Key insight: For each j, the map (i,k) -> A[i,j]*B[j,k] is rank 1.
    # We can try to find a decomposition that exploits this structure.
    #
    # Approach: Try ranks from lower bound upward, use structured initialization
    # based on known small-case algorithms, then verify.
    
    # First, let's try to construct a decomposition with rank close to optimal.
    # For 3x3x3, Laderman gives rank 23.
    # For 3x3x8, we can think of it as 3x3x3 * (8/3) roughly.
    # A reasonable target might be around 3*8 + some overhead.
    
    # Let's try a systematic search with structured patterns.
    # The idea: partition the j-index into groups and use Strassen-like identities.
    
    # Try several candidate ranks
    best_result = None
    best_rank = n * m * p  # Start with trivial
    
    # Generate the trivial decomposition as fallback
    def trivial_decomp():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}, rank
    
    # Try to find better decompositions using structured approaches
    # For each candidate rank, try to solve the system
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly represents the multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions
    # Idea: Use the fact that for fixed j, we have outer products
    # We can try to combine terms across j using algebraic identities
    
    # Method 1: Direct construction using known patterns
    # For 3x3x3, one approach uses 23 terms. Let's try to extend this.
    
    # Method 2: Use random search with exact arithmetic verification
    # Try small integer/half-integer values
    
    # Let's try a greedy/structured approach:
    # For each j in range(m), we need to represent the "slice" T[:,j,:]
    # which is an n x p matrix of all 1s.
    # The rank of this slice is 1 (it's all ones), but we need the full 3D structure.
    
    # Key: The full tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = (1 for all i,j,k)
    # This is the tensor product of three all-ones vectors: 1_n ⊗ 1_m ⊗ 1_p
    # But that's rank 1 in the tensor sense only if we allow arbitrary coefficients.
    # Wait, no: the tensor T[i,j,k] = 1 for ALL (i,j,k) is indeed rank 1!
    # It's just the outer product of (1,1,...,1) with itself.
    # 
    # But wait - that can't be right for matrix multiplication. Let me reconsider.
    # 
    # The multiplication tensor is NOT T[i,j,k] = 1 for all (i,j,k).
    # The multiplication tensor encodes: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 if the term A[i,j]*B[j,k]
    # contributes to C[i,k]. So T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # But the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    # 
    # So we need the sum of products to equal 1 for each (i,j,k).
    # This is NOT the same as the tensor being rank 1. The tensor T[i,j,k]=1
    # as a 3D array has rank 1 (it's the outer product of three all-ones vectors),
    # but the DECOMPOSITION constraint is different because the indices are coupled
    # in a specific way through the U, V, W factor structure.
    #
    # Actually, I think I'm confusing myself. Let me re-examine.
    #
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor with entries
    # M_{i,j,k} = 1 for all (i,j,k) in the valid range. The rank of this tensor
    # (as a 3-way tensor) is indeed 1, since M = 1_n ⊗ 1_m ⊗ 1_p.
    #
    # BUT - the "rank" in the context of matrix multiplication algorithms is NOT
    # the tensor rank of the all-ones tensor. It's the rank of the multiplication
    # operation, which is the minimum number of scalar multiplications needed.
    #
    # The connection is: A rank-r decomposition of the multiplication tensor gives
    # an algorithm with r multiplications. The tensor here is NOT the all-ones tensor
    # in the naive sense. Let me look at this more carefully.
    #
    # Actually, I think the standard formulation is:
    # The multiplication tensor T ∈ R^{n×m × m×p × p×n} where
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and i=i' and k=k', else 0.
    # 
    # No wait, that's not right either. Let me think about this from first principles.
    #
    # We want to compute C = A * B where A is n×m, B is m×p, C is n×p.
    # C[i,k] = Σ_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # A bilinear algorithm with r multiplications computes:
    # C[i,k] = Σ_{r=0}^{r-1} (Σ_{i,j} U[i,j,r] * A[i,j]) * (Σ_{j,k} V[j,k,r] * B[j,k]) * W[i,k,r]
    #
    # For this to be correct, we need:
    # Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = δ_{j,j'} (the j-index must match)
    #
    # Wait, I think the correct condition is:
    # For the algorithm to correctly compute C[i,k] = Σ_j A[i,j] * B[j,k],
    # we need:
    # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = δ_{j,j'}
    #
    # So the tensor is actually the identity in the j-dimension, not all ones!
    # T[i,j,j',k] = δ_{j,j'} for the appropriate index structure.
    #
    # Given the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The condition should be:
    # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = δ_{j,j'}
    #
    # But the problem statement says the tensor is for matrix multiplication, and
    # the trivial decomposition has U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # for each (i,j,k). Let's verify: for a given (i,j',k), the sum over r where
    # the r-th term corresponds to (i_r, j_r, k_r):
    # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r]
    # = Σ_{(i_r,j_r,k_r)} 1[i*m+j == i_r*m+j_r] * 1[j'*p+k == j_r*p+k_r] * 1[k*n+i == k_r*n+i_r]
    # = 1 if there exists (i_r,j_r,k_r) such that i_r=i, j_r=j, k_r=k AND j_r=j', k_r=k, i_r=i
    # = 1 if j=j' (and i=i, k=k which is automatic)
    #
    # So the condition is: Σ_r ... = δ_{j,j'}
    #
    # This makes sense! The tensor has entries T[(i,j), (j',k), (k',i')] = δ_{j,j'} * δ_{i,i'} * δ_{k,k'}
    # which simplifies to δ_{j,j'} when we fix the other indices.
    
    # OK so now I understand the problem correctly.
    # We need: Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = δ_{j,j'}
    # for all i, j, j', k.
    
    # Let me rewrite the verification:
    def verify_decomp_correct(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[jp * p + k][r] * W[k * n + i][r]
                        expected = 1 if j == jp else 0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    # Now, the trivial decomposition:
    # For each (i,j,k), set U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # Check: Σ_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
    # = 1 if (i,j,k) matches in all three, i.e., j=jp
    # = δ_{j,jp} ✓
    
    # Now let's try to find better decompositions.
    # The key insight: we can use algebraic identities to reduce the number of terms.
    
    # For m=3, we can use the identity:
    # For any a,b,c: (a+b)(b+c)(c+a) - a*b*c - ... (Strassen-like)
    
    # Let's try a systematic numerical search.
    # We'll try ranks from 24 upward and use random initialization + verification.
    
    # Actually, let's try a smarter approach: use the structure of the problem.
    # The condition is:
    # Σ_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = δ_{j,jp}
    
    # For fixed i,k, this is a condition on the j,jp indices.
    # Let's define for each (i,k):
    # M_{i,k}[j,jp] = Σ_r U[i*m+j, r] * W[k*n+i, r] * V[jp*p+k, r]
    # We need M_{i,k} = I_m (the m×m identity)
    
    # So for each (i,k) pair, we need the matrix product to give identity.
    # This means: for each (i,k), the vectors {U[i*m+j, r] * W[k*n+i, r] for j,r}
    # and {V[jp*p+k, r] for jp,r} must form dual bases.
    
    # A simple construction: for each (i,k), use m terms where
    # U[i*m+j, r_{i,k,j}] = 1, W[k*n+i, r_{i,k,j}] = 1, V[j*p+k, r_{i,k,j}] = 1
    # This gives m terms per (i,k), total n*p*m = 72 terms. That's the trivial one.
    
    # To do better, we need to share terms across different (i,k) pairs.
    # This is possible if the required identity matrices can be represented
    # with fewer shared rank-1 components.
    
    # Let's try a random search with exact verification.
    # Use small integer values to increase chance of exact solutions.
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(24, 73):
        if time.time() - start_time > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() - start_time > deadline:
                break
                
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomp_correct(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return trivial
    result, rank = trivial_decomp()
    return result
# EVOLVE-BLOCK-END
