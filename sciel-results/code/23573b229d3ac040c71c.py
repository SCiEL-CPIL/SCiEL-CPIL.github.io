"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def schoolbook_decomposition():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build the tensor from decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    T[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    
    # The matrix multiplication tensor should be:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matrix mult M_{n,m,p} is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = delta_{j,j} = 1 for all valid i,j,k? No.
    # 
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor where:
    # M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (with appropriate indices)
    # Actually, it's defined by: for matrices A (n×m) and B (m×p),
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{ij} ⊗ E_{jk} ⊗ E_{ki}
    # where E_{ij} is the matrix with 1 at position (i,j)
    #
    # In index notation: T[i,j,k] should equal 1 for all i,j,k in the decomposition
    # to reproduce C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Wait, that's not right either. Let me think more carefully.
    #
    # The matrix multiplication tensor M_{n,m,p} ∈ K^{n×m} ⊗ K^{m×p} ⊗ K^{n×p}
    # is defined by:
    # M_{n,m,p} = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    #
    # So when we apply this tensor to A and B:
    # <M, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # This equals trace(C^T * (A*B)) when C is the output.
    #
    # For the decomposition U, V, W with:
    # U ∈ K^{n*m × rank}, V ∈ K^{m*p × rank}, W ∈ K^{p*n × rank}
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Because the tensor M has value 1 at every (i,j,k) position.
    
    # So T[i,j,k] should all be 1
    expected = np.ones((n, m, p))
    
    if np.allclose(T, expected, atol=1e-9):
        return True, rank
    else:
        return False, rank

def try_rank_decomposition(max_rank, timeout=100):
    """Try to find a decomposition with the given rank using random initialization and verification."""
    n, m, p = DIMENSIONS
    deadline = time.time() + timeout
    
    # We'll try various approaches
    # Approach 1: Randomized search with verification
    
    # The constraint is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 32 equations
    # With rank * (n*m + m*p + p*n) = rank * (8 + 16 + 8) = 32*rank unknowns
    
    # For rank < 32, we have more unknowns than equations, so solutions should exist
    # if the system is consistent.
    
    # Let's try a structured approach:
    # For n=2, we can think of this as two independent row multiplications
    # Row i of A times B gives row i of C.
    
    # Actually, let me try a direct construction approach.
    # 
    # Key insight: The tensor M_{2,4,4} can be viewed as:
    # M = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{3} e_i ⊗ e_j ⊗ e_k (in appropriate spaces)
    #
    # One approach: use the fact that we can share some terms between the two rows.
    
    # Let's try a parametric approach with small rank.
    # For rank r, we need to find U (8×r), V (16×r), W (8×r) such that
    # for all i,j,k: sum_{x=0}^{r-1} U[i*4+j, x] * V[j*4+k, x] * W[k*2+i, x] = 1
    
    # This is a 3-tensor decomposition problem.
    
    # Let's try rank from 16 down to see what works.
    for rank in range(max_rank, 7, -1):  # Try from max_rank down to 8 (must be >= 8 at minimum)
        if time.time() > deadline:
            break
        
        # Try random initialization
        np.random.seed(rank * 12345)
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = np.random.randint(-1, 2, (n * m, rank)).astype(float)
            V = np.random.randint(-1, 2, (m * p, rank)).astype(float)
            W = np.random.randint(-1, 2, (p * n, rank)).astype(float)
            
            # Check
            ok, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist())
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct_structured():
    """Try a structured decomposition for 2x4x4."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # For 2x4x4, let's think about what structure might help.
    # 
    # The key observation: when n=2, the W factor has only 8 rows (p*n = 8).
    # Each column of W is an 8-dimensional vector indexed by (k,i) pairs.
    #
    # Let's try a decomposition where we group terms.
    #
    # Idea: Use rank 16 with a block structure.
    # For each (j,k) pair in B, we need contributions to both rows of C.
    #
    # Actually, let me try a different structural idea:
    # Decompose the 4x4 multiplication into smaller pieces.
    #
    # 4x4 = 2x2 blocks:
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # A = [A1, A2] where A1, A2 are 1x4 = [A11, A12] and [A21, A22] (each 1x2)
    #
    # C1 = A1 * B = A11*B11 + A12*B21 + A11... wait, this is getting complex.
    
    # Let me try a more direct approach: enumerate possible small-rank decompositions
    # by exploiting symmetries.
    
    # For rank r, the unknowns are:
    # U: 8*r entries
    # V: 16*r entries  
    # W: 8*r entries
    # Total: 32*r unknowns
    # Constraints: 32 (one for each (i,j,k))
    
    # For r=8: 256 unknowns, 32 constraints - very underdetermined
    # So solutions should exist! The question is whether they can have integer/half-integer entries.
    
    # Let's try to construct rank 8 explicitly.
    # 
    # We need: for all i,j,k in {0,1,2,3} (with i in {0,1}):
    # sum_{x=0}^{7} U[i*4+j, x] * V[j*4+k, x] * W[k*2+i, x] = 1
    
    # This is a hard constraint satisfaction problem.
    # Let's try a systematic construction.
    
    # Approach: Assume U and W have simple structures.
    # Let U[i*4+j, x] depend only on i and x (not on j) for some x, etc.
    
    # Actually, let me try a greedy approach:
    # Build the decomposition term by term.
    
    # For each (i,j,k), we need the sum to be 1.
    # Let's assign values greedily.
    
    # Start with rank 8
    rank = 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We have 32 constraints and 8 rank terms.
    # Each term contributes to all 32 positions (i,j,k) as a product.
    
    # Let's try: for term x, assign a pattern.
    # Term x covers a subset of (i,j,k) positions.
    
    # Simple approach: each term covers 4 positions (since 32/8 = 4)
    # Position (i,j,k) gets contribution from terms where the product is 1.
    
    # Let's define 8 groups of 4 (i,j,k) tuples:
    positions = list(itertools.product(range(n), range(m), range(p)))
    assert len(positions) == 32
    
    # Group them: 8 groups of 4
    groups = []
    for g in range(rank):
        groups.append(positions[g*4:(g+1)*4])
    
    # For each group, we need to find U_vals, V_vals, W_vals such that
    # for all (i,j,k) in the group: U[i*4+j] * V[j*4+k] * W[k*2+i] = 1
    # And for (i,j,k) not in the group: the product = 0 (or cancels out)
    
    # This is tricky. Let me try a different approach.
    
    # Use the fact that we can have negative values.
    # Try to find a set of 8 rank-1 tensors whose sum is the all-ones tensor.
    
    # A rank-1 tensor is defined by vectors u (8-dim), v (16-dim), w (8-dim)
    # such that T[i,j,k] = u[i*4+j] * v[j*4+k] * w[k*2+i]
    
    # We need sum of 8 such tensors = ones tensor.
    
    # Let me try: use Fourier-type characters or similar.
    
    # Actually, let me just try a numerical optimization approach with scipy.
    from scipy.optimize import minimize
    
    def objective(params):
        # params: 32*rank values flattened
        U_flat = params[:8*rank]
        V_flat = params[8*rank:8*rank+16*rank]
        W_flat = params[8*rank+16*rank:]
        
        U_mat = U_flat.reshape(8, rank)
        V_mat = V_flat.reshape(16, rank)
        W_mat = W_flat.reshape(8, rank)
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*4+j, x] * V_mat[j*4+k, x] * W_mat[k*2+i, x] for x in range(rank))
                    error += (val - 1)**2
        return error
    
    # Initial guess: random
    np.random.seed(42)
    x0 = np.random.randint(-1, 2, 32*rank).astype(float) * 0.1
    
    # This might be too slow for 150s. Let's check timing.
    # For rank 8, 32*8=256 variables, 32 constraints.
    # minimize might take a while.
    
    # Let me try a simpler approach: solve the linear system after fixing some variables.
    
    # Actually, the equations are trilinear, not linear. So we can't just solve a linear system.
    
    # Let me try a different tactic: use the structure of small dimensions.
    
    # For n=2, p=4: W has 8 rows indexed by (k,i) for k in 0..3, i in 0..1.
    # Let's denote W[k*2+i, x] = W_ki_x
    
    # The constraint for (i,j,k) is:
    # sum_x U_{i,j,x} * V_{j,k,x} * W_{k,i,x} = 1
    
    # Let's fix W to simple values and solve for U, V.
    
    # Try: W_{k,i,x} = 1 for x = i*4+k, else 0. (sparse W)
    # Then for each (i,j,k), only x = i*4+k contributes:
    # U_{i,j,i*4+k} * V_{j,k,i*4+k} * 1 = 1
    # So U_{i,j,i*4+k} = 1/V_{j,k,i*4+k}
    
    # But U has only 8 rows and rank 8, so U_{i,j,x} for x in 0..7.
    # i*4+k ranges over 0..7 for i in {0,1}, k in {0,1,2,3}.
    # So this works! Each (i,k) pair maps to a unique x.
    
    # But we need: for all j, U_{i,j,i*4+k} * V_{j,k,i*4+k} = 1
    # And U_{i,j,x} = 0 for x != i*4+k (for the relevant k)
    
    # Wait, but U_{i,j,x} can be nonzero for other x too, as long as the sum works out.
    # With sparse W, only one x contributes per (i,k), so:
    # U_{i,j,i*4+k} * V_{j,k,i*4+k} = 1 for all i,j,k
    
    # Let's set V_{j,k,i*4+k} = 1 for all j,k,i.
    # Then U_{i,j,i*4+k} = 1 for all i,j,k.
    
    # But U_{i,j,x} where x = i*4+k. For fixed i and j, as k varies, x varies.
    # U has shape (8, 8). Row i*4+j has entries for x=0..7.
    # For row i*4+j, we need: for each k, U_{i*4+j, i*4+k} = 1.
    # So row i*4+j has 1s at columns i*4+k for k=0,1,2,3.
    # That's 4 ones per row.
    
    # Let's construct this:
    U = np.zeros((8, 8))
    V = np.zeros((16, 8))
    W = np.zeros((8, 8))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * 4 + k  # index into rank
                U[i*4 + j, x] = 1
                V[j*4 + k, x] = 1
                W[k*2 + i, x] = 1
    
    # Check: for (i,j,k), the sum over x of U[i*4+j,x]*V[j*4+k,x]*W[k*2+i,x]
    # Only x = i*4+k has all three nonzero (assuming no other x has all three).
    # For x = i*4+k: U[i*4+j, i*4+k]=1, V[j*4+k, i*4+k]=1, W[k*2+i, i*4+k]=1. Product=1.
    # For other x: need to check if any other x gives product=1.
    
    # Check: W[k*2+i, x] = 1 only when x = i*4+k.
    # V[j*4+k, x] = 1 only when... we set V[j*4+k, x]=1 when x=i*4+k for the specific i.
    # But V[j*4+k, x] is set for multiple i values!
    # V[j*4+k, x] where x = i*4+k. For fixed j,k, x ranges over {0*4+k, 1*4+k} = {k, 4+k}.
    # So V[j*4+k, k] = 1 (from i=0) and V[j*4+k, 4+k] = 1 (from i=1).
    
    # Similarly, U[i*4+j, x] = 1 when x = i*4+k. For fixed i,j, x ranges over {i*4+0, i*4+1, i*4+2, i*4+3}.
    # So U[i*4+j, x] = 1 for x in {i*4, i*4+1, i*4+2, i*4+3}.
    
    # W[k*2+i, x] = 1 when x = i*4+k. For fixed k,i, x = i*4+k.
    # So W[k*2+i, x] = 1 only at x = i*4+k.
    
    # Now for the product U[i*4+j, x]*V[j*4+k, x]*W[k*2+i, x]:
    # W[k*2+i, x] is nonzero only at x = i*4+k.
    # At x = i*4+k:
    #   U[i*4+j, i*4+k] = 1 (since i*4+k is in {i*4, i*4+1, i*4+2, i*4+3})
    #   V[j*4+k, i*4+k]: V[j*4+k, x] = 1 when x = i'*4+k for i' in {0,1}.
    #   At x = i*4+k: V[j*4+k, i*4+k] = 1 (since i is either 0 or 1, and x = i*4+k matches i'=i).
    #   W[k*2+i, i*4+k] = 1.
    # Product = 1. Good.
    
    # But are there other x where the product is nonzero?
    # W[k*2+i, x] = 0 for x != i*4+k. So no other x contributes.
    
    # So the sum is exactly 1 for each (i,j,k). This works!
    
    # But wait: we need to verify that no other (i',j',k') accidentally gets a contribution.
    # The constraint is per (i,j,k), and we've shown each gets exactly 1.
    
    # Let me verify with code:
    ok, r = verify_decomposition(U.tolist(), V.tolist(), W.tolist())
    if ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If the above didn't work (shouldn't happen), fall back to schoolbook
    return schoolbook_decomposition()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the structured rank-8 decomposition
    try:
        result = construct_structured()
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END

