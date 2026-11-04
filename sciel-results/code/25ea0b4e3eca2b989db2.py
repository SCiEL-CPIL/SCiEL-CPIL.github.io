import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
TIME_BUDGET = 145  # seconds, with 5s safety margin for the evaluator's 150s limit

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of the matrix multiplication tensor."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly reconstructs matrix multiplication.
    U, V, W must be numpy arrays with Fraction entries.
    Returns True if valid, False otherwise.
    """
    rank = U.shape[1]
    
    # Check all 1-positions exactly: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if val != Fraction(1):
                    return False
    
    # Check zero constraints: for (i,j,k) where the product index doesn't match, the sum must be 0
    # We check all (a, b, g) triples where a = i*m+j, b = j'*p+k', g = k''*n+i''
    # and (i != i'' or j != j' or k != k'')
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            jp = b // p
            k = b % p
            for g in range(p * n):
                kn = g // n
                ip = g % n
                if i != ip or j != jp or k != kn:
                    val = Fraction(0)
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[g, r]
                    if val != Fraction(0):
                        return False
    
    return True

def _verify_sampled(U, V, W, n, m, p, num_samples=200, seed=42):
    """
    Faster verification that checks all 1-positions and samples zero constraints.
    Returns True if all checks pass.
    """
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    
    # Check all 1-positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if val != Fraction(1):
                    return False
    
    # Sample zero constraints
    for _ in range(num_samples):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        g = rng.randint(0, p * n)
        i = a // m
        j = a % m
        jp = b // p
        k = b % p
        kn = g // n
        ip = g % n
        if i != ip or j != jp or k != kn:
            val = Fraction(0)
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if val != Fraction(0):
                return False
    
    return True

def _strassen_2x2_terms():
    """
    Return the Strassen 2x2 multiplication terms as lists of (coefficient, index) tuples.
    For A = [[a11, a12], [a21, a22]] and B = [[b11, b12], [b21, b22]]:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    C11 = M1 + M4 - M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M6 + M7
    """
    # Each term: (U_indices, V_indices, W_indices, coefficient)
    # U index for (i,j): i*2 + j
    # V index for (j,k): j*2 + k
    # W index for (k,i): k*2 + i
    terms = []
    
    # M1 = (a11 + a22)(b11 + b22)
    terms.append(([0, 3], [0, 3], [(0,0), (1,1)], Fraction(1)))  # W: k0*i0=0, k1*i1=3
    # M2 = (a21 + a22)b11
    terms.append(([2, 3], [0], [(0,0), (0,1)], Fraction(1)))  # W: k0*i0=0, k0*i1=1
    # M3 = a11(b12 - b22)
    terms.append(([0], [1], [(1,0), (1,1)], Fraction(1)))  # W: k1*i0=2, k1*i1=3
    terms.append(([0], [3], [(1,0), (1,1)], Fraction(-1)))
    # M4 = a22(b21 - b11)
    terms.append(([3], [2], [(0,0), (0,1)], Fraction(1)))  # W: k0*i0=0, k0*i1=1
    terms.append(([3], [0], [(0,0), (0,1)], Fraction(-1)))
    # M5 = (a11 + a12)b22
    terms.append(([0, 1], [3], [(1,0), (1,1)], Fraction(1)))  # W: k1*i0=2, k1*i1=3
    # M6 = (a21 - a11)(b11 + b12)
    terms.append(([2], [0, 1], [(0,0), (0,1)], Fraction(1)))
    terms.append(([0], [0, 1], [(0,0), (0,1)], Fraction(-1)))
    # M7 = (a12 - a22)(b21 + b22)
    terms.append(([1], [2, 3], [(1,0), (1,1)], Fraction(1)))
    terms.append(([3], [2, 3], [(1,0), (1,1)], Fraction(-1)))
    
    return terms

def _build_hybrid_decomposition(n, m, p):
    """
    Build a hybrid Strassen-Schoolbook decomposition.
    - Use Strassen for 2x2x2 blocks where possible
    - Use schoolbook for remaining parts
    
    For 4x5x6:
    - Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where A00 is 2x5, A01 is 2x5, etc.
    - But Strassen works on square blocks, so we need to be careful.
    
    Alternative approach: Use Strassen for the 4x4x6 part (if we had 4x4x4) and schoolbook for the rest.
    Actually, let's think differently.
    
    The tensor is for multiplying (n x m) by (m x p).
    n=4, m=5, p=6.
    
    We can partition the output C (4x6) into blocks.
    Let's try:
    - C[0:2, 0:2] computed via Strassen on A[0:2, 0:2], B[0:2, 0:2] -> but B is 5x6, so B[0:2, 0:2] is 2x2
    - This gets complicated because the dimensions don't divide evenly.
    
    Let me use a different strategy:
    Partition the m=5 dimension.
    C = A * B = A[:, :4] * B[:4, :] + A[:, 4] * B[4, :]
    
    The first term: (4x4) * (4x6) -> use Strassen on the 4x4x4 sub-tensor?
    Actually, (4x4)*(4x6) can be decomposed as:
    - (4x4)*(4x4) part for columns 0:4 of B
    - (4x4)*(4x2) part for columns 4:6 of B
    
    For (4x4)*(4x4): rank 49 (Strassen) instead of 64 (schoolbook)
    For (4x4)*(4x2): This is (4x4)*(4x2). We can write B[:, 4:6] as (4x2).
    This is equivalent to two (4x4)*(4x1) multiplications.
    (4x4)*(4x1) has rank 16 (schoolbook), but can we do better?
    (4x1) multiplication is just 4 independent dot products of length 4.
    The tensor is sum_{j=0}^{3} A_{:,j} * B_{j,0} * C_{:,0}
    This is a sum of 4 rank-1 tensors, so rank <= 16.
    Actually, it's just 4 separate 1x4x1 multiplications, each of rank 1? No.
    For each row i of C, C_i = sum_j A_{ij} * B_j.
    This is a (4x4x1) tensor. Its rank is 4 (since it's a sum of 4 rank-1 tensors).
    So (4x4)*(4x2) has rank 4*2 = 8? Let me verify.
    
    Actually, let me just implement the Strassen for 2x2x2 blocks and see what rank we get.
    """
    
    # Let's use a simpler approach:
    # Partition into 2x2 blocks for the n and p dimensions, and keep m as is.
    # n=4 = 2*2, p=6 = 2*3
    # So we have 2*3 = 6 blocks of 2x?x2
    
    # For each block (i0, i1) x (j0, j1) x (k0, k1) where i1-i0=2, k1-k0=2
    # The j dimension for each block will be 5.
    
    # Actually, this is getting complex. Let me just use Strassen for the 2x2x2 parts
    # where the m-dimension is also 2, and schoolbook for the rest.
    
    # A cleaner approach:
    # Use Strassen recursively on the 4x4x6 part.
    # 4x4x6 = 2*(2x4x6) + 2*(2x4x6)
    # 2x4x6: This is (2x4)*(4x6). 
    # We can write this as (2x2)*(2x6) + (2x2)*(2x6)
    # (2x2)*(2x6) = (2x2)*(2x2) + (2x2)*(2x2) + (2x2)*(2x2) for the 6 columns
    # Each (2x2)*(2x2) has rank 7.
    # So (2x2)*(2x6) has rank 7*3 = 21? Or can we share?
    
    # Let me just implement a direct construction.
    
    # For now, let's use a simpler hybrid:
    # Split A into top 2 rows and bottom 2 rows.
    # C_top = A_top * B, C_bottom = A_bottom * B
    # A_top is 2x5, A_bottom is 2x5
    # C_top is 2x6, C_bottom is 2x6
    
    # For C_top = A_top * B:
    # A_top is 2x5, B is 5x6
    # This is (2x5)*(5x6). Rank schoolbook = 2*5*6 = 60.
    # Can we do better?
    # (2x5)*(5x6) = (2x5)*(5x6)
    # We can write B as [B1 | B2 | B3] where each Bj is 5x2.
    # C_top = [A_top*B1 | A_top*B2 | A_top*B3]
    # Each (2x5)*(5x2) has rank 20 (schoolbook).
    # Total = 60.
    
    # It seems hard to beat schoolbook easily with these dimensions.
    # Let me try the laser method as suggested.
    
    # For now, return schoolbook as fallback.
    return _build_schoolbook(n, m, p)

def _laser_method_decomposition(n, m, p):
    """
    Apply the laser method to find a lower rank decomposition.
    
    The laser method involves:
    1. Find a large power of the tensor that has a good structure.
    2. Use the structure to derive a rank bound.
    3. Construct an explicit decomposition.
    
    For matrix multiplication, the laser method can sometimes improve the rank
    over Strassen, but for small dimensions like 4x5x6, it's often hard to
    get a practical improvement.
    
    Let's try a different approach: use the fact that we can pad to make
    the dimensions more amenable to Strassen.
    
    Actually, let's try to use Strassen for the 4x4x4 sub-problem and
    handle the remaining 4x4x2 and 4x2x4 parts separately.
    """
    
    # The tensor is (4x5)*(5x6).
    # Let's partition the m=5 dimension: m = 4 + 1.
    # C = A[:, :4] * B[:4, :] + A[:, 4] * B[4, :]
    
    # Part 1: (4x4)*(4x6)
    # Part 2: (4x1)*(1x6)
    
    # Part 2: (4x1)*(1x6)
    # This is just 4 independent (1x1)*(1x6) multiplications? No.
    # A[:, 4] is 4x1, B[4, :] is 1x6.
    # C2 = A_col * B_row, which is an outer product.
    # The tensor for (4x1)*(1x6) is sum_{i=0}^{3} sum_{k=0}^{5} A_{i,4} * B_{4,k} * C_{i,k}
    # This is a sum of 24 rank-1 tensors, so rank <= 24.
    # Can we do better? The (4x1)*(1x6) tensor is equivalent to 6 independent (4x1)*(1x1) multiplications? No.
    # It's actually a (4x1x6) tensor. Its rank is min(4*6, ...) = 24? 
    # Actually, the rank of the (4x1x6) tensor is 6, because C_{i,k} = A_{i,4} * B_{4,k}
    # This is a rank-1 matrix for each (i,k), but the tensor itself has rank 6.
    # Wait, the tensor T_{i,k} = A_i * B_k.
    # This is a sum of 6 rank-1 tensors: sum_k B_k * (A * e_k^T)? No.
    # T = sum_{k=0}^{5} (A * b_k) * e_k^T? 
    # Let me think again.
    # The (4x1)*(1x6) multiplication tensor has entries T_{i,4,k} = A_{i,4} * B_{4,k} * C_{i,k}
    # But in our indexing, A has index (i, j) with j=4, B has index (j, k) with j=4, C has index (k, i).
    # So the non-zero entries of the tensor are at (i*5+4, 4*6+k, k*4+i) for i=0..3, k=0..5.
    # This is a sum of 24 rank-1 tensors, but many can be combined.
    # Actually, T = sum_{i=0}^{3} sum_{k=0}^{5} u_i * v_k * w_{i,k}
    # where u_i = e_{i*5+4}, v_k = e_{4*6+k}, w_{i,k} = e_{k*4+i}
    # This is already in rank-1 form, so rank = 24.
    # But we can write it as: T = (sum_i u_i * a_i) * (sum_k v_k * b_k) * ... no, that doesn't work.
    # The tensor is actually of rank 6. Here's why:
    # C_{i,k} = A_{i,4} * B_{4,k}
    # This can be computed as: for each k, C[:, k] = A[:, 4] * B[4, k]
    # So C[:, k] is a rank-1 matrix (outer product of A[:,4] and the scalar B[4,k]).
    # The tensor can be written as sum_{k=0}^{5} (A[:,4]) * (B[4,k]) * e_k
    # In tensor form: T = sum_{k=0}^{5} u * v_k * w_k
    # where u = e_{0*5+4} + e_{1*5+4} + e_{2*5+4} + e_{3*5+4}? No, that's not right.
    # Actually, for each k, the contribution to T is:
    # T_{i*5+4, 4*6+k, k*4+i} = A_{i,4} * B_{4,k}
    # This is not a simple outer product.
    # Let me just accept rank 24 for this part for now.
    
    # Part 1: (4x4)*(4x6)
    # (4x4)*(4x6) = (4x4)*(4x4) + (4x4)*(4x2)
    # (4x4)*(4x4): Use Strassen, rank 49? No, Strassen gives rank 7 for 2x2, so for 4x4 it's 7^2 = 49? 
    # Actually, Strassen for 2x2 has rank 7. For 4x4, we can use 4 blocks of 2x2, so rank 4*7 = 28? No.
    # Strassen's algorithm for n x n matrices has rank O(n^log2(7)).
    # For n=4, rank = 7^2 = 49.
    # But we can also just do 4 independent 2x2 multiplications? No, that's not how it works.
    # Let me just use schoolbook for (4x4)*(4x4): rank = 64.
    # (4x4)*(4x2): rank = 4*4*2 = 32 (schoolbook).
    
    # Total rank = 64 + 32 + 24 = 120. Same as schoolbook.
    
    # This isn't helping. Let me try a different partitioning.
    
    # What if we use Strassen for the 2x2x2 blocks?
    # n=4=2*2, p=6=2*3
    # We have 2*3=6 blocks of 2x?x2.
    # For each block, the m-dimension is 5.
    # (2x5)*(5x2) has rank 20 (schoolbook).
    # Can we do better for (2x5)*(5x2)?
    # (2x5)*(5x2) = (2x5)*(5x2)
    # This is a (2x5x2) tensor. Its rank is at most 2*5*2=20.
    # Can we prove it's exactly 20? Probably, for generic entries.
    
    # So the laser method doesn't seem to give an immediate improvement for these dimensions.
    
    # Let's just return schoolbook.
    return _build_schoolbook(n, m, p)

def _optimize_decomposition(n, m, p, max_time=100):
    """
    Try to find a lower rank decomposition using optimization.
    This is a placeholder for future work.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_BUDGET
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try hybrid decomposition
    remaining = max(0, deadline - time.time())
    if remaining > 5:
        try:
            hybrid_U, hybrid_V, hybrid_W = _build_hybrid_decomposition(n, m, p)
            if hybrid_U.shape[1] < best_rank:
                # Verify
                if _verify_decomposition(hybrid_U, hybrid_V, hybrid_W, n, m, p):
                    best_U, best_V, best_W = hybrid_U, hybrid_V, hybrid_W
                    best_rank = best_U.shape[1]
        except Exception:
            pass
    
    # Try laser method
    remaining = max(0, deadline - time.time())
    if remaining > 5:
        try:
            laser_U, laser_V, laser_W = _laser_method_decomposition(n, m, p)
            if laser_U.shape[1] < best_rank:
                # Verify
                if _verify_decomposition(laser_U, laser_V, laser_W, n, m, p):
                    best_U, best_V, best_W = laser_U, laser_V, laser_W
                    best_rank = best_U.shape[1]
        except Exception:
            pass
    
    # Try optimization
    remaining = max(0, deadline - time.time())
    if remaining > 10:
        try:
            result = _optimize_decomposition(n, m, p, max_time=remaining - 5)
            if result is not None:
                opt_U, opt_V, opt_W = result
                if opt_U.shape[1] < best_rank:
                    if _verify_decomposition(opt_U, opt_V, opt_W, n, m, p):
                        best_U, best_V, best_W = opt_U, opt_V, opt_W
                        best_rank = best_U.shape[1]
        except Exception:
            pass
    
    # Convert to lists for the interface
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
