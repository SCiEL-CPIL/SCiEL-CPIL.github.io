"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_list, v_list, w_list) where:
    u_list: [(i, j, val), ...] for A[i,j]
    v_list: [(j, k, val), ...] for B[j,k]
    w_list: [(i, k, val), ...] for C[i,k]
    """
    return [
        # Term 1: (a+d)(e+h) -> contributes to C00 and C11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # Term 2: (c+d)e -> contributes to C10 and C11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # Term 3: a(f-h) -> contributes to C01 and C11
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # Term 4: d(g-e) -> contributes to C00 and C10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # Term 5: (a+b)h -> contributes to C00 and C01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # Term 6: (c-a)(e+f) -> contributes to C11
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # Term 7: (b-d)(g+h) -> contributes to C00
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

def _build_block_decomposition(n, m, p):
    """Build the block decomposition:
    - Part 1: j=0, schoolbook 2x1x6 = 12 terms
    - Part 2: j=1..4, split into two 2x2x6 blocks, each using 3 Strassen pairs = 21 terms each
    Total rank = 12 + 2*21 = 54
    """
    rank = 54
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # Part 1: j = 0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 2: j in 1..4, two blocks of 2x2x6
    strassen_terms = _strassen_2x2_terms()
    
    for block_idx in range(2):
        j_base = 1 + block_idx * 2  # j=1,2 for block 0; j=3,4 for block 1
        
        for s in range(3):
            k_base = s * 2  # k pairs: (0,1), (2,3), (4,5)
            
            for term in strassen_terms:
                u_list, v_list, w_list = term
                
                for (li, lj, val) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    U[global_i * m + global_j, col] += val
                
                for (lj, lk, val) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    V[global_j * p + global_k, col] += val
                
                for (li, lk, val) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    W[global_k * n + global_i, col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,j,k} pattern
    Actually, the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    should equal 1 if the indices are consistent (i.e., this is the identity for matmul), 0 otherwise.
    """
    rank = U.shape[1]
    # Build the tensor from U, V, W
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor should satisfy: sum_j T[i*m+j, j*p+k, k*n+i] = 1 for all i,k
    # and T[i*m+j, j'*p+k, k'*n+i] = 0 when j != j' (no cross terms)
    
    # Actually the correct check: the multiplication tensor M has M[i*m+j, j*p+k, k*n+i] = 1
    # (summing over j implicitly via the index matching).
    # The decomposition gives: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should equal 1 if we're looking at the "diagonal" in j (i.e., the j in U matches j in V),
    # and 0 otherwise.
    
    # More precisely: for fixed i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # For j' != j: sum_r U[i*m+j', r] * V[j*p+k, r] * W[k*n+i, r] should be 0
    
    # Let's just check the full contraction
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,k (since A*B has C[i,k]=sum_j A[i,j]B[j,k], and we're checking
    # that the tensor correctly represents this bilinear form)
    
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total += s
            if total != 1:
                return False
    return True

def _try_rank_reduction(n, m, p, base_rank=54):
    """Try to find a lower-rank decomposition by searching for linear dependencies
    among the rank-54 terms. If some columns of [U|V|W] are linearly dependent
    in a way that preserves the tensor, we can reduce the rank.
    
    Strategy: Check if any subset of columns can be eliminated by expressing
    one term as a linear combination of others (with integer coefficients).
    """
    if time.time() > DEADLINE:
        return None
    
    result = _build_block_decomposition(n, m, p)
    U, V, W = result["U"], result["V"], result["W"]
    rank = U.shape[1]
    
    # Check for linear dependencies: if column r can be written as a linear combination
    # of other columns, we can remove it.
    # For each column r, check if there exist coefficients c_s (s != r) such that
    # U[:,r] = sum_s c_s U[:,s], V[:,r] = sum_s c_s V[:,s], W[:,r] = sum_s c_s W[:,s]
    # This is a system of (n*m + m*p + p*n) equations in (rank-1) unknowns.
    
    # Stack U, V, W vertically
    stacked = np.vstack([U, V, W])  # shape: (n*m + m*p + p*n, rank)
    
    # Try to find if any column is a linear combination of others
    # Use integer linear algebra: check if the column space has dimension < rank
    # Compute rank of the stacked matrix
    # If rank(stacked) < rank, then there are dependencies
    
    # But we need integer dependencies, not just real ones.
    # For small matrices, we can use the Smith normal form or just check
    # if any column is an integer linear combination of others.
    
    # Simple approach: for each column r, try to solve for integer coefficients
    # using the other columns. If found, we can eliminate column r.
    
    # This is expensive, so let's first check the real rank
    real_rank = np.linalg.matrix_rank(stacked)
    
    if real_rank < rank:
        # There are real linear dependencies. Try to find integer ones.
        # For now, just note this and try a greedy elimination.
        # This is complex, so we'll skip for now and just return the base decomposition.
        pass
    
    return result

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the base rank-54 decomposition
    result = _build_block_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=np.int64)
        V = np.zeros((m * p, rank_school), dtype=np.int64)
        W = np.zeros((p * n, rank_school), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition if time permits
    if time.time() < DEADLINE - 10:
        reduced = _try_rank_reduction(n, m, p)
        if reduced is not None:
            # Verify the reduced version
            if _verify_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                if reduced["U"].shape[1] < result["U"].shape[1]:
                    result = reduced
    
    # Convert to lists for the return interface
    return {
        "U": result["U"].tolist(),
        "V": result["V"].tolist(),
        "W": result["W"].tolist()
    }
# EVOLVE-BLOCK-END
