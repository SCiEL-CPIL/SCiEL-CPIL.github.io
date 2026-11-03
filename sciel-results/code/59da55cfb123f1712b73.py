"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _build_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Construct the multiplication tensor check efficiently
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # We want T to match the standard multiplication tensor.
    # Standard tensor: T_std[i, j, k] = 1 if i==j and j==k? No.
    # Matrix mult: C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor form: T_{i, j, k} such that C_{i, k} = sum_{j, r} A_{i, j} B_{j, k} W_{k, r} U_{i, r} V_{j, r}?
    # Wait, the interface is:
    # U shape (n*m, rank) -> indices (i, j)
    # V shape (m*p, rank) -> indices (j, k)
    # W shape (p*n, rank) -> indices (k, i)
    # Result C[i, k] = sum_{j, r} U[i, j, r] * A[i, j] * V[j, k, r] * B[j, k] * W[k, i, r]?
    # No, the standard tensor decomposition for matrix mult is:
    # C[i, k] = sum_{j, r} U_{i, j, r} V_{j, k, r} W_{k, i, r} A_{i, j} B_{j, k}
    # This requires U, V, W to be 3-tensors.
    # Here they are matrices.
    # The contract says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "exactly decompose the matrix multiplication tensor".
    # Usually, this means:
    # sum_{r} U[i, j, r] V[j, k, r] W[k, i, r] = 1 if (i, j, k) is a valid path?
    # Actually, the standard way to view matrix multiplication as a tensor is:
    # T(i, j, k) = 1 if i==j and j==k is not right.
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So T is a 3-index tensor T[i, j, k] where T[i, j, k] = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # Since C[i,k] = sum_j A[i,j]B[j,k], the tensor is T[i, j, k] = 1 for all i, j, k?
    # No, the indices are fixed. For a given (i, k), we sum over j.
    # The tensor decomposition usually refers to the bilinear complexity.
    # C[i, k] = sum_{j, r} U_{i, j, r} V_{j, k, r} W_{k, i, r} A_{i, j} B_{j, k}
    # This is not quite right either. The standard Strassen form is:
    # M_r = sum_{i,j} u_{i,j,r} A_{i,j}
    # N_r = sum_{j,k} v_{j,k,r} B_{j,k}
    # C_{i,k} = sum_r w_{i,k,r} M_r N_r
    # Here W is indexed by (k, i) in the prompt? "W[k*n+i, r]".
    # So W_{k, i, r} = W[k*n+i, r].
    # Let's check the identity:
    # sum_r W[k, i, r] * (sum_{j'} U[i, j', r] A[i, j']) * (sum_{k'} V[j', k', r] B[j', k'])
    # This doesn't look like it contracts to C[i, k] easily unless U, V, W are structured.
    
    # Let's look at the schoolbook example in the current program:
    # U[i*m + j][column] = 1
    # V[j*p + k][column] = 1
    # W[k*n + i][column] = 1
    # Column index corresponds to (i, j, k).
    # Term: W[k, i, (i,j,k)] * U[i, j, (i,j,k)] * V[j, k, (i,j,k)] * A[i, j] * B[j, k]
    # = 1 * 1 * 1 * A[i, j] * B[j, k]
    # Sum over all columns (i', j', k'):
    # Sum_{i', j', k'} W[k, i, (i', j', k')] U[i, j, (i', j', k')] V[j, k, (i', j', k')] A[i, j] B[j, k]
    # W[k, i, (i', j', k')] is 1 only if k'=k and i'=i.
    # U[i, j, (i', j', k')] is 1 only if i'=i and j'=j.
    # V[j, k, (i', j', k')] is 1 only if j'=j and k'=k.
    # So the term is non-zero only if (i', j', k') = (i, j, k).
    # The sum becomes A[i, j] B[j, k].
    # This is correct for C[i, k] = sum_j A[i, j] B[j, k].
    
    # So the condition is:
    # For all i, j, k:
    # sum_{r} U[i, j, r] V[j, k, r] W[k, i, r] = 1
    # And for all i, j, k' where k' != k? No, the indices are separate.
    # The tensor equation is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Where T[i, j, k] = 1 if the indices (i, j, k) form a valid "path" in the multiplication?
    # Actually, in the schoolbook case, T[i, j, k] = 1 for all i, j, k?
    # Let's check:
    # C[i, k] = sum_j (sum_r U[i, j, r] V[j, k, r] W[k, i, r] A[i, j] B[j, k])
    # If T[i, j, k] = 1 for all i, j, k, then C[i, k] = sum_j A[i, j] B[j, k].
    # So the target tensor T is simply all 1s?
    # Wait. If T[i, j, k] = 1 for all i, j, k, then:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    # This is a system of n*m*p equations.
    
    # Let's verify this interpretation.
    # If we use Strassen, T is NOT all 1s.
    # In Strassen, we compute intermediate products M_r.
    # C[i, k] = sum_r W_{i, k, r} M_r N_r.
    # M_r = sum_{i', j'} U_{i', j', r} A_{i', j'}
    # N_r = sum_{j', k'} V_{j', k', r} B_{j', k'}
    # C[i, k] = sum_{i', j', j', k', r} W_{i, k, r} U_{i', j', r} V_{j', k', r} A_{i', j'} B_{j', k'}
    # We want this to equal sum_j A_{i, j} B_{j, k}.
    # So the coefficient of A_{i', j'} B_{j', k'} in the expansion must be:
    # 1 if i'=i, j'=j, j'=k' (wait, j' is the inner index).
    # The term is A_{i', j'} B_{j', k'}.
    # We want the sum to be A_{i, j} B_{j, k}.
    # So we need the coefficient of A_{i', j'} B_{j', k'} to be:
    # 1 if i'=i, j'=j, and j'=k' is not required?
    # Actually, C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i', j'] B[j', k'] appears in C[i, k] with coefficient 1 if i'=i and j'=k' is false?
    # No, A[i', j'] B[j', k'] contributes to C[i', k']? No, it contributes to C[i', k'] if we sum over j'?
    # No, C[i, k] depends on A[i, *] and B[*, k].
    # The term A[i', j'] B[j', k'] contributes to C[i', k']?
    # Let's check indices.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So A[i', j'] B[j', k'] contributes to C[i', k'] ONLY IF j' is the summed index?
    # No, j' is a specific index.
    # A[i', j'] B[j', k'] is a specific product.
    # It contributes to C[i', k']?
    # In the sum sum_j A[i, j] B[j, k], the term is A[i, j] B[j, k].
    # So A[i', j'] B[j', k'] contributes to C[i', k'] if and only if i'=i and k'=k?
    # And j' is the specific j.
    # So yes, A[i', j'] B[j', k'] contributes to C[i', k'] with weight 1.
    # Wait, does it contribute to any other C[i, k]?
    # C[i, k] only contains terms A[i, j] B[j, k].
    # So A[i', j'] B[j', k'] only appears in C[i', k'].
    # Therefore, the tensor T[i', j', k'] (coefficient of A[i', j'] B[j', k'] in C[i', k']? No, in C[i, k]?)
    # Let's re-read the decomposition identity.
    # C[i, k] = sum_r W[k, i, r] (sum_{i', j'} U[i', j', r] A[i', j']) (sum_{j', k'} V[j', k', r] B[j', k'])
    # C[i, k] = sum_{i', j', j', k', r} W[k, i, r] U[i', j', r] V[j', k', r] A[i', j'] B[j', k']
    # We want C[i, k] = sum_j A[i, j] B[j, k].
    # So the coefficient of A[i', j'] B[j', k'] in C[i, k] must be:
    # 1 if i'=i and j'=k' is NOT a condition.
    # The term A[i', j'] B[j', k'] is part of the sum for C[i, k] if:
    # i' = i (from A index)
    # k' = k (from B index)
    # AND j' is the shared index?
    # In the standard multiplication, the shared index is j.
    # So A[i, j] B[j, k].
    # So A[i', j'] B[j', k'] contributes to C[i, k] if i'=i, k'=k, and j'=j?
    # But j is summed over.
    # So for a fixed (i, k), we sum over j.
    # The term A[i, j] B[j, k] is included.
    # So A[i', j'] B[j', k'] contributes to C[i, k] if i'=i, k'=k, and j'=j?
    # No, j' is a specific value.
    # So A[i', j'] B[j', k'] contributes to C[i', k']?
    # Let's check: C[i', k'] = sum_j A[i', j] B[j, k'].
    # The term with j=j' is A[i', j'] B[j', k'].
    # So A[i', j'] B[j', k'] contributes to C[i', k'] with weight 1.
    # Does it contribute to any other C[i, k]?
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i', j'] B[j', k'] appears if i=i', k=k'.
    # So yes, it ONLY contributes to C[i', k'].
    
    # Therefore, the condition is:
    # For all i, j, k:
    # Coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # Coefficient of A[i, j] B[j, k] in C[i', k'] (where (i', k') != (i, k)) is 0.
    
    # In the formula:
    # Coeff of A[i', j'] B[j', k'] in C[i, k] is:
    # sum_r W[k, i, r] U[i', j', r] V[j', k', r]
    
    # We require:
    # sum_r W[k, i, r] U[i', j', r] V[j', k', r] = 1 if i'=i and k'=k
    # sum_r W[k, i, r] U[i', j', r] V[j', k', r] = 0 otherwise
    
    # Note that the indices i', j', k' are independent in the sum?
    # No, U depends on i', j'. V depends on j', k'.
    # So for a fixed (i, k), we have a matrix M_{i, k} of size (m * (m*p))? No.
    # The variables are U, V, W.
    # This is a system of quadratic equations (bilinear in U,V,W).
    # However, if we fix U and V, it is linear in W.
    # Or if we fix two, it is linear in the third.
    
    # Strategy:
    # 1. Initialize U and V randomly (or with a known decomposition).
    # 2. Solve for W using linear equations.
    # 3. Check if W is valid (rank constraint is implicit in the number of columns).
    # 4. If we want to reduce rank, we can try to find a decomposition with fewer columns.
    #    This is equivalent to finding a rank-r decomposition of the tensor.
    #    This is generally hard.
    
    # Alternative Strategy:
    # Use the known Strassen decomposition as a starting point (rank 77).
    # Try to reduce the rank by merging columns or finding a null space.
    # Or, use a "tensor rank" minimization approach.
    
    # Given the time limit, let's try a simple local search or a specific structured search.
    # The prompt suggests "rank-minimal exact tensor decomposition".
    # For 4x4x4, the best known is 49 (Strassen).
    # For 4x4x6, we can view it as 4x4x4 + 4x4x2.
    # Rank(4x4x4) = 49.
    # Rank(4x4x2) = ?
    # 4x4x2 is equivalent to two 4x4x1 multiplications.
    # Rank(4x4x1) is the rank of the matrix multiplication tensor for a column.
    # This is just the rank of the identity? No.
    # C[i] = sum_j A[i, j] B[j].
    # This is a linear map. The "tensor" is just the identity map on the column?
    # No, B is a column vector.
    # C = A B.
    # This is bilinear in A and B.
    # The rank of this tensor is the rank of the matrix representing the bilinear form?
    # Actually, for fixed A, C is linear in B.
    # For fixed B, C is linear in A.
    # The tensor is T[i, j, k] where k is the index of B?
    # B has indices j (row) and k (col, which is 0 since it's 1 col).
    # So B[j, 0].
    # C[i, 0] = sum_j A[i, j] B[j, 0].
    # The tensor T[i, j, 0] = 1.
    # The rank of this tensor is the rank of the matrix T[i, j] which is 1?
    # No, the rank of a 3-tensor T[i, j, k] is the min r such that T[i, j, k] = sum_{r=1}^R U[i, r] V[j, r] W[k, r].
    # Here k is always 0.
    # T[i, j, 0] = 1.
    # So we need 1 = sum_r U[i, r] V[j, r] W[0, r].
    # Let W[0, r] = 1.
    # Then sum_r U[i, r] V[j, r] = 1.
    # This means the matrix M = U V^T must be the all-ones matrix.
    # The rank of the all-ones matrix is 1.
    # So Rank(4x4x1) = 1?
    # Let's check.
    # U[i, 1] = 1. V[j, 1] = 1. W[0, 1] = 1.
    # Sum = 1*1*1 = 1.
    # Yes, rank is 1.
    
    # So Rank(4x4x2) = 2 * Rank(4x4x1) = 2?
    # Let's check if we can share the rank-1 components.
    # C[:, 0] = A B[:, 0].
    # C[:, 1] = A B[:, 1].
    # We can compute M = A * (B[:, 0] + B[:, 1]) and N = A * (B[:, 0] - B[:, 1])?
    # No, A is the same.
    # M1 = A * (B[:, 0] + B[:, 1])
    # M2 = A * (B[:, 0] - B[:, 1])
    # C[:, 0] = (M1 + M2) / 2
    # C[:, 1] = (M1 - M2) / 2
    # This uses 2 multiplications.
    # So Rank(4x4x2) = 2.
    
    # Total Rank = Rank(4x4x4) + Rank(4x4x2) = 49 + 2 = 51?
    # Wait, can we share the "A" part?
    # In Strassen, we compute 7 intermediate products M_r.
    # For the 4x4x4 part, we use 7 products.
    # For the 4x4x2 part, we use 2 products.
    # Total 9 products?
    # The products are of the form (linear combo of A) * (linear combo of B).
    # For the 4x4x4 part, the B part involves all 4 columns of B (first 4).
    # For the 4x4x2 part, the B part involves the last 2 columns of B.
    # The A part is the same matrix A.
    # Can we combine the A linear combinations?
    # Strassen for 4x4x4 uses 7 combinations of A blocks.
    # The 4x4x2 part uses 2 combinations of A (which are just A itself, effectively, or linear combos).
    # If the A combinations for the 4x4x2 part are already computed in the Strassen step, we save rank.
    # In Strassen 4x4x4, we compute:
    # M1 = (A00+A11)(B00+B11)
    # ...
    # The A combinations are sums of blocks of A.
    # For the 4x4x2 part, we need A * B_right.
    # This is just 2 matrix-vector multiplications.
    # If we can express A * B_right using the existing M_r's?
    # Probably not directly, because M_r's are products with B_left.
    # However, we can just add the 2 new products.
    # So rank 49 + 2 = 51.
    
    # Let's verify this construction.
    
def _build_strassen_4x4_4x4():
    """Build rank-49 Strassen decomposition for 4x4 * 4x4."""
    # Reuse the logic from the current program, but return arrays
    factors = _strassen_2x2_factors()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            l1_factor = factors[l1]
            l2_factor = factors[l2]
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) in l1_factor['U']:
                        c1 = l1_factor['U'][(bi, bj)]
                        for si in range(2):
                            for sj in range(2):
                                if (si, sj) in l2_factor['U']:
                                    c2 = l2_factor['U'][(si, sj)]
                                    ai = bi * 2 + si
                                    aj = bj * 2 + sj
                                    U[ai * 4 + aj, col] += c1 * c2
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) in l1_factor['V']:
                        c1 = l1_factor['V'][(bi, bj)]
                        for si in range(2):
                            for sj in range(2):
                                if (si, sj) in l2_factor['V']:
                                    c2 = l2_factor['V'][(si, sj)]
                                    bri = bi * 2 + si
                                    bcj = bj * 2 + sj
                                    V[bri * 4 + bcj, col] += c1 * c2
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) in l1_factor['W']:
                        c1 = l1_factor['W'][(bi, bj)]
                        for si in range(2):
                            for sj in range(2):
                                if (si, sj) in l2_factor['W']:
                                    c2 = l2_factor['W'][(si, sj)]
                                    cri = bi * 2 + si
                                    ccj = bj * 2 + sj
                                    W[ccj * 4 + cri, col] += c1 * c2
            
            col += 1
    
    return U, V, W

def _build_strassen_2x2_factors():
    """Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication."""
    factors = []
    factors.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    factors.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}})
    factors.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,1): 1, (1,1): 1}})
    factors.append({'U': {(1,1): 1}, 'V': {(1,1): 1, (0,1): -1}, 'W': {(0,0): 1, (1,0): 1}})
    factors.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): -1}})
    factors.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    factors.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    return factors

def _build_hybrid_4x4_4x6():
    """
    Build a hybrid decomposition:
    1. Strassen 4x4x4 for the first 4 columns of B. (Rank 49)
    2. Direct multiplication for the last 2 columns of B. (Rank 2)
    Total Rank: 51.
    """
    n, m, p = 4, 4, 6
    rank_s = 49
    rank_d = 2
    rank_total = rank_s + rank_d
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Part 1: Strassen 4x4x4
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U[:, :rank_s] = U_s
    V[:, :rank_s] = V_s
    W[:, :rank_s] = W_s
    
    # Part 2: Direct for 4x4 * 4x2
    # We need to compute C[:, 4] = A * B[:, 4] and C[:, 5] = A * B[:, 5].
    # This can be done with 2 scalar multiplications? No, matrix-vector.
    # The tensor for 4x4x1 has rank 1.
    # So we need 2 rank-1 terms.
    
    # Term 1: C[:, 4] = A * B[:, 4]
    # U1[i, j] = A[i, j] -> U1[i*m+j, r1] = 1
    # V1[j, 4] = B[j, 4] -> V1[j*p+4, r1] = 1
    # W1[4, i] = C[i, 4] -> W1[4*n+i, r1] = 1
    # Check: sum_r W1[4, i, r1] U1[i, j, r1] V1[j, 4, r1] A[i, j] B[j, 4]
    # = 1 * 1 * 1 * A[i, j] B[j, 4]
    # Sum over j: sum_j A[i, j] B[j, 4] = C[i, 4]. Correct.
    
    # Term 2: C[:, 5] = A * B[:, 5]
    # U2[i, j] = 1
    # V2[j, 5] = 1
    # W2[5, i] = 1
    
    r1 = rank_s
    r2 = rank_s + 1
    
    for i in range(n):
        for j in range(m):
            U[i*m + j, r1] = 1
            U[i*m + j, r2] = 1
            
            V[j*p + 4, r1] = 1
            V[j*p + 5, r2] = 1
            
            W[4*n + i, r1] = 1
            W[5*n + i, r2] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p
    
    # Try Hybrid Strassen + Direct
    hybrid = _build_hybrid_4x4_4x6()
    if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
        rank = len(hybrid["U"][0])
        if rank < best_rank:
            best = hybrid
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
