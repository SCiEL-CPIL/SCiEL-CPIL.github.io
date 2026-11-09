"""Rewritten program for matrix multiplication 3x4x4 using block decomposition."""

import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x4x4 matrix multiplication.
    
    Strategy: Decompose the problem into 3 independent 1x4x4 matrix multiplications.
    Each 1x4x4 multiplication has rank 16.
    Total rank = 3 * 16 = 48.
    
    This matches the schoolbook rank but uses a structured block construction 
    as per the directive to explore Kronecker-like block structures.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # U shape: (n*m, rank) = (12, 48)
    # V shape: (m*p, rank) = (16, 48)
    # W shape: (p*n, rank) = (12, 48)
    
    total_rank = n * m * p  # 48
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # We construct the decomposition as a sum of 'n' blocks.
    # Block i corresponds to the i-th row of A (dimension 1xm) multiplying B (mxp).
    # This produces the i-th row of C (dimension 1xp).
    
    current_col = 0
    for i in range(n):
        if time.time() > deadline:
            break
            
        # For each row i, we decompose the 1x4x4 multiplication.
        # The tensor for 1x4x4 is:
        # T_{j,k} = 1 for j in 0..m-1, k in 0..p-1
        # Factors:
        # U_part[j] corresponds to A_{i,j}
        # V_part[j,k] corresponds to B_{j,k}
        # W_part[k] corresponds to C_{i,k}
        
        # In the global indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        for j in range(m):
            if time.time() > deadline:
                break
            for k in range(p):
                if time.time() > deadline:
                    break
                    
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][current_col] = 1.0
                V[v_idx][current_col] = 1.0
                W[w_idx][current_col] = 1.0
                
                current_col += 1
                
    # Verify decomposition quickly to ensure no indexing errors
    # Note: Full verification of rank 48 is fast (48 * 48 mults)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to direct loop construction if verification fails (should not happen)
        U_fb, V_fb, W_fb = _schoolbook_direct(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}


def _schoolbook_direct(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly computes matrix multiplication.
    Checks that sum_r U[u,r] V[v,r] W[w,r] == 1 for all valid (u,v,w) 
    corresponding to (i,j,k) and 0 otherwise.
    For matrix multiplication, the tensor is 1 for all (i,j,k) in the domain?
    No, the tensor T is defined such that the bilinear form computes the product.
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] is part of the sum for C[i,k].
    In standard multiplication, every (i,j,k) triple contributes exactly once.
    So we check that for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    And we should also check orthogonality for invalid indices?
    The indices in U, V, W are fixed by the dimensions.
    U has size n*m. V has size m*p. W has size p*n.
    The "invalid" triples don't exist in the domain of the tensor definition 
    because the tensor is defined on the full grid [n]x[m]x[p].
    The condition is simply that the reconstruction of the multiplication tensor is exact.
    """
    rank = len(U[0])
    
    # Check main terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - 1.0) > 1e-9:
                    return False
                    
    # Check that terms for "wrong" combinations are 0.
    # A "wrong" combination would be if we picked an index from U, V, W 
    # that doesn't form a valid (i,j,k) path?
    # But the indices are just integers. 
    # The tensor is defined on the cartesian product of the index sets.
    # If we pick u_idx corresponding to (i1, j1), v_idx to (j2, k1), w_idx to (k2, i2).
    # For the term to be non-zero in the sum for a specific output, we need consistency.
    # Actually, the decomposition defines the tensor T[u,v,w] = sum_r U[u,r]V[v,r]W[w,r].
    # We need T[u,v,w] to be 1 if (u,v,w) corresponds to a valid multiplication term, 
    # and 0 otherwise.
    # Valid term: u=(i,j), v=(j,k), w=(k,i). Note the j must match between u and v, 
    # and k must match between v and w, and i must match between w and u.
    # So, for ANY u, v, w in the ranges:
    # Let u = i*m + j
    # Let v = j'*p + k
    # Let w = k'*n + i'
    # We require sum_r ... to be 1 if j==j', k==k', i==i' (and implicitly i'==i, etc from indices)
    # Specifically:
    # u encodes (i,j)
    # v encodes (j',k)
    # w encodes (k',i')
    # The term is "valid" if j == j' AND k == k' AND i == i'.
    # If any of these fail, the sum must be 0.
    
    for u_idx in range(len(U)):
        i = u_idx // m
        j = u_idx % m
        
        for v_idx in range(len(V)):
            j_prime = v_idx // p
            k = v_idx % p
            
            if j != j_prime:
                # Must be 0
                # We can skip checking all W if we trust the structure, 
                # but for safety we check a few or all.
                # Since rank is 48 and dimensions are small, checking all is feasible.
                pass # We'll do a comprehensive check below for performance if needed, 
                     # but here we rely on the block structure which ensures zeros 
                     # because the supports in U, V, W for different blocks/indices are orthogonal 
                     # in the rank dimension? No, the rank dimension is shared.
                     # However, in schoolbook, each column r is associated with a unique (i,j,k).
                     # U[i*m+j, r] is non-zero only if r corresponds to that (i,j,k).
                     # So if we pick a different j', the U row for (i,j) and V row for (j',k) 
                     # will have non-zeros in disjoint columns of r?
                     # Yes, in schoolbook, column r is set to 1 for specific u,v,w.
                     # For a given r, let it correspond to (i_r, j_r, k_r).
                     # U[u, r] is 1 only if u == i_r*m+j_r.
                     # V[v, r] is 1 only if v == j_r*p+k_r.
                     # W[w, r] is 1 only if w == k_r*n+i_r.
                     # If we pick u, v such that j != j_prime, can we have a non-zero product?
                     # If U[u,r]!=0 and V[v,r]!=0, then j must be j_r and j_prime must be j_r. 
                     # So j == j_prime. Contradiction.
                     # Thus, for j != j_prime, sum is 0.
                     
            for w_idx in range(len(W)):
                k_prime = w_idx // n
                i_prime = w_idx % n
                
                # Check if (j==j_prime) and (k==k_prime) and (i==i_prime)
                if (j == j_prime) and (k == k_prime) and (i == i_prime):
                    expected = 1.0
                else:
                    expected = 0.0
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - expected) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
