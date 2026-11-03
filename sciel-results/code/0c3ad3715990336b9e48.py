"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact using vectorized operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    """
    rank = U.shape[1]
    # Build index arrays for vectorized computation
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Extract rows: (n*m*p, rank)
    U_sel = U[u_idx, :]
    V_sel = V[v_idx, :]
    W_sel = W[w_idx, :]
    
    # Compute sum over rank dimension: (n*m*p,)
    T_flat = np.sum(U_sel * V_sel * W_sel, axis=1)
    T = T_flat.reshape(n, m, p)
    
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _strassen_2x2_products():
    """
    Return the 7 Strassen products for 2x2x2 matrix multiplication.
    Each product is a tuple of (A_indices, B_indices, C_indices, sign)
    where A_indices are (row, col) in A, B_indices are (row, col) in B,
    C_indices are (row, col) in C, and sign is +1 or -1.
    
    Strassen's 7 products:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B10-B11)
    P4 = A11*(B01-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B10)
    P7 = (A01-A11)*(B01+B11)
    
    C00 = P1 + P4 - P5 - P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P7
    """
    # Each entry: list of (a_row, a_col, b_row, b_col, c_row, c_col, coeff)
    # We'll represent each Strassen product as a combination of schoolbook terms
    # that we need to track for the decomposition.
    
    # The 7 Strassen products expressed in terms of A and B elements:
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #     = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # P2 = (A[1,0]+A[1,1])*B[0,0]
    #     = A[1,0]*B[0,0] + A[1,1]*B[0,0]
    # P3 = A[0,0]*(B[1,0]-B[1,1])
    #     = A[0,0]*B[1,0] - A[0,0]*B[1,1]
    # P4 = A[1,1]*(B[0,1]-B[0,0])
    #     = A[1,1]*B[0,1] - A[1,1]*B[0,0]
    # P5 = (A[0,0]+A[0,1])*B[1,1]
    #     = A[0,0]*B[1,1] + A[0,1]*B[1,1]
    # P6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
    #     = A[1,0]*B[0,0] + A[1,0]*B[1,0] - A[0,0]*B[0,0] - A[0,0]*B[1,0]
    # P7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    #     = A[0,1]*B[0,1] + A[0,1]*B[1,1] - A[1,1]*B[0,1] - A[1,1]*B[1,1]
    
    # C elements in terms of P's:
    # C[0,0] = P1 + P4 - P5 - P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P7
    
    # For the tensor decomposition, each Strassen product corresponds to one rank-1 term.
    # The U, V, W vectors for each product are determined by the linear combinations.
    
    # Product 1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U: A[0,0] has coeff 1, A[1,1] has coeff 1
    # V: B[0,0] has coeff 1, B[1,1] has coeff 1
    # W: C[0,0] has coeff 1, C[0,1] has coeff 0, C[1,0] has coeff 0, C[1,1] has coeff 1
    #     (from C[0,0]=P1+... and C[1,1]=P1+...)
    
    # Actually, let me think about this differently.
    # In the tensor decomposition, we have:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For a Strassen block at position (i0, j0, k0), the 7 products give 7 rank-1 terms
    # that together cover all 8 schoolbook terms for the 2x2x2 sub-problem.
    
    # Let me define the 7 products with their U, V, W contributions.
    # Local indices: a in {0,1}, b in {0,1}, c in {0,1}
    # Global: A row = i0+a, A col = j0+b, B row = j0+b, B col = k0+c
    #         C row = i0+a, C col = k0+c
    
    products = []
    
    # P1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # U: (i0*m+j0)=1, ((i0+1)*m+(j0+1))=1
    # V: (j0*p+k0)=1, ((j0+1)*p+(k0+1))=1
    # W: (k0*n+i0)=1, ((k0+1)*n+(i0+1))=1
    products.append({
        'U': {0: 1, 1: 1},  # local (a,b) -> coeff
        'V': {0: 1, 1: 1},
        'W': {0: 1, 1: 1},
    })
    
    # P2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    # U: (i0+1,j0)=1, (i0+1,j0+1)=1
    # V: (j0,k0)=1
    # W: (k0,i0+1)=1
    products.append({
        'U': {1: 1, 3: 1},  # (1,0) and (1,1) in local 2x2
        'V': {0: 1},
        'W': {2: 1},  # (k0+0, i0+1) -> local (0,1) in C
    })
    
    # P3 = A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    # U: (i0,j0)=1
    # V: (j0+1,k0)=1, (j0+1,k0+1)=-1
    # W: (k0,i0)=1, (k0+1,i0)=1
    products.append({
        'U': {0: 1},
        'V': {2: 1, 3: -1},
        'W': {0: 1, 1: 1},
    })
    
    # P4 = A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    # U: (i0+1,j0+1)=1
    # V: (j0,k0+1)=1, (j0,k0)=-1
    # W: (k0+1,i0+1)=1, (k0,i0+1)=1
    products.append({
        'U': {3: 1},
        'V': {1: 1, 0: -1},
        'W': {3: 1, 2: 1},
    })
    
    # P5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # U: (i0,j0)=1, (i0,j0+1)=1
    # V: (j0+1,k0+1)=1
    # W: (k0+1,i0)=1
    products.append({
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {1: 1},
    })
    
    # P6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0+1,k0])
    # U: (i0+1,j0)=1, (i0,j0)=-1
    # V: (j0,k0)=1, (j0+1,k0)=1
    # W: (k0,i0+1)=1
    products.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 2: 1},
        'W': {2: 1},
    })
    
    # P7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0,k0+1]+B[j0+1,k0+1])
    # U: (i0,j0+1)=1, (i0+1,j0+1)=-1
    # V: (j0,k0+1)=1, (j0+1,k0+1)=1
    # W: (k0+1,i0)=1, (k0+1,i0+1)=1
    products.append({
        'U': {1: 1, 3: -1},
        'V': {1: 1, 3: 1},
        'W': {1: 1, 3: 1},
    })
    
    return products


def _construct_with_strassen(n, m, p, strassen_blocks):
    """
    Construct a decomposition using specified Strassen blocks.
    strassen_blocks: list of (i0, j0, k0) tuples for 2x2x2 Strassen placements.
    
    For each Strassen block, we use 7 products instead of 8 schoolbook products.
    All other (i,j,k) combinations use schoolbook (1 product each).
    """
    rank = n * m * p - len(strassen_blocks)  # save 1 per Strassen block
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Mark which (i,j,k) are covered by Strassen blocks
    strassen_set = set()
    for (i0, j0, k0) in strassen_blocks:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    strassen_set.add((i0 + a, j0 + b, k0 + c))
    
    products = _strassen_2x2_products()
    
    col = 0
    
    # First, add Strassen products
    for (i0, j0, k0) in strassen_blocks:
        for prod in products:
            # U contributions
            for local_idx, coeff in prod['U'].items():
                a = local_idx // 2
                b = local_idx % 2
                U[(i0 + a) * m + (j0 + b), col] += coeff
            # V contributions
            for local_idx, coeff in prod['V'].items():
                b = local_idx // 2
                c = local_idx % 2
                V[(j0 + b) * p + (k0 + c), col] += coeff
            # W contributions
            for local_idx, coeff in prod['W'].items():
                c = local_idx // 2
                a = local_idx % 2
                W[(k0 + c) * n + (i0 + a), col] += coeff
            col += 1
    
    # Then, add schoolbook products for non-Strassen (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_set:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return U, V, W


def _blocks_overlap(block1, block2):
    """Check if two 2x2x2 Strassen blocks overlap.
    Block at (i0,j0,k0) covers rows i0,i0+1; cols j0,j0+1; k-indices k0,k0+1.
    Two blocks overlap if their covered ranges intersect in all 3 dimensions.
    """
    i1, j1, k1 = block1
    i2, j2, k2 = block2
    # Ranges: block1 covers [i1, i1+1], block2 covers [i2, i2+1]
    # They overlap in dimension if NOT (i1+1 < i2 or i2+1 < i1)
    i_overlap = not (i1 + 1 < i2 or i2 + 1 < i1)
    j_overlap = not (j1 + 1 < j2 or j2 + 1 < j1)
    k_overlap = not (k1 + 1 < k2 or k2 + 1 < k1)
    return i_overlap and j_overlap and k_overlap


def _all_valid_block_sets(n, m, p):
    """Find maximum set of non-overlapping Strassen blocks via greedy + backtracking."""
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Greedy: try to pack as many non-overlapping blocks as possible
    # For 3x3x8: i0 in {0,1}, j0 in {0,1}, k0 in {0,1,2,3,4,5,6}
    # Total possible blocks: 2*2*7 = 28
    
    # Use a simple greedy approach with multiple orderings
    best_set = []
    
    # Try different orderings to find good packings
    orderings = [
        lambda b: (b[0], b[1], b[2]),
        lambda b: (b[0], b[2], b[1]),
        lambda b: (b[2], b[0], b[1]),
        lambda b: (b[1], b[0], b[2]),
        lambda b: (-b[0], -b[1], -b[2]),
        lambda b: (b[0] + b[1] + b[2], b[0], b[1]),
    ]
    
    for order_fn in orderings:
        sorted_blocks = sorted(all_blocks, key=order_fn)
        current_set = []
        for block in sorted_blocks:
            if all(not _blocks_overlap(block, existing) for existing in current_set):
                current_set.append(block)
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Also try a more systematic approach: for small grids, enumerate
    # Maximum possible: floor(3/2)*floor(3/2)*floor(8/2) = 1*1*4 = 4
    # So we can have at most 4 non-overlapping 2x2x2 blocks
    
    # Try to find 4 blocks systematically
    # i0 must be 0 or 1 (but only one since they overlap in i)
    # j0 must be 0 or 1 (but only one since they overlap in j)
    # k0 can be 0,2,4,6 (non-overlapping in k dimension)
    
    # Actually for i: blocks at i0=0 cover {0,1}, i0=1 covers {1,2} - they overlap at row 1
    # So we can only pick one i0 value
    # Same for j: j0=0 covers {0,1}, j0=1 covers {1,2} - overlap at col 1
    # For k: k0=0 covers {0,1}, k0=2 covers {2,3}, k0=4 covers {4,5}, k0=6 covers {6,7}
    # These are all non-overlapping!
    
    # So maximum 4 blocks: fix i0, fix j0, use k0 in {0,2,4,6}
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            k_values = list(range(0, p - 1, 2))  # 0, 2, 4, 6 for p=8
            candidate = [(i0, j0, k) for k in k_values]
            # Verify no overlaps
            valid = True
            for a in range(len(candidate)):
                for b in range(a + 1, len(candidate)):
                    if _blocks_overlap(candidate[a], candidate[b]):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(candidate) > len(best_set):
                best_set = candidate
    
    return best_set


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as baseline
    U, V, W = _schoolbook(n, m, p)
    
    # Verify schoolbook
    assert _verify(U, V, W, n, m, p), "Schoolbook verification failed"
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Search for best Strassen block configuration
    if time.time() - start_time < DEADLINE - 10:
        best_blocks = _all_valid_block_sets(n, m, p)
        if best_blocks:
            U2, V2, W2 = _construct_with_strassen(n, m, p, best_blocks)
            if _verify(U2, V2, W2, n, m, p):
                if U2.shape[1] < best_rank:
                    best_rank = U2.shape[1]
                    best_U, best_V, best_W = U2, V2, W2
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
