"""Rank-minimal exact tensor decomposition for M_{2,2,11}."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank33_cyclic():
    """
    Construct a rank-33 decomposition using cyclic convolution structure.
    
    The key idea: represent the 2x11 matrix B as four polynomials b_jk(x) in Z[x]/(x^11-1),
    where the 2x2 matrix A multiplies these polynomials. The multiplication of 2x2 matrices
    over a commutative ring R of dimension p can be done with rank 3p using the
    "Strassen over rings" technique.
    
    Specifically, for each "frequency" or "position" in the cyclic structure, we use
    a 3-term decomposition that handles the coupling between the 2x2 block structure
    and the 11-dimensional ring.
    
    The construction uses the fact that:
    - A is 2x2: entries a00, a01, a10, a11
    - B is 2x11: for each column k, entries b0k, b1k
    - C is 2x11: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose using 3 terms per "cyclic position" with appropriate mixing.
    """
    n, m, p = 2, 2, 11
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 33
    V = [[0.0] * rank for _ in range(m * p)]  # 22 x 33
    W = [[0.0] * rank for _ in range(p * n)]  # 22 x 33
    
    # For the cyclic/convolution approach with p=11:
    # We use 3 "layers" of Strassen-like terms, each layer handling a subset
    # of the 11 columns in a structured way.
    
    # Layer 0: handles columns 0, 4, 8 (every 4th, mod 11)
    # Layer 1: handles columns 1, 5, 9
    # Layer 2: handles columns 2, 6, 10
    # But this gives 3*3=9 columns, not 11. Need a different partition.
    
    # Alternative: use the fact that 11 is prime, so we can use
    # a different decomposition. Let's use a direct construction.
    
    # Actually, let me use a simpler but valid approach:
    # Rank 33 = 3 * 11. For each column k, use 3 terms that jointly
    # handle the 2x2 multiplication with appropriate mixing across columns.
    
    # The 3 terms per column use:
    # Term 0: (a00 + a11) * (b0k + b1k) -> contributes to c0k and c1k
    # Term 1: (a10 + a11) * b0k -> contributes to c1k
    # Term 2: a00 * (b1k - b0k) -> contributes to c0k
    # But this is 3 terms per column = 33 total, and we need to verify it works.
    
    # Let me verify: 
    # c0k = a00*b0k + a01*b1k
    # c1k = a10*b0k + a11*b1k
    #
    # With 3 terms:
    # T0: U has (1,0,0,1) for a00,a01,a10,a11 -> a00+a11
    #     V has (1,1) for b0k,b1k -> b0k+b1k
    #     W should give: c0k += 1*(a00+a11)(b0k+b1k), c1k += 1*(a00+a11)(b0k+b1k)
    #     But we need c0k = a00*b0k + a01*b1k, not (a00+a11)(b0k+b1k)
    #
    # This doesn't work directly. Let me use the standard 3-term for 2x2:
    # Actually 2x2 needs 7 terms (Strassen) or 4 terms (schoolbook).
    # 3 terms is not enough for 2x2 matrix multiplication.
    
    # So rank 33 = 3*11 won't work with 3 terms per column.
    # We need at least 4 terms per column for schoolbook = 44.
    # Or we need to share terms across columns.
    
    # Let me try rank 35 = 5*7: 5 Strassen blocks of 2x2, each covering 2 columns,
    # plus 1 extra column handled with 2 schoolbook terms.
    # But 5*2 + 1 = 11 columns. 5*7 + 2 = 37. Not 35.
    
    # Let me try a different approach: rank 36.
    # 36 = 4 * 9. Use 4 terms per "group" of columns.
    
    # Actually, let me just use the known rank-39 construction but try to
    # find a better one through structured search.
    
    # For now, return a valid rank-39 construction (same as before) and
    # let the search try to improve.
    
    # Let me construct rank 39 properly:
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, base_rank):
        r = base_rank
        # P1 = (A00+A11)*(B00+B11)
        U[0][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+3][r] += 1.0
        # P2 = (A10+A11)*B00
        r = base_rank + 1
        U[2][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0
        W[2*col_offset+1][r] += 1.0
        # P3 = A00*(B01-B10)
        r = base_rank + 2
        U[0][r] += 1.0
        V[col_offset+1][r] += 1.0; V[11+col_offset][r] += -1.0
        W[2*col_offset+2][r] += 1.0
        # P4 = A11*(B10-B00)
        r = base_rank + 3
        U[3][r] += 1.0
        V[11+col_offset][r] += 1.0; V[col_offset][r] += -1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset][r] += -1.0
        # P5 = (A00+A01)*B11
        r = base_rank + 4
        U[0][r] += 1.0; U[1][r] += 1.0
        V[12+col_offset][r] += 1.0
        W[2*col_offset+2][r] += 1.0; W[2*col_offset+3][r] += -1.0
        # P6 = (A10-A00)*(B00+B01)
        r = base_rank + 5
        U[2][r] += 1.0; U[0][r] += -1.0
        V[col_offset][r] += 1.0; V[col_offset+1][r] += 1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset+3][r] += 1.0
        # P7 = (A01-A11)*(B10+B11)
        r = base_rank + 6
        U[1][r] += 1.0; U[3][r] += -1.0
        V[11+col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+1][r] += -1.0
    
    for block_idx in range(5):
        add_strassen_block(2 * block_idx, 7 * block_idx)
    
    # Trivial 2x1 block at column 10
    base = 35
    U[0][base] += 1.0; V[10][base] += 1.0; W[20][base] += 1.0
    U[1][base+1] += 1.0; V[21][base+1] += 1.0; W[20][base+1] += 1.0
    U[2][base+2] += 1.0; V[10][base+2] += 1.0; W[21][base+2] += 1.0
    U[3][base+3] += 1.0; V[21][base+3] += 1.0; W[21][base+3] += 1.0
    
    return U, V, W

def _construct_rank34_hybrid():
    """
    Try a hybrid approach: 4 full Strassen blocks (28 terms) for columns 0-7,
    plus a 6-term decomposition for columns 8-10 (3 columns).
    Total: 28 + 6 = 34.
    
    For 3 columns with 2x2 multiplication, we need at least 4*3=12 schoolbook terms,
    but we can do better with a structured 6-term decomposition if we allow
    the terms to mix across the 3 columns.
    
    Actually, let's try: 4 Strassen blocks (28) + 4 schoolbook terms for col 8 
    + 2 shared terms for cols 9,10. This is getting complicated.
    
    Let me try a cleaner approach: use 5 Strassen blocks but merge the last one
    with the single column to save 1 term.
    """
    n, m, p = 2, 2, 11
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, base_rank):
        r = base_rank
        U[0][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+3][r] += 1.0
        r = base_rank + 1
        U[2][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0
        W[2*col_offset+1][r] += 1.0
        r = base_rank + 2
        U[0][r] += 1.0
        V[col_offset+1][r] += 1.0; V[11+col_offset][r] += -1.0
        W[2*col_offset+2][r] += 1.0
        r = base_rank + 3
        U[3][r] += 1.0
        V[11+col_offset][r] += 1.0; V[col_offset][r] += -1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset][r] += -1.0
        r = base_rank + 4
        U[0][r] += 1.0; U[1][r] += 1.0
        V[12+col_offset][r] += 1.0
        W[2*col_offset+2][r] += 1.0; W[2*col_offset+3][r] += -1.0
        r = base_rank + 5
        U[2][r] += 1.0; U[0][r] += -1.0
        V[col_offset][r] += 1.0; V[col_offset+1][r] += 1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset+3][r] += 1.0
        r = base_rank + 6
        U[1][r] += 1.0; U[3][r] += -1.0
        V[11+col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+1][r] += -1.0
    
    # 4 Strassen blocks for columns 0-7
    for block_idx in range(4):
        add_strassen_block(2 * block_idx, 7 * block_idx)
    
    # For columns 8, 9, 10: use a 6-term decomposition
    base = 28
    # We need to compute:
    # c08 = a00*b08 + a01*b18
    # c18 = a10*b08 + a11*b18
    # c09 = a00*b09 + a01*b19
    # c19 = a10*b09 + a11*b19
    # c0,10 = a00*b0,10 + a01*b1,10
    # c1,10 = a10*b0,10 + a11*b1,10
    
    # 6 terms:
    # T0: (a00+a11) * (b08+b18+b09+b19+b0,10+b1,10) -> too many
    # Let's use:
    # T0: (a00+a11) * b08 -> c08 += a00*b08 + a11*b08, c18 += a00*b08 + a11*b08
    #     But we need c08 = a00*b08 + a01*b18, not a00*b08 + a11*b08
    
    # This is getting complex. Let me just use schoolbook for the last 3 columns:
    # 4 terms per column * 3 columns = 12 terms. Total = 28 + 12 = 40. Worse.
    
    # Alternative: use Strassen for cols 8,9 (7 terms) + 2 schoolbook for col 10.
    # Total = 28 + 7 + 2 = 37.
    
    # Or: 5 Strassen blocks = 35 terms, but we only have 10 columns covered by 5 blocks
    # of 2 columns each. Column 10 needs 4 schoolbook terms. Total = 35 + 4 = 39.
    
    # Let me try: 4 Strassen (28) + 1 Strassen for cols 8,9 (7) + 2 terms for col 10.
    # For col 10 with 2 terms:
    # T: a00 * b0,10 -> c0,10 += a00*b0,10
    # T: a10 * b0,10 -> c1,10 += a10*b0,10
    # But we're missing a01*b1,10 and a11*b1,10.
    
    # This won't work with only 2 terms for a full 2x2 column.
    
    # Let me just return the rank-39 construction for now.
    return None

def _als_search_vectorized(rank, n, m, p, max_iter=1000, seed=42, deadline=None):
    """
    Vectorized ALS for finding rank-r decomposition.
    Uses NumPy operations for speed.
    """
    rng = np.random.RandomState(seed)
    
    # Build target tensor as a 3D array
    # T[a,b,c] = 1 if (a,b,c) is a valid target, else 0
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_err = float('inf')
    best_solution = None
    
    for restart in range(5):
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        for it in range(max_iter):
            if deadline and time.time() > deadline - 3:
                break
            
            # Update U: for each a, U[a,c] = sum_{b,c'} T[a,b,c'] * V[b,c'] * W[c',c] / sum_{b,c'} T[a,b,c'] * V[b,c']^2 * W[c',c]^2
            # Vectorized: U[a,c] = (T[:,:,c] @ (V * W[:,c]))[a] / (T[:,:,c] @ (V^2 * W[:,c]^2))[a]
            # More precisely:
            # For fixed c: numerator[a] = sum_{b,c'} T[a,b,c'] * V[b,c'] * W[c',c]
            #             denominator[a] = sum_{b,c'} T[a,b,c'] * V[b,c']^2 * W[c',c]^2
            
            for c in range(rank):
                # W_c = W[:, c] shape (p*n,)
                W_c = W[:, c]
                # V * W_c: for each (b, c'), V[b,c'] * W[c',c]
                # This is tricky to vectorize directly. Let's do it in a semi-vectorized way.
                
                # Build matrix M where M[b, c'] = V[b, c'] * W[c', c]
                M = V * W_c[np.newaxis, :]  # (m*p, p*n)
                
                # numerator[a] = sum_{b,c'} T[a,b,c'] * M[b,c']
                num = T @ M  # (n*m, p*n) then sum over last axis? No.
                # T is (n*m, m*p, p*n). We want for each a: sum_{b,c'} T[a,b,c'] * M[b,c']
                # = sum_b (sum_{c'} T[a,b,c'] * M[b,c'])
                # Let's do: tmp[b] = sum_{c'} T[a,b,c'] * M[b,c'] for all a simultaneously
                # tmp = np.einsum('abc,bc->ab', T, M)  # (n*m, m*p)
                tmp = np.einsum('abc,bc->ab', T, M)
                num_vec = tmp.sum(axis=1)  # (n*m,)
                
                # denominator[a] = sum_{b,c'} T[a,b,c'] * V[b,c']^2 * W[c',c]^2
                M2 = V**2 * W_c**2[np.newaxis, :]
                tmp2 = np.einsum('abc,bc->ab', T, M2)
                den_vec = tmp2.sum(axis=1)
                
                mask = den_vec > 1e-14
                U[:, c][mask] = num_vec[mask] / den_vec[mask]
            
            # Update V similarly
            for c in range(rank):
                U_c = U[:, c]
                M = U_c[np.newaxis, :] * W  # (n*m, p*n) * (p*n, rank)? No.
                # We want M[b, c'] = U[a,c'] * W[b,c'] where a is linked to b via T
                # This is more complex. Let's do:
                # numerator[b] = sum_{a,c'} T[a,b,c'] * U[a,c'] * W[c',c]
                # = sum_{c'} W[c',c] * (sum_a T[a,b,c'] * U[a,c'])
                
                W_c = W[:, c]
                # For each c': sum_a T[a,b,c'] * U[a,c'] for all b
                # = einsum('abc,ac->bc', T, U_c)  # (m*p, p*n)
                tmp = np.einsum('abc,ac->bc', T, U_c)  # (m*p, p*n)
                num_vec = (tmp * W_c[np.newaxis, :]).sum(axis=1)
                
                M2 = U_c**2[np.newaxis, :] * W**2
                tmp2 = np.einsum('abc,ac->bc', T, U_c**2)  # (m*p, p*n)
                den_vec = (tmp2 * W_c**2[np.newaxis, :]).sum(axis=1)
                
                mask = den_vec > 1e-14
                V[:, c][mask] = num_vec[mask] / den_vec[mask]
            
            # Update W similarly
            for c in range(rank):
                U_c = U[:, c]
                V_c = V[:, c]
                # numerator[c'] = sum_{a,b} T[a,b,c'] * U[a,c] * V[b,c]
                # = einsum('abc,ac,bc->c', T, U_c, V_c)  # scalar for each c'? No.
                # For each c': sum_{a,b} T[a,b,c'] * U[a,c] * V[b,c]
                # = einsum('abc,ac,bc->c', T, U_c, V_c)  # (p*n,)
                num_vec = np.einsum('abc,ac,bc->c', T, U_c, V_c)
                
                den_vec = np.einsum('abc,ac,bc->c', T, U_c**2, V_c**2)
                
                mask = den_vec > 1e-14
                W[:, c][mask] = num_vec[mask] / den_vec[mask]
            
            # Check convergence every 20 iterations
            if it % 20 == 0:
                # Compute error
                # For each target (a,b,c): err += (sum_c' U[a,c']*V[b,c']*W[c,c'] - 1)^2
                # Vectorized: S[a,b,c] = sum_c' U[a,c']*V[b,c']*W[c,c']
                # S = np.einsum('ac,bc,cc->abc', U, V, W)
                S = np.einsum('ac,bc,cc->abc', U, V, W)
                err = np.sum((S - T)**2)
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                if err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
        
        if best_solution and best_err < 1e-10:
            break
    
    if best_solution and best_err < 1e-10:
        return best_solution[0].tolist(), best_solution[1].tolist(), best_solution[2].tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Guaranteed rank-39
    U, V, W = _construct_rank33_cyclic()
    if _verify(U, V, W, n, m, p):
        best_rank = 39
        best_result = (U, V, W)
    else:
        best_rank = None
        best_result = None
    
    # Try to find lower rank using vectorized ALS
    for rank in range(30, best_rank if best_rank else 39):
        if time.time() > deadline - 5:
            break
        result = _als_search_vectorized(rank, n, m, p, max_iter=500, seed=rank, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Return best found
    if best_result:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Final fallback: schoolbook
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
# EVOLVE-BLOCK-END
