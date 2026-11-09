"""Matrix multiplication 4x5x5 rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 x 2x2 multiplication.
    Returns (U, V, W) where U is (4,7), V is (4,7), W is (4,7).
    
    The 7 products:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: maps A entries (A11, A12, A21, A22) to the 7 Strassen products
    # V: maps B entries (B11, B12, B21, B22) to the 7 Strassen products
    # W: maps C entries (C11, C12, C21, C22) from the 7 Strassen products
    
    # Product 1: (A11+A22)*(B11+B22)
    # Product 2: (A21+A22)*B11
    # Product 3: A11*(B12-B22)
    # Product 4: A22*(B21-B11)
    # Product 5: (A11+A12)*B22
    # Product 6: (A21-A11)*(B11+B12)
    # Product 7: (A12-A22)*(B21+B22)
    
    U = [
        [1, 0, 0, 0, 0, 0, 0],   # A11
        [0, 1, 0, 0, 0, 0, 0],   # A12
        [0, 0, 1, 0, 0, 0, 0],   # A21
        [0, 0, 0, 1, 0, 0, 0],   # A22
    ]
    
    # Actually let me be more careful. U should have shape (4, 7) where
    # U[i][c] is the coefficient of A_i in product c.
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Product 1: (A11+A22)*(B11+B22) -> U col: A11=1, A22=1; V col: B11=1, B22=1
    # Product 2: (A21+A22)*B11 -> U col: A21=1, A22=1; V col: B11=1
    # Product 3: A11*(B12-B22) -> U col: A11=1; V col: B12=1, B22=-1
    # Product 4: A22*(B21-B11) -> U col: A22=1; V col: B21=1, B11=-1
    # Product 5: (A11+A12)*B22 -> U col: A11=1, A12=1; V col: B22=1
    # Product 6: (A21-A11)*(B11+B12) -> U col: A21=1, A11=-1; V col: B11=1, B12=1
    # Product 7: (A12-A22)*(B21+B22) -> U col: A12=1, A22=-1; V col: B21=1, B22=1
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1],   # A22
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1],   # B22
    ]
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # The tensor T[i,j,k] = delta_{ik} should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Actually the decomposition is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 if we think of it as the multiplication tensor
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = 1 (the multiplication tensor has all 1s in the "right" sense)
                # Actually, the multiplication tensor is defined as:
                # T[i,j,k] = 1 for all i,j,k? No.
                # The matrix multiplication tensor T_{i,j,k} satisfies:
                # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
                # So T_{i,j,k} = 1 for all i,j,k.
                # 
                # Wait, that's not right either. Let me think again.
                # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # The bilinear form: C_{i,k} = sum_j A_{i,j} B_{j,k}
                # As a trilinear form: sum_{i,j,k} C_{i,k} X_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} X_{i,k}
                # = sum_{i,j,k} (A_{i,j} B_{j,k} X_{i,k})
                # So the tensor is T[i,j,k] = 1 for all valid i,j,k.
                
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple sub-decompositions into one full decomposition.
    decomps: list of (U_sub, V_sub, W_sub, i_start, j_start, k_start)
    where the sub-problem is A[i_start:i_start+ns, j_start:j_start+ms] * B[j_start:j_start+ms, k_start:k_start+ps]
    
    Actually, let me think about this differently.
    For 4x5x5, we can split into blocks.
    
    Let's use a block approach:
    A is 4x5, B is 5x5, C is 4x5.
    
    We can split A into two 2x5 blocks (rows 0-1 and rows 2-3),
    and B into blocks of columns.
    
    Actually, a cleaner approach: split the 5 dimension.
    A is 4x5, B is 5x5.
    
    Split j (the shared dimension) into j1=2 and j2=3.
    A = [A1 (4x2), A2 (4x3)]
    B = [B1 (2x5); B2 (3x5)]
    C = A1*B1 + A2*B2
    
    For A1*B1: 4x2 x 2x5
    For A2*B2: 4x3 x 3x5
    
    For 4x2 x 2x5: split i into two 2x2 blocks, and k into... hmm.
    
    Let me try a different decomposition strategy.
    
    4x5x5:
    - Split i into 2+2 (two groups of 2 rows)
    - Split j into 2+3
    - Split k into 2+3
    
    This gives 2*2*2 = 8 sub-problems:
    (2x2 x 2x2), (2x2 x 2x3), (2x2 x 3x2), (2x2 x 3x3),
    (2x2 x 2x2), (2x2 x 2x3), (2x2 x 3x2), (2x2 x 3x3)
    
    Hmm, this is getting complicated. Let me just do a simpler block decomposition.
    
    Actually, the simplest improvement: use Strassen on 2x2x2 blocks.
    
    For 4x5x5, I can decompose as:
    - 4x2 x 2x5 part + 4x3 x 3x5 part
    
    For the 4x2 x 2x5 part:
    - Split into two 2x2 x 2x2 (Strassen, rank 7 each) + two 2x2 x 2x3 blocks
    
    This is getting complex. Let me just implement a general block decomposition
    and use Strassen where possible.
    """
    pass


def _block_multiply(n, m, p):
    """
    Build a decomposition for n x m times m x p matrix multiplication
    using block decomposition with Strassen for 2x2x2 blocks.
    
    Strategy for 4x5x5:
    Split j into 2+3.
    Part 1: 4x2 x 2x5
    Part 2: 4x3 x 3x5
    
    For 4x2 x 2x5:
    Split i into 2+2, k into 2+3.
    Sub-blocks: (2x2 x 2x2), (2x2 x 2x3), (2x2 x 2x2), (2x2 x 2x3)
    
    For 4x3 x 3x5:
    Split i into 2+2, k into 2+3.
    Sub-blocks: (2x3 x 3x2), (2x3 x 3x3), (2x3 x 3x2), (2x3 x 3x3)
    
    This is still complex. Let me just use schoolbook for non-2x2x2 blocks
    and Strassen for 2x2x2 blocks.
    """
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_block(i0, i1, j0, j1, k0, k1):
        """Add decomposition for A[i0:i1, j0:j1] * B[j0:j1, k0:k1]."""
        ns = i1 - i0
        ms = j1 - j0
        ps = k1 - k0
        
        if ns == 2 and ms == 2 and ps == 2:
            Us, Vs, Ws = _strassen_2x2()
            for r in range(2):
                U_rows.append([0] * 7)
                V_rows.append([0] * 7)
                W_rows.append([0] * 7)
            # Map local indices to global
            for r in range(2):
                for c in range(7):
                    # U: row (i0+r)*m + (j0) .. (i0+r)*m + (j1-1)
                    # Actually U has shape (n*m, rank). Each row corresponds to (i,j).
                    # For this block, the A part is A[i0+r, j0:c_local] where c_local is 0 or 1
                    # U_local[r, c] is the coefficient for A[i0+r, j0+c_local]
                    # So U[(i0+r)*m + (j0+c_local), col] += U_local[r, c]
                    for c_local in range(2):
                        U_rows[(i0 + r) * m + (j0 + c_local)] = [0] * 7
                        U_rows[(i0 + r) * m + (j0 + c_local)][c] = Us[r][c]
                    
                    # V: row (j0+c_local)*p + (k0+d_local)
                    for c_local in range(2):
                        for d_local in range(2):
                            V_rows[(j0 + c_local) * p + (k0 + d_local)] = [0] * 7
                            V_rows[(j0 + c_local) * p + (k0 + d_local)][c] = Vs[c_local][c]
                    
                    # W: row (k0+d_local)*n + (i0+r)
                    for d_local in range(2):
                        W_rows[(k0 + d_local) * n + (i0 + r)] = [0] * 7
                        W_rows[(k0 + d_local) * n + (i0 + r)][c] = Ws[d_local][c]
        
        else:
            # Schoolbook for this block
            rank_s = ns * ms * ps
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        col = j * ps + k  # local column
                        # U
                        U_rows[(i0 + i) * m + (j0 + j)] = [0] * rank_s
                        U_rows[(i0 + i) * m + (j0 + j)][col] = 1
                        # V
                        V_rows[(j0 + j) * p + (k0 + k)] = [0] * rank_s
                        V_rows[(j0 + j) * p + (k0 + k)][col] = 1
                        # W
                        W_rows[(k0 + k) * n + (i0 + i)] = [0] * rank_s
                        W_rows[(k0 + k) * n + (i0 + i)][col] = 1
    
    # This approach has issues with overlapping rows. Let me use a different strategy.
    # Build separate decompositions and concatenate.
    
    blocks = []
    
    def add_block_simple(i0, i1, j0, j1, k0, k1):
        ns = i1 - i0
        ms = j1 - j0
        ps = k1 - k0
        
        if ns == 2 and ms == 2 and ps == 2:
            Us, Vs, Ws = _strassen_2x2()
            # Create local U, V, W with global indexing
            U_local = [[0] * 7 for _ in range(ns * ms)]
            V_local = [[0] * 7 for _ in range(ms * ps)]
            W_local = [[0] * 7 for _ in range(ps * ns)]
            
            for r in range(ns):
                for c_local in range(ms):
                    U_local[r * ms + c_local] = Us[r][:]
            for c_local in range(ms):
                for d_local in range(ps):
                    V_local[c_local * ps + d_local] = Vs[c_local][:]
            for d_local in range(ps):
                for r in range(ns):
                    W_local[d_local * ns + r] = Ws[d_local][:]
            
            blocks.append((U_local, V_local, W_local, i0, j0, k0, ns, ms, ps))
        else:
            rank_s = ns * ms * ps
            U_local = [[0] * rank_s for _ in range(ns * ms)]
            V_local = [[0] * rank_s for _ in range(ms * ps)]
            W_local = [[0] * rank_s for _ in range(ps * ns)]
            
            col = 0
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U_local[i * ms + j][col] = 1
                        V_local[j * ps + k][col] = 1
                        W_local[k * ns + i][col] = 1
                        col += 1
            
            blocks.append((U_local, V_local, W_local, i0, j0, k0, ns, ms, ps))
    
    # Decompose 4x5x5
    # Split j into 2+3
    # Part 1: 4x2 x 2x5 -> split i into 2+2, k into 2+3
    #   (0,2,0,2,0,2) 2x2x2 Strassen
    #   (0,2,0,2,2,5) 2x2x3 schoolbook
    #   (2,4,0,2,0,2) 2x2x2 Strassen
    #   (2,4,0,2,2,5) 2x2x3 schoolbook
    # Part 2: 4x3 x 3x5 -> split i into 2+2, k into 2+3
    #   (0,2,2,5,0,2) 2x3x2 schoolbook
    #   (0,2,2,5,2,5) 2x3x3 schoolbook
    #   (2,4,2,5,0,2) 2x3x2 schoolbook
    #   (2,4,2,5,2,5) 2x3x3 schoolbook
    
    add_block_simple(0, 2, 0, 2, 0, 2)  # 2x2x2 Strassen
    add_block_simple(0, 2, 0, 2, 2, 5)  # 2x2x3 schoolbook
    add_block_simple(2, 4, 0, 2, 0, 2)  # 2x2x2 Strassen
    add_block_simple(2, 4, 0, 2, 2, 5)  # 2x2x3 schoolbook
    add_block_simple(0, 2, 2, 5, 0, 2)  # 2x3x2 schoolbook
    add_block_simple(0, 2, 2, 5, 2, 5)  # 2x3x3 schoolbook
    add_block_simple(2, 4, 2, 5, 0, 2)  # 2x3x2 schoolbook
    add_block_simple(2, 4, 2, 5, 2, 5)  # 2x3x3 schoolbook
    
    # Now combine all blocks by concatenating columns
    # Each block has its own rank. We need to place them in non-overlapping column ranges.
    total_rank = sum(len(b[0][0]) for b in blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (U_local, V_local, W_local, i0, j0, k0, ns, ms, ps) in blocks:
        rank_s = len(U_local[0])
        for r in range(ns):
            for c_local in range(ms):
                for c in range(rank_s):
                    U[(i0 + r) * m + (j0 + c_local)][col_offset + c] = U_local[r * ms + c_local][c]
        for c_local in range(ms):
            for d_local in range(ps):
                for c in range(rank_s):
                    V[(j0 + c_local) * p + (k0 + d_local)][col_offset + c] = V_local[c_local * ps + d_local][c]
        for d_local in range(ps):
            for r in range(ns):
                for c in range(rank_s):
                    W[(k0 + d_local) * n + (i0 + r)][col_offset + c] = W_local[d_local * ns + r][c]
        col_offset += rank_s
    
    return U, V, W, total_rank


def _try_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition via random search / optimization.
    This is a placeholder for more sophisticated search.
    """
    # For now, return the block decomposition
    U, V, W, rank = _block_multiply(n, m, p)
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # First, try the block decomposition with Strassen
    U, V, W, rank = _block_multiply(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = n * m * p
    
    # Check if we have time for search
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try to improve with random perturbations / local search
        # For now, just return what we have
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
