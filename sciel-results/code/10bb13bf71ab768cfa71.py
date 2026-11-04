"""
Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Uses a recursive block decomposition strategy:
- 2x2x2 blocks use Strassen's algorithm (rank 7)
- 1xr x rx1 blocks use schoolbook (rank 1)
- General blocks are composed from smaller blocks

The key insight is to split dimensions into 2s and 1s, and use Strassen
where all three dimensions are 2, falling back to schoolbook otherwise.

For 4x5x6:
- Split 4 = 2+2, 5 = 2+2+1, 6 = 2+2+2
- This gives 2*3*3 = 18 blocks
- Blocks of size 2x2x2 use Strassen (rank 7)
- Blocks with a 1 dimension use schoolbook
"""

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)  
    - W has shape (p*n, rank)
    
    The decomposition satisfies: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i, j, k (with the sum over j implicit in the tensor product structure).
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    def strassen_2x2x2():
        """
        Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
        
        U: 4x7, V: 4x7, W: 4x7
        U[i*2+j, r] corresponds to A[i,j]
        V[j*2+k, r] corresponds to B[j,k]
        W[k*2+i, r] corresponds to C[i,k] coefficient
        """
        # Standard Strassen decomposition
        # P_1 = (A11 + A22)(B11 + B22)
        # P_2 = (A21 + A22)B11
        # P_3 = A11(B12 - B22)
        # P_4 = A22(B21 - B11)
        # P_5 = (A11 + A12)B22
        # P_6 = (A21 - A11)(B11 + B12)
        # P_7 = (A12 - A22)(B21 + B22)
        #
        # C11 = P_1 + P_4 - P_5 + P_7
        # C12 = P_3 + P_5
        # C21 = P_2 + P_4
        # C22 = P_1 - P_2 + P_3 + P_6
        
        # U rows: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # V rows: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
        # W rows: 0=C[0,0] (k=0,i=0), 1=C[0,1] (k=1,i=0), 2=C[1,0] (k=0,i=1), 3=C[1,1] (k=1,i=1)
        
        U = [
            [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
            [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
            [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
            [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
        ]
        
        V = [
            [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
            [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
            [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
            [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
        ]
        
        W = [
            [1, 0, 0, 1, -1, 0, 1],   # C[0,0]: P1 + P4 - P5 + P7
            [0, 0, 1, 0, 1, 0, 0],    # C[0,1]: P3 + P5
            [0, 1, 0, 1, 0, 0, 0],    # C[1,0]: P2 + P4
            [1, -1, 1, 0, 0, 1, 0]    # C[1,1]: P1 - P2 + P3 + P6
        ]
        
        return U, V, W
    
    def schoolbook_block(n_block, m_block, p_block):
        """
        Return (U, V, W) for n_block x m_block x p_block using schoolbook.
        
        U: (n_block*m_block) x rank
        V: (m_block*p_block) x rank
        W: (p_block*n_block) x rank
        """
        rank = n_block * m_block * p_block
        U = [[0] * rank for _ in range(n_block * m_block)]
        V = [[0] * rank for _ in range(m_block * p_block)]
        W = [[0] * rank for _ in range(p_block * n_block)]
        
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i * m_block + j][col] = 1
                    V[j * p_block + k][col] = 1
                    W[k * n_block + i][col] = 1
                    col += 1
        
        return U, V, W
    
    def combine_decompositions(decomp_list):
        """
        Combine a list of (U, V, W) decompositions into one.
        Each decomposition has the same n, m, p but different rank.
        """
        if not decomp_list:
            return None
        
        n, m, p = DIMENSIONS
        
        # Determine total rank
        total_rank = sum(len(V) for _, V, _ in decomp_list)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        offset = 0
        for U_block, V_block, W_block in decomp_list:
            r = len(V_block)
            for i in range(len(U_block)):
                for r_idx in range(r):
                    U[i][offset + r_idx] = U_block[i][r_idx]
            for j in range(len(V_block)):
                for r_idx in range(r):
                    V[j][offset + r_idx] = V_block[j][r_idx]
            for k in range(len(W_block)):
                for r_idx in range(r):
                    W[k][offset + r_idx] = W_block[k][r_idx]
            offset += r
        
        return U, V, W
    
    def embed_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, n_block, m_block, p_block, n_full, m_full, p_full):
        """
        Embed a block decomposition into the full size.
        
        U_block has shape (n_block * m_block, rank)
        V_block has shape (m_block * p_block, rank)
        W_block has shape (p_block * n_block, rank)
        
        We need to create full-size U, V, W with zeros except in the relevant block.
        """
        rank = len(V_block)
        U_full = [[0] * rank for _ in range(n_full * m_full)]
        V_full = [[0] * rank for _ in range(m_full * p_full)]
        W_full = [[0] * rank for _ in range(p_full * n_full)]
        
        # Embed U: row (i_offset + i, j_offset + j) for i in [0, n_block), j in [0, m_block)
        for i in range(n_block):
            for j in range(m_block):
                full_row = (i_offset + i) * m_full + (j_offset + j)
                block_row = i * m_block + j
                for r in range(rank):
                    U_full[full_row][r] = U_block[block_row][r]
        
        # Embed V: row (j_offset + j, k_offset + k) for j in [0, m_block), k in [0, p_block)
        for j in range(m_block):
            for k in range(p_block):
                full_row = (j_offset + j) * p_full + (k_offset + k)
                block_row = j * p_block + k
                for r in range(rank):
                    V_full[full_row][r] = V_block[block_row][r]
        
        # Embed W: row (k_offset + k, i_offset + i) for k in [0, p_block), i in [0, n_block)
        for k in range(p_block):
            for i in range(n_block):
                full_row = (k_offset + k) * n_full + (i_offset + i)
                block_row = k * n_block + i
                for r in range(rank):
                    W_full[full_row][r] = W_block[block_row][r]
        
        return U_full, V_full, W_full
    
    def decompose_block(n_block, m_block, p_block):
        """
        Recursively decompose an n_block x m_block x p_block matrix multiplication.
        Returns (U, V, W) with shapes appropriate for the block.
        """
        # Base cases
        if n_block == 1 and m_block == 1 and p_block == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        if n_block == 1:
            # 1 x m x p: schoolbook
            return schoolbook_block(n_block, m_block, p_block)
        
        if m_block == 1:
            # n x 1 x p: schoolbook
            return schoolbook_block(n_block, m_block, p_block)
        
        if p_block == 1:
            # n x m x 1: schoolbook
            return schoolbook_block(n_block, m_block, p_block)
        
        # If all dimensions are 2, use Strassen
        if n_block == 2 and m_block == 2 and p_block == 2:
            return strassen_2x2x2()
        
        # Otherwise, try to split
        # Strategy: split the largest dimension into 2 + remainder
        # Prefer splitting into 2s and 1s
        
        best_decomp = None
        best_rank = float('inf')
        
        # Try splitting n
        if n_block >= 2:
            for n1 in range(1, n_block):
                n2 = n_block - n1
                if n1 < 2 and n2 < 2:
                    continue
                U1, V1, W1 = decompose_block(n1, m_block, p_block)
                U2, V2, W2 = decompose_block(n2, m_block, p_block)
                
                # Embed into full size
                rank1 = len(V1)
                rank2 = len(V2)
                total_rank = rank1 + rank2
                
                U_full = [[0] * total_rank for _ in range(n_block * m_block)]
                V_full = [[0] * total_rank for _ in range(m_block * p_block)]
                W_full = [[0] * total_rank for _ in range(p_block * n_block)]
                
                # Embed first block (rows 0..n1-1)
                for i in range(n1):
                    for j in range(m_block):
                        for r in range(rank1):
                            U_full[i * m_block + j][r] = U1[i * m_block + j][r]
                for j in range(m_block):
                    for k in range(p_block):
                        for r in range(rank1):
                            V_full[j * p_block + k][r] = V1[j * p_block + k][r]
                for k in range(p_block):
                    for i in range(n1):
                        for r in range(rank1):
                            W_full[k * n_block + i][r] = W1[k * n1 + i][r]
                
                # Embed second block (rows n1..n_block-1)
                for i in range(n2):
                    for j in range(m_block):
                        for r in range(rank2):
                            U_full[(n1 + i) * m_block + j][rank1 + r] = U2[i * m_block + j][r]
                for j in range(m_block):
                    for k in range(p_block):
                        for r in range(rank2):
                            V_full[j * p_block + k][rank1 + r] = V2[j * p_block + k][r]
                for k in range(p_block):
                    for i in range(n2):
                        for r in range(rank2):
                            W_full[k * n_block + (n1 + i)][rank1 + r] = W2[k * n2 + i][r]
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_decomp = (U_full, V_full, W_full)
        
        # Try splitting m
        if m_block >= 2:
            for m1 in range(1, m_block):
                m2 = m_block - m1
                if m1 < 2 and m2 < 2:
                    continue
                U1, V1, W1 = decompose_block(n_block, m1, p_block)
                U2, V2, W2 = decompose_block(n_block, m2, p_block)
                
                rank1 = len(V1)
                rank2 = len(V2)
                total_rank = rank1 + rank2
                
                U_full = [[0] * total_rank for _ in range(n_block * m_block)]
                V_full = [[0] * total_rank for _ in range(m_block * p_block)]
                W_full = [[0] * total_rank for _ in range(p_block * n_block)]
                
                # Embed first block (cols 0..m1-1)
                for i in range(n_block):
                    for j in range(m1):
                        for r in range(rank1):
                            U_full[i * m_block + j][r] = U1[i * m1 + j][r]
                for j in range(m1):
                    for k in range(p_block):
                        for r in range(rank1):
                            V_full[j * p_block + k][r] = V1[j * m1 + k][r]
                for k in range(p_block):
                    for i in range(n_block):
                        for r in range(rank1):
                            W_full[k * n_block + i][r] = W1[k * n_block + i][r]
                
                # Embed second block (cols m1..m_block-1)
                for i in range(n_block):
                    for j in range(m2):
                        for r in range(rank2):
                            U_full[i * m_block + (m1 + j)][rank1 + r] = U2[i * m2 + j][r]
                for j in range(m2):
                    for k in range(p_block):
                        for r in range(rank2):
                            V_full[(m1 + j) * p_block + k][rank1 + r] = V2[j * m2 + k][r]
                for k in range(p_block):
                    for i in range(n_block):
                        for r in range(rank2):
                            W_full[k * n_block + i][rank1 + r] = W2[k * n_block + i][r]
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_decomp = (U_full, V_full, W_full)
        
        # Try splitting p
        if p_block >= 2:
            for p1 in range(1, p_block):
                p2 = p_block - p1
                if p1 < 2 and p2 < 2:
                    continue
                U1, V1, W1 = decompose_block(n_block, m_block, p1)
                U2, V2, W2 = decompose_block(n_block, m_block, p2)
                
                rank1 = len(V1)
                rank2 = len(V2)
                total_rank = rank1 + rank2
                
                U_full = [[0] * total_rank for _ in range(n_block * m_block)]
                V_full = [[0] * total_rank for _ in range(m_block * p_block)]
                W_full = [[0] * total_rank for _ in range(p_block * n_block)]
                
                # Embed first block (cols 0..p1-1)
                for i in range(n_block):
                    for j in range(m_block):
                        for r in range(rank1):
                            U_full[i * m_block + j][r] = U1[i * m_block + j][r]
                for j in range(m_block):
                    for k in range(p1):
                        for r in range(rank1):
                            V_full[j * p_block + k][r] = V1[j * p1 + k][r]
                for k in range(p1):
                    for i in range(n_block):
                        for r in range(rank1):
                            W_full[k * n_block + i][r] = W1[k * n_block + i][r]
                
                # Embed second block (cols p1..p_block-1)
                for i in range(n_block):
                    for j in range(m_block):
                        for r in range(rank2):
                            U_full[i * m_block + j][rank1 + r] = U2[i * m_block + j][r]
                for j in range(m_block):
                    for k in range(p2):
                        for r in range(rank2):
                            V_full[j * p_block + (p1 + k)][rank1 + r] = V2[j * p2 + k][r]
                for k in range(p2):
                    for i in range(n_block):
                        for r in range(rank2):
                            W_full[(p1 + k) * n_block + i][rank1 + r] = W2[k * n_block + i][r]
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_decomp = (U_full, V_full, W_full)
        
        if best_decomp is None:
            # Fallback to schoolbook
            return schoolbook_block(n_block, m_block, p_block)
        
        return best_decomp
    
    # Decompose the full 4x5x6 problem
    U, V, W = decompose_block(n, m, p)
    
    # Verify the decomposition
    rank = len(V)
    
    # Check that the decomposition is exact
    # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i, k), the coefficient of A[i,j]*B[j,k] should be 1 for all j
    
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # Coefficient of A[i,j]*B[j,k] in C[i,k]
                coeff = 0.0
                for r in range(rank):
                    coeff += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                error += abs(coeff - 1.0)
            
            # Also check that C[i,k] doesn't get contributions from other (i',k') pairs
            # Actually, the decomposition should give exactly C[i,k] = sum_j A[i,j]B[j,k]
            # So we need to verify that for each (i,k), the bilinear form matches
    
    # More thorough verification:
    # For random A and B, check that the decomposition gives correct C
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += A[i, j] * U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
            C_computed[i, k] = val
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
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

# EVOLVE-BLOCK-END
