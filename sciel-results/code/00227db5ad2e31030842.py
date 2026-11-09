"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _make_rank1(i, j, k, n, m, p):
    """Create a single rank-1 tensor (schoolbook term)."""
    U = [0.0] * (n * m)
    V = [0.0] * (m * p)
    W = [0.0] * (p * n)
    U[i * m + j] = 1.0
    V[j * p + k] = 1.0
    W[k * n + i] = 1.0
    return U, V, W

def _strassen_2x2x2():
    """Return 7 rank-1 decompositions for 2x2x2 matrix multiplication.
    Each is (U, V, W) with entries being 0 or +/-1.
    U index: i*m+j = i*2+j (4 entries)
    V index: j*p+k = j*2+k (4 entries)
    W index: k*n+i = k*2+i (4 entries)
    """
    decomps = []
    # The 7 Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    def add(U, V, W):
        decomps.append((U[:], V[:], W[:]))
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is at (0,0), A22 at (1,1)
    # B11 is at (0,0), B22 at (1,1)
    U = [1,0,0,1]; V = [1,0,0,1]; W = [1,0,0,1]
    add(U, V, W)
    
    # M2 = (A21+A22)B11
    # A21 at (1,0), A22 at (1,1)
    # B11 at (0,0)
    U = [0,0,1,1]; V = [1,0,0,0]; W = [1,0,0,0]
    add(U, V, W)
    
    # M3 = A11(B21-B22)
    # A11 at (0,0)
    # B21 at (1,0), B22 at (1,1)
    U = [1,0,0,0]; V = [0,0,1,-1]; W = [0,0,1,-1]
    add(U, V, W)
    
    # M4 = A22(B12-B11)
    # A22 at (1,1)
    # B12 at (0,1), B11 at (0,0)
    U = [0,0,0,1]; V = [-1,1,0,0]; W = [-1,1,0,0]
    add(U, V, W)
    
    # M5 = (A11+A12)B22
    # A11 at (0,0), A12 at (0,1)
    # B22 at (1,1)
    U = [1,1,0,0]; V = [0,0,0,1]; W = [0,0,0,1]
    add(U, V, W)
    
    # M6 = (A21-A11)(B11+B12)
    # A21 at (1,0), A11 at (0,0)
    # B11 at (0,0), B12 at (0,1)
    U = [-1,0,1,0]; V = [1,1,0,0]; W = [1,1,0,0]
    add(U, V, W)
    
    # M7 = (A12-A22)(B21+B22)
    # A12 at (0,1), A22 at (1,1)
    # B21 at (1,0), B22 at (1,1)
    U = [0,1,0,-1]; V = [0,0,1,1]; W = [0,0,1,1]
    add(U, V, W)
    
    return decomps

def _block_strassen(n, m, p, i_off, j_off, k_off):
    """Apply Strassen decomposition to an n x m x p sub-block.
    Splits into 2x2 blocks where possible, using recursive structure.
    Returns list of (U, V, W) decompositions for the full n*m x m*p x p*n block.
    """
    if n == 2 and m == 2 and p == 2:
        base = _strassen_2x2x2()
        result = []
        for U, V, W in base:
            # Offset indices
            U_off = [0.0] * (n * m)
            V_off = [0.0] * (m * p)
            W_off = [0.0] * (p * n)
            for idx in range(n * m):
                if U[idx] != 0:
                    U_off[idx] = U[idx]
            for idx in range(m * p):
                if V[idx] != 0:
                    V_off[idx] = V[idx]
            for idx in range(p * n):
                if W[idx] != 0:
                    W_off[idx] = W[idx]
            result.append((U_off, V_off, W_off))
        return result
    
    # For general n, m, p, we'll use schoolbook with Strassen where possible
    # This is a fallback - for now, use schoolbook
    result = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = [0.0] * (n * m)
                V = [0.0] * (m * p)
                W = [0.0] * (p * n)
                U[i * m + j] = 1.0
                V[j * p + k] = 1.0
                W[k * n + i] = 1.0
                result.append((U, V, W))
    return result

def _rectangular_strassen(n, m, p):
    """Find a low-rank decomposition for n x m x p matrix multiplication.
    Uses recursive splitting and Strassen where dimensions allow.
    """
    if n == 1 or m == 1 or p == 1:
        # Schoolbook: rank = n*m*p
        result = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i * m + j] = 1.0
                    V[j * p + k] = 1.0
                    W[k * n + i] = 1.0
                    result.append((U, V, W))
        return result
    
    # Try to split and use Strassen-like recursion
    # Split dimensions in half where possible
    result = []
    
    # If all dimensions are even, we can split into 2x2 blocks
    # and apply Strassen to each block
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n1 = n // 2
        m1 = m // 2
        p1 = p // 2
        
        # Split A into 2x2 blocks: A11, A12; A21, A22
        # Split B into 2x2 blocks: B11, B12; B21, B22
        # C = A*B has blocks: C11=A11*B11+A12*B21, C12=A11*B12+A12*B22
        # C21=A21*B11+A22*B21, C22=A21*B12+A22*B22
        
        # Use Strassen on the 2x2x2 block structure
        # But we need to handle the fact that blocks are n1 x m1 x p1
        
        # For each Strassen multiplication, we need a decomposition of n1 x m1 x p1
        sub_decomp = _rectangular_strassen(n1, m1, p1)
        
        # Now apply the 7 Strassen formulas
        strassen_2x2 = _strassen_2x2x2()
        
        for U_s, V_s, W_s in strassen_2x2:
            # U_s, V_s, W_s have 4 entries each (for 2x2 blocks)
            # Map to full dimensions
            # U index in 2x2: block_i * 2 + block_j
            # Full U index: (block_i * n1 + i_local) * m + (block_j * m1 + j_local)
            
            for sub_U, sub_V, sub_W in sub_decomp:
                U = [0.0] * (n * m)
                V = [0.0] * (m * p)
                W = [0.0] * (p * n)
                
                # U_s has 4 entries corresponding to 2x2 A blocks
                # sub_U has n1*m1 entries
                for ui in range(4):
                    val = U_s[ui]
                    if val != 0:
                        bi = ui // 2  # block row
                        bj = ui % 2   # block col
                        for su in range(n1 * m1):
                            if sub_U[su] != 0:
                                si = su // m1
                                sj = su % m1
                                full_idx = (bi * n1 + si) * m + (bj * m1 + sj)
                                U[full_idx] += val * sub_U[su]
                
                for vi in range(4):
                    val = V_s[vi]
                    if val != 0:
                        bj = vi // 2  # block row in B
                        bk = vi % 2   # block col in B
                        for sv in range(m1 * p1):
                            if sub_V[sv] != 0:
                                sj = sv // p1
                                sk = sv % p1
                                full_idx = (bj * m1 + sj) * p + (bk * p1 + sk)
                                V[full_idx] += val * sub_V[sv]
                
                for wi in range(4):
                    val = W_s[wi]
                    if val != 0:
                        bk = wi // 2  # block row in C (k direction)
                        bi = wi % 2   # block col in C (i direction)
                        for sw in range(p1 * n1):
                            if sub_W[sw] != 0:
                                sk = sw // n1
                                si = sw % n1
                                full_idx = (bk * p1 + sk) * n + (bi * n1 + si)
                                W[full_idx] += val * sub_W[sw]
                
                result.append((U, V, W))
        
        return result
    
    # Fallback: schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = [0.0] * (n * m)
                V = [0.0] * (m * p)
                W = [0.0] * (p * n)
                U[i * m + j] = 1.0
                V[j * p + k] = 1.0
                W[k * n + i] = 1.0
                result.append((U, V, W))
    return result

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for c in range(rank):
                    computed += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(computed - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    # Start with schoolbook
    best_U = [[0] * (n*m*p) for _ in range(n*m)]
    best_V = [[0] * (n*m*p) for _ in range(m*p)]
    best_W = [[0] * (n*m*p) for _ in range(p*n)]
    best_rank = n * m * p
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try rectangular Strassen
    try:
        decomp = _rectangular_strassen(n, m, p)
        rank = len(decomp)
        if rank < best_rank:
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for c, (U_c, V_c, W_c) in enumerate(decomp):
                for idx in range(n * m):
                    if U_c[idx] != 0:
                        U[idx][c] = U_c[idx]
                for idx in range(m * p):
                    if V_c[idx] != 0:
                        V[idx][c] = V_c[idx]
                for idx in range(p * n):
                    if W_c[idx] != 0:
                        W[idx][c] = W_c[idx]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_U = U
                best_V = V
                best_W = W
                best_rank = rank
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Return best result found
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
