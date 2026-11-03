import time
import numpy as np

def _build_strassen_2x2x2_block(k0, k1, col_start):
    """Build 7 Strassen terms for a 2x2x2 block with k indices k0, k1.
    Returns list of (U_entries, V_entries, W_entries) tuples.
    U entries: list of (row, val)
    V entries: list of (row, val)  
    W entries: list of (row, val)
    """
    terms = []
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    
    # Our A is 2x2: A[i,j] where i in {0,1} (rows), j in {1,2} (cols, but we index as 0,1 locally)
    # Our B is 2x2: B[j,k] where j in {1,2}, k in {k0,k1}
    # Our C is 2x2: C[i,k] where i in {0,1}, k in {k0,k1}
    
    # M1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
    terms.append((
        [(0*m+1, 1), (1*m+2, 1)],
        [(1*p+k0, 1), (2*p+k1, 1)],
        [(k0*n+0, 1), (k1*n+1, 1)]
    ))
    
    # M2 = (A[1,1]+A[1,2]) * B[1,k0]
    terms.append((
        [(1*m+1, 1), (1*m+2, 1)],
        [(1*p+k0, 1)],
        [(k0*n+1, 1), (k1*n+1, -1)]
    ))
    
    # M3 = A[0,1] * (B[1,k1]-B[2,k1])
    terms.append((
        [(0*m+1, 1)],
        [(1*p+k1, 1), (2*p+k1, -1)],
        [(k1*n+0, 1), (k1*n+1, 1)]
    ))
    
    # M4 = A[1,2] * (B[2,k0]-B[1,k0])
    terms.append((
        [(1*m+2, 1)],
        [(2*p+k0, 1), (1*p+k0, -1)],
        [(k0*n+0, 1), (k0*n+1, 1)]
    ))
    
    # M5 = (A[0,1]+A[0,2]) * B[2,k1]
    terms.append((
        [(0*m+1, 1), (0*m+2, 1)],
        [(2*p+k1, 1)],
        [(k0*n+0, -1), (k1*n+0, 1)]
    ))
    
    # M6 = (A[1,1]-A[0,1]) * (B[1,k0]+B[1,k1])
    terms.append((
        [(1*m+1, 1), (0*m+1, -1)],
        [(1*p+k0, 1), (1*p+k1, 1)],
        [(k1*n+1, 1)]
    ))
    
    # M7 = (A[0,2]-A[1,2]) * (B[2,k0]+B[2,k1])
    terms.append((
        [(0*m+2, 1), (1*m+2, -1)],
        [(2*p+k0, 1), (2*p+k1, 1)],
        [(k0*n+0, 1)]
    ))
    
    return terms

def _build_rank_44():
    """Build the known rank-44 decomposition."""
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Part 2: 2x2x8 part (j=1,2)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _verify_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) with same (i,j,k)
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _try_rank_40(deadline):
    """Attempt to find a rank-40 decomposition using zero-padded embedding.
    
    Strategy: Embed 3x8 into 4x8 with zero-padded row 3.
    Then 2x4x8 = 2x(2x2)x8. Apply Strassen on the 2x2 (j=1,2) and 2x2 (j=0,3) blocks.
    Since row 3 of B is zero, terms involving B[3,*] vanish, potentially allowing
    rank reduction.
    
    Actually, let's try a different approach: use 16 terms for j=0 (schoolbook)
    and try to do j=1,2 with fewer than 28 terms by sharing structure across k-blocks.
    """
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    target_rank = 40
    
    # Approach: Try to find a valid rank-40 by modifying the rank-44 solution
    # to eliminate 4 redundant terms.
    
    # The rank-44 has:
    # - 16 terms for j=0 (schoolbook: 2 rows * 8 cols)
    # - 28 terms for j=1,2 (4 blocks * 7 Strassen terms)
    
    # Can we do j=1,2 with 24 terms? That would be 6 per 2x2x2 block.
    # This is impossible in general (Strassen needs 7), but maybe with the
    # specific structure of our problem, we can merge some terms.
    
    # Alternative: Use a different block decomposition.
    # Split j=1,2,k=0..7 into 2 blocks of 2x2x4 instead of 4 blocks of 2x2x2.
    # For 2x2x4, we can use 2*7=14 terms (two 2x2x2 Strassen) or potentially fewer.
    
    # Let's try: 16 terms for j=0, and 24 terms for j=1,2.
    # For j=1,2: use 6 terms per 2x2x2 block for 4 blocks = 24 terms.
    # This requires that 6 terms suffice for each 2x2x2 block, which is not
    # generally possible. So this won't work directly.
    
    # Better approach: Try to find any rank-40 solution via random search
    # or structured search. Given time constraints, let's try a few structured
    # patterns.
    
    # Pattern 1: 16 terms for j=0, 24 terms for j=1,2
    # Use 3 blocks: k=0..1, k=2..3, k=4..5, k=6..7
    # But use a different decomposition for the 2x2x2 blocks that uses 6 terms.
    
    # Actually, let's try a completely different approach:
    # Use 12 terms for j=0 (by sharing across k) and 28 terms for j=1,2.
    # For j=0: A[:,0] is 2x1, B[0,:] is 1x8. C[:,k] = A[:,0]*B[0,k].
    # We can write: for each k, C[0,k] = A[0,0]*B[0,k], C[1,k] = A[1,0]*B[0,k].
    # That's 16 terms. Can we do better?
    # C[0,k] - C[1,k] = (A[0,0]-A[1,0])*B[0,k]
    # C[0,k] + C[1,k] = (A[0,0]+A[1,0])*B[0,k]
    # Still 16 terms if we need both C[0,k] and C[1,k] independently.
    # Unless we can combine across k. But B[0,k] are independent, so no.
    
    # Let's try: 14 terms for j=0, 26 terms for j=1,2.
    # Or: 12 terms for j=0, 28 terms for j=1,2.
    
    # Given the difficulty, let's just try to verify if rank 40 is achievable
    # by attempting a few specific patterns.
    
    # Pattern: Use 16 terms for j=0, and for j=1,2 use 24 terms by
    # exploiting that we only need to compute 2x8 = 16 output entries.
    # The minimum rank for 2x2x8 is at least 16 (since we have 16 independent
    # output entries). So 24 terms for j=1,2 is 8 above minimum.
    
    # Let's try a structured approach:
    # For j=1,2, k=0..7:
    # C[0,k] = A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Group k into pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair (k0,k1):
    # C[0,k0] = A[0,1]*B[1,k0] + A[0,2]*B[2,k0]
    # C[0,k1] = A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
    # C[1,k0] = A[1,1]*B[1,k0] + A[1,2]*B[2,k0]
    # C[1,k1] = A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
    
    # That's 4 outputs per pair, 4 pairs = 16 outputs.
    # Minimum rank per 2x2x2 block is 7 (Strassen).
    # 4 blocks * 7 = 28 terms. Total: 16 + 28 = 44.
    
    # To get to 40, we need to save 4 terms.
    # Idea: Share some terms between adjacent blocks.
    # For example, if we use a different decomposition for the 2x2x2 blocks
    # that allows sharing of linear combinations across blocks.
    
    # This is getting complex. Let's just return None and use rank 44.
    return None

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Check all (a,b,c) triples
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = 0
                for r in range(R):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                if val != expected:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with known rank-44 solution
    result = _build_rank_44()
    
    # Try to find a better (lower rank) solution
    if time.time() < deadline - 10:
        better = _try_rank_40(deadline)
        if better is not None:
            if _verify_fast(better["U"], better["V"], better["W"]):
                result = better
    
    return result

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Check all (a,b,c) triples
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = 0
                for r in range(R):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                if val != expected:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
