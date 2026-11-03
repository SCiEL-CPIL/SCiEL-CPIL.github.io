"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import random

def _mul_tensor(n, m, p):
    """Return dict {(i,j,k):1} for matmul tensor T_{i,j,k}."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,j,k}."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1.0 if (i, j, k) in _mul_tensor(n, m, p) else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def _strassen_block_2x2(rank_offset, U, V, W, i0, j0, k0):
    """Fill 2x2 Strassen block at rows i0..i0+1, cols j0..j0+1, out k0..k0+1.
       U rows: (i0..i0+1, j0..j0+1) -> 4 rows
       V rows: (j0..j0+1, k0..k0+1) -> 4 rows
       W rows: (k0..k0+1, i0..i0+1) -> 4 rows (order k*n+i)
    """
    # Strassen for C = A*B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # We need to map these to U,V,W entries.
    # U index: row = i*m + j, col = c
    # V index: row = j*p + k, col = c
    # W index: row = k*n + i, col = c
    # For 2x2 block, local indices:
    # A11 -> (i0,j0), A12 -> (i0,j0+1), A21 -> (i0+1,j0), A22 -> (i0+1,j0+1)
    # B11 -> (j0,k0), B12 -> (j0,k0+1), B21 -> (j0+1,k0), B22 -> (j0+1,k0+1)
    # C11 -> (k0,i0), C12 -> (k0,i0+1), C21 -> (k0+1,i0), C22 -> (k0+1,i0+1)
    
    # M1: (A11+A22)(B11+B22)
    c = rank_offset
    # U: A11 and A22 contribute to M1
    U[i0*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    # V: B11 and B22 contribute to M1
    V[j0*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    # W: C11 and C22 get M1
    W[k0*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    rank_offset += 1
    
    # M2: (A21+A22)B11
    c = rank_offset
    U[(i0+1)*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    W[(k0+1)*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    rank_offset += 1
    
    # M3: A11(B21-B22)
    c = rank_offset
    U[i0*m + j0][c] += 1
    V[(j0+1)*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += -1
    W[k0*n + (i0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    rank_offset += 1
    
    # M4: A22(B12-B21)
    c = rank_offset
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + (k0+1)][c] += 1
    V[(j0+1)*p + k0][c] += -1
    W[k0*n + i0][c] += 1
    W[(k0+1)*n + i0][c] += 1
    rank_offset += 1
    
    # M5: (A11+A12)B22
    c = rank_offset
    U[i0*m + j0][c] += 1
    U[i0*m + (j0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += -1
    W[(k0+1)*n + (i0+1)][c] += -1
    rank_offset += 1
    
    # M6: (A21-A11)(B11+B12)
    c = rank_offset
    U[(i0+1)*m + j0][c] += 1
    U[i0*m + j0][c] += -1
    V[j0*p + k0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    rank_offset += 1
    
    # M7: (A12-A22)(B21+B22)
    c = rank_offset
    U[i0*m + (j0+1)][c] += 1
    U[(i0+1)*m + (j0+1)][c] += -1
    V[(j0+1)*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
    rank_offset += 1
    
    return rank_offset

def _build_strassen_partition(n, m, p):
    """Build a Strassen-based decomposition for n x m * m x p.
       Partition into 2x2 blocks where possible, handle remainders with schoolbook.
    """
    rank = 0
    U = [[0] * (n*m* p) for _ in range(n*m)]  # over-allocate, will trim
    V = [[0] * (n*m* p) for _ in range(m*p)]
    W = [[0] * (n*m* p) for _ in range(p*n)]
    
    # Process 2x2 blocks
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    rank = _strassen_block_2x2(rank, U, V, W, i0, j0, k0)
                else:
                    # Schoolbook for remaining
                    for i in range(i0, n):
                        for j in range(j0, m):
                            for k in range(k0, p):
                                U[i*m+j][rank] = 1
                                V[j*p+k][rank] = 1
                                W[k*n+i][rank] = 1
                                rank += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        # If we broke out early, handle remaining rows/cols
        pass
    
    # Actually, let's do a cleaner partition:
    # For each 2x2x2 block, use Strassen. For remaining strips, use schoolbook.
    rank = 0
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    rank = _strassen_block_2x2(rank, U, V, W, i0, j0, k0)
                else:
                    # Schoolbook for this 2x2x2 region (or smaller)
                    for i in range(i0, min(i0+2, n)):
                        for j in range(j0, min(j0+2, m)):
                            for k in range(k0, min(k0+2, p)):
                                U[i*m+j][rank] = 1
                                V[j*p+k][rank] = 1
                                W[k*n+i][rank] = 1
                                rank += 1
    # Handle remaining rows/cols not covered by 2x2 blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) was already covered
                # Simple approach: just add all schoolbook terms and let duplicates cancel? No, that's wrong.
                # Better: track covered indices
                pass
    
    # Let's just do full schoolbook for now as fallback, and try to improve
    # Actually, let's just return the Strassen + schoolbook mix properly
    # For simplicity, let's just do schoolbook and then try to optimize
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    rank = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][rank] = 1
                V[j*p+k][rank] = 1
                W[k*n+i][rank] = 1
                rank += 1
    return U[:n*m][:rank] if rank > 0 else U, V[:m*p][:rank] if rank > 0 else V, W[:p*n][:rank] if rank > 0 else W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][k] = 1
                V[j*p+k][i*m+j] = 1  # Wait, this is wrong. Let me fix.
                W[k*n+i][i*m+j] = 1
    
    # Actually, let's just use the original schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try Strassen-based construction
    if time.time() < deadline - 20:
        # Build Strassen 2x2 blocks
        U_s = [[0] * (n*m*p) for _ in range(n*m)]
        V_s = [[0] * (n*m*p) for _ in range(m*p)]
        W_s = [[0] * (n*m*p) for _ in range(p*n)]
        rank_s = 0
        covered = set()
        
        for i0 in range(0, n, 2):
            for j0 in range(0, m, 2):
                for k0 in range(0, p, 2):
                    if i0+1 < n and j0+1 < m and k0+1 < p:
                        rank_s = _strassen_block_2x2(rank_s, U_s, V_s, W_s, i0, j0, k0)
                        for i in [i0, i0+1]:
                            for j in [j0, j0+1]:
                                for k in [k0, k0+1]:
                                    covered.add((i, j, k))
        
        # Add schoolbook for uncovered
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_s[i*m+j][rank_s] = 1
                        V_s[j*p+k][rank_s] = 1
                        W_s[k*n+i][rank_s] = 1
                        rank_s += 1
        
        if _verify(U_s, V_s, W_s, n, m, p):
            if rank_s < best_rank:
                best_rank = rank_s
                best_U = [row[:rank_s] for row in U_s]
                best_V = [row[:rank_s] for row in V_s]
                best_W = [row[:rank_s] for row in W_s]
    
    # Try to reduce rank by zeroing out columns that don't contribute
    if time.time() < deadline - 10:
        # Remove zero columns
        for c in range(best_rank - 1, -1, -1):
            if all(best_U[i][c] == 0 for i in range(n*m)) and \
               all(best_V[j][c] == 0 for j in range(m*p)) and \
               all(best_W[k][c] == 0 for k in range(p*n)):
                for i in range(n*m):
                    best_U[i].pop(c)
                for j in range(m*p):
                    best_V[j].pop(c)
                for k in range(p*n):
                    best_W[k].pop(c)
                best_rank -= 1
    
    # Try random perturbation + rounding to find smaller rank
    if time.time() < deadline - 5:
        # This is a heuristic: try to find a lower-rank decomposition by random search
        # Not very effective, but worth a try
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
