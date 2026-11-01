import time
import random

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, num_tests=10):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(num_tests):
        random.seed(seed * 7 + 13)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s
        
        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s
        
        if C != AB:
            return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2x6():
    """
    Strassen-based decomposition for 2x2x6 matrix multiplication.
    Splits B into 3 blocks of 2 columns and applies Strassen to each.
    Rank = 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_s[s][ij] where ij = i*2+j
    U_s = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1],  # M7: (A01-A11)
    ]
    
    # V_s[s][jk] where jk = j*2+k
    V_s = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [-1, 0, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1],   # M7: (B10+B11)
    ]
    
    # W_s[s][ki] where ki = k*2+i
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    W_s = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, 0],   # M2: C10+1
        [0, 1, 0, -1],  # M3: C01+1, C11-1
        [-1, 0, 1, 0],  # M4: C00-1, C10+1
        [0, 1, 0, 0],   # M5: C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0],   # M7: C00+1
    ]
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            # U: same for all blocks
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            # V: offset by block*2 in the k-dimension
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    v_s_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][v_s_idx]
            # W: offset by block*2 in the k-dimension
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][w_s_idx]
                    
    return {"U": U, "V": V, "W": W}


def _try_rank_reduction(U, V, W, n, m, p, max_iterations=5000):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    If two or more terms are linearly dependent in a certain sense, we can combine them.
    
    Strategy: For each pair of rank-1 terms (r1, r2), check if we can express
    term r2 as a linear combination of other terms. This is a linear algebra problem.
    
    More practically: try to find a vector c such that sum_r c[r] * U[:,r] = 0,
    sum_r c[r] * V[:,r] = 0, but sum_r c[r] * W[:,r] != 0.
    If such a c exists, we can eliminate one term.
    
    Actually, a simpler approach: try to merge pairs of terms.
    For terms r1 and r2, if U[:,r1] = a * U[:,r2] and V[:,r1] = b * V[:,r2],
    then we can potentially merge them if W allows.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try to find pairs that can be merged
    # Two terms r1, r2 can be merged if there exists alpha, beta such that:
    # alpha * U[:,r1] + beta * U[:,r2] = 0 (or proportional)
    # alpha * V[:,r1] + beta * V[:,r2] = 0 (or proportional)  
    # And the merged W works out.
    
    # Simpler: check if U[:,r1] is proportional to U[:,r2] AND V[:,r1] is proportional to V[:,r2]
    for attempt in range(max_iterations):
        if time.time() > 100:  # safety margin
            break
        
        found_merge = False
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                # Check if U[:,r1] is proportional to U[:,r2]
                # Find the first non-zero element in U[:,r1]
                scale_u = None
                for i in range(n * m):
                    if U[i][r1] != 0:
                        if U[i][r2] == 0:
                            break
                        scale_u = U[i][r2] / U[i][r1]
                        break
                else:
                    continue
                    
                # Verify proportionality
                valid = True
                for i in range(n * m):
                    expected = scale_u * U[i][r1]
                    if abs(U[i][r2] - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check if V[:,r1] is proportional to V[:,r2]
                scale_v = None
                for i in range(m * p):
                    if V[i][r1] != 0:
                        if V[i][r2] == 0:
                            break
                        scale_v = V[i][r2] / V[i][r1]
                        break
                else:
                    continue
                    
                # Verify proportionality
                valid = True
                for i in range(m * p):
                    expected = scale_v * V[i][r1]
                    if abs(V[i][r2] - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If both are proportional, try to merge
                # The merged term would have:
                # U_merged = U[:,r1] (or any multiple)
                # V_merged = V[:,r1]
                # W_merged = W[:,r1] + (scale_v/scale_u) * W[:,r2] ... this needs care
                
                # Actually, if U[:,r2] = scale_u * U[:,r1] and V[:,r2] = scale_v * V[:,r1],
                # then term r2 contributes scale_u * scale_v * (U[:,r1] * V[:,r1] * W[:,r2])
                # So we can absorb it into term r1 by adding scale_u * scale_v * W[:,r2] to W[:,r1]
                
                factor = scale_u * scale_v
                for i in range(p * n):
                    W[i][r1] += factor * W[i][r2]
                
                # Remove term r2
                for i in range(n * m):
                    U[i].pop(r2)
                for i in range(m * p):
                    V[i].pop(r2)
                for i in range(p * n):
                    W[i].pop(r2)
                
                rank -= 1
                found_merge = True
                break
            if found_merge:
                break
        
        if not found_merge:
            break
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try Strassen-based decomposition first (rank 21)
    result = _strassen_2x2x6()
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce rank further
        if time.time() < deadline - 30:
            reduced = _try_rank_reduction(result["U"], result["V"], result["W"], n, m, p)
            if _check_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                return reduced
        return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
