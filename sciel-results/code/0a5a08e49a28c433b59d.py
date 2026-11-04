"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _strassen_2x2_subblock(n, m, p, i0, j0, k0):
    """
    Return 7 (U_row, V_row, W_row) triples for a Strassen 2x2 sub-block
    at offset (i0, j0, k0). Each triple is (U_idx, V_idx, W_idx, coeff) 
    where U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i.
    """
    # Indices for the 2x2 sub-block
    # A rows: i0, i0+1; A cols: j0, j0+1
    # B rows: j0, j0+1; B cols: k0, k0+1
    # C rows: i0, i0+1; C cols: k0, k0+1
    
    # A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
    # B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
    # C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
    
    # W index: C[i,k] -> W[k*n+i]
    W00 = k0 * n + i0
    W01 = (k0+1) * n + i0
    W10 = k0 * n + (i0+1)
    W11 = (k0+1) * n + (i0+1)
    
    # Strassen terms:
    # M1 = (A00 - A11)(B11 - B00)
    # M2 = (A10 + A11)B00
    # M3 = A00(B00 - B11)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M3 + M5
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M4 + M6
    
    terms = []
    
    # M1: U: A[i0,j0]=1, A[i0+1,j0+1]=-1; V: B[j0+1,k0+1]=1, B[j0,k0]=-1; W: C[i0,k0]=1, C[i0+1,k0+1]=1
    terms.append(([(i0*m+j0, 1), ((i0+1)*m+(j0+1), -1)],
                  [(j0*p+k0, -1), ((j0+1)*p+(k0+1), 1)],
                  [(W00, 1), (W11, 1)]))
    
    # M2: U: A[i0+1,j0]=1, A[i0+1,j0+1]=1; V: B[j0,k0]=1; W: C[i0+1,k0]=1, C[i0+1,k0+1]=-1
    terms.append(([((i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
                  [(j0*p+k0, 1)],
                  [(W10, 1), (W11, -1)]))
    
    # M3: U: A[i0,j0]=1; V: B[j0,k0]=1, B[j0+1,k0+1]=-1; W: C[i0,k0]=-1, C[i0,k0+1]=1
    terms.append(([(i0*m+j0, 1)],
                  [(j0*p+k0, 1), ((j0+1)*p+(k0+1), -1)],
                  [(W00, -1), (W01, 1)]))
    
    # M4: U: A[i0+1,j0+1]=1; V: B[j0+1,k0+1]=1, B[j0,k0]=-1; W: C[i0,k0]=1, C[i0+1,k0]=1
    terms.append(([((i0+1)*m+(j0+1), 1)],
                  [(j0*p+k0, -1), ((j0+1)*p+(k0+1), 1)],
                  [(W00, 1), (W10, 1)]))
    
    # M5: U: A[i0,j0]=1, A[i0,j0+1]=1; V: B[j0+1,k0+1]=1; W: C[i0,k0]=1, C[i0+1,k0]=-1
    terms.append(([(i0*m+j0, 1), (i0*m+(j0+1), 1)],
                  [((j0+1)*p+(k0+1), 1)],
                  [(W00, 1), (W10, -1)]))
    
    # M6: U: A[i0+1,j0]=1, A[i0,j0]=-1; V: B[j0,k0]=1, B[j0,k0+1]=1; W: C[i0+1,k0+1]=1
    terms.append(([((i0+1)*m+j0, 1), (i0*m+j0, -1)],
                  [(j0*p+k0, 1), (j0*p+(k0+1), 1)],
                  [(W11, 1)]))
    
    # M7: U: A[i0,j0+1]=1, A[i0+1,j0+1]=-1; V: B[j0+1,k0]=1, B[j0+1,k0+1]=1; W: C[i0,k0+1]=1
    terms.append(([(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)],
                  [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
                  [(W01, 1)]))
    
    return terms


def _verify(n, m, p, U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    This should equal 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True


def _strassen_2x2():
    """
    Classic Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indexes (i,j) in 2x2, V indexes (j,k) in 2x2, W indexes (k,i) in 2x2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11-A22)*(B22-B11)
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = -1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11*(B11-B22)
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22*(B22-B11)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _try_structured_search(n, m, p, target_rank, timeout=20):
    """
    Try to find a decomposition of the given rank using structured patterns.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    if target_rank < n * m * p * 7 // 8:
        return None
    
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with a pattern: start from schoolbook and perturb
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((pn, target_rank), dtype=float)
        
        # First, fill with a schoolbook-like pattern for the first min(target_rank, n*m*p) columns
        base_rank = min(target_rank, n * m * p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        if col < target_rank:
            # Fill remaining columns with small random values
            for c in range(col, target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=nm) * 0.5
                V[:, c] = np.random.choice([-1, 0, 1], size=mp) * 0.5
                W[:, c] = np.random.choice([-1, 0, 1], size=pn) * 0.5
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    dot = np.dot(u_val, v_val * w_val)
                    if abs(dot - 1.0) > 1e-8:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U],
                "V": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V],
                "W": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
            }
    
    return None


def _construct_hybrid(n, m, p):
    """
    Construct a hybrid decomposition using Strassen for 2x2x2 sub-blocks
    and schoolbook for the rest. This gives a guaranteed valid decomposition
    with rank < n*m*p.
    """
    rank = n * m * p  # Start with schoolbook rank, will reduce
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    used_strassen = set()
    
    # Try to place non-overlapping 2x2x2 Strassen blocks
    # A Strassen block at (i0,j0,k0) covers:
    #   U indices: i0*m+j0, i0*m+(j0+1), (i0+1)*m+j0, (i0+1)*m+(j0+1)
    #   V indices: j0*p+k0, j0*p+(k0+1), (j0+1)*p+k0, (j0+1)*p+(k0+1)
    #   W indices: k0*n+i0, (k0+1)*n+i0, k0*n+(i0+1), (k0+1)*n+(i0+1)
    #   (i,j,k) triples: all (i0+di, j0+dj, k0+dk) for di,dj,dk in {0,1}
    
    strassen_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check if this 2x2x2 block overlaps with any existing Strassen block
                overlap = False
                for (si0, sj0, sk0) in strassen_blocks:
                    # Check if any (i,j,k) in this block is in the other block
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                i, j, k = i0+di, j0+dj, k0+dk
                                if si0 <= i <= si0+1 and sj0 <= j <= sj0+1 and sk0 <= k <= sk0+1:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        if overlap:
                            break
                    if overlap:
                        break
                if not overlap:
                    strassen_blocks.append((i0, j0, k0))
    
    # For each Strassen block, use 7 columns instead of 8
    # For remaining (i,j,k) triples, use schoolbook 1 column each
    
    # First, identify all (i,j,k) triples covered by Strassen blocks
    strassen_triples = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_triples.add((i0+di, j0+dj, k0+dk))
    
    # Fill Strassen blocks
    for (i0, j0, k0) in strassen_blocks:
        terms = _strassen_2x2_subblock(n, m, p, i0, j0, k0)
        for u_terms, v_terms, w_terms in terms:
            for (u_idx, coeff) in u_terms:
                U[u_idx][col] = coeff
            for (v_idx, coeff) in v_terms:
                V[v_idx][col] = coeff
            for (w_idx, coeff) in w_terms:
                W[w_idx][col] = coeff
            col += 1
    
    # Fill remaining triples with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_triples:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim to actual rank used
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}


def _try_random_search(n, m, p, target_rank, timeout=15):
    """
    Pure random search for a decomposition of the given rank.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(200):
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice([-1, 0, 1], size=(nm, target_rank))
        V = np.random.choice([-1, 0, 1], size=(mp, target_rank))
        W = np.random.choice([-1, 0, 1], size=(pn, target_rank))
        
        # Quick check: sum of products should be 1 for each (i,j,k)
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    dot = np.dot(u_val, v_val * w_val)
                    if abs(dot - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) for x in row] for row in U],
                "V": [[int(x) for x in row] for row in V],
                "W": [[int(x) for x in row] for row in W]
            }
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid Strassen+schoolbook construction
    if time.time() < deadline - 5:
        hybrid = _construct_hybrid(n, m, p)
        U, V, W = hybrid["U"], hybrid["V"], hybrid["W"]
        hybrid_rank = len(U[0]) if U else 0
        if _verify(n, m, p, U, V, W):
            if hybrid_rank < best_rank:
                best = hybrid
                best_rank = hybrid_rank
    
    # Try to find a lower rank decomposition via search
    for target_rank in range(40, best_rank):
        if time.time() > deadline - 5:
            break
        result = _try_structured_search(n, m, p, target_rank, timeout=5)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(n, m, p, U, V, W):
                best = result
                best_rank = target_rank
                break
    
    # If still schoolbook rank, try random search
    if best_rank == n * m * p:
        for target_rank in range(45, best_rank):
            if time.time() > deadline - 5:
                break
            result = _try_random_search(n, m, p, target_rank, timeout=5)
            if result is not None:
                U, V, W = result["U"], result["V"], result["W"]
                if _verify(n, m, p, U, V, W):
                    best = result
                    best_rank = target_rank
                    break
    
    return best
# EVOLVE-BLOCK-END
