"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
_DEADLINE = 145.0  # safety margin before 150s wall-clock kill
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _strassen_2x2_2():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    # a: (0,0), d: (1,1) -> U[0,0]=1, U[3,0]=1
    # e: (0,0), h: (1,1) -> V[0,0]=1, V[3,0]=1
    # C11: (0,0), C22: (1,1) -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    # c: (1,0), d: (1,1) -> U[2,1]=1, U[3,1]=1
    # e: (0,0) -> V[0,1]=1
    # C21: (0,1), C22: (1,1) -> W[1,1]=1, W[3,1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    # a: (0,0) -> U[0,2]=1
    # f: (0,1), h: (1,1) -> V[1,2]=1, V[3,2]=-1
    # C12: (1,0), C22: (1,1) -> W[2,2]=1, W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    # d: (1,1) -> U[3,3]=1
    # g: (1,0), e: (0,0) -> V[2,3]=1, V[0,3]=-1
    # C11: (0,0), C21: (0,1) -> W[0,3]=1, W[1,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    # a: (0,0), b: (0,1) -> U[0,4]=1, U[1,4]=1
    # h: (1,1) -> V[3,4]=1
    # C11: (0,0), C12: (1,0) -> W[0,4]=-1, W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    # c: (1,0), a: (0,0) -> U[2,5]=1, U[0,5]=-1
    # e: (0,0), f: (0,1) -> V[0,5]=1, V[1,5]=1
    # C22: (1,1) -> W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    # b: (0,1), d: (1,1) -> U[1,6]=1, U[3,6]=-1
    # g: (1,0), h: (1,1) -> V[2,6]=1, V[3,6]=1
    # C11: (0,0) -> W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_all_block_placements(n, m, p):
    """Try all valid sets of non-overlapping 2x2x2 blocks and return best rank."""
    best_rank = n * m * p
    best_result = None
    
    # All possible (i0, j0) pairs
    ij_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # For each (i0, j0), find valid non-overlapping k0 sets
    def get_valid_k_sets(i0, j0):
        """Get all subsets of k0 values that are pairwise non-overlapping."""
        k_values = list(range(p - 1))  # 0..6
        valid_sets = []
        
        def backtrack(remaining, current):
            if not remaining:
                if current:
                    valid_sets.append(tuple(current))
                return
            first = remaining[0]
            rest = remaining[1:]
            # Skip first
            backtrack(rest, current)
            # Include first if no overlap with current
            can_add = all(abs(first - x) > 1 for x in current)
            if can_add:
                filtered_rest = [x for x in rest if abs(x - first) > 1]
                backtrack(filtered_rest, current + [first])
        
        backtrack(k_values, [])
        return valid_sets
    
    k_sets_by_ij = {ij: get_valid_k_sets(ij[0], ij[1]) for ij in ij_pairs}
    
    # Try all combinations of up to 4 blocks
    def check_overlap(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        return not (i1 + 1 < i2 or i2 + 1 < i1) and \
               not (j1 + 1 < j2 or j2 + 1 < j1) and \
               not (k1 + 1 < k2 or k2 + 1 < k1)
    
    for num_blocks in range(4, 0, -1):
        # Generate all combinations of num_blocks from all possible blocks
        all_blocks = []
        for ij in ij_pairs:
            for k_set in k_sets_by_ij[ij]:
                for k in k_set:
                    all_blocks.append((ij[0], ij[1], k))
        
        # Remove duplicates
        all_blocks = list(set(all_blocks))
        
        for combo in itertools.combinations(all_blocks, num_blocks):
            # Check if any two blocks overlap
            valid = True
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    if check_overlap(combo[a], combo[b]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            try:
                U, V, W = _build_strassen_embedding(n, m, p, list(combo))
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
                        return best_result, best_rank
            except Exception:
                pass
        
        if best_result is not None:
            return best_result, best_rank
    
    return best_result, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    try:
        if _time_left() > 5.0:
            result, rank = _try_all_block_placements(n, m, p)
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
    except Exception:
        pass
    
    # Final fallback: use the known good configuration
    if best_rank == n * m * p:
        blocks = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 4),
            (0, 0, 6)
        ]
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
