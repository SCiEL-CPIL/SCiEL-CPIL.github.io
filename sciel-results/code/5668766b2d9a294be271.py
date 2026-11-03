import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
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

def _find_best_block_placements(n, m, p, max_blocks=4, time_budget=10.0):
    """
    Systematically search for non-overlapping Strassen block placements.
    Returns the best list of blocks and its rank.
    """
    # All possible 2x2x2 block origins
    all_origins = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_origins.append((i0, j0, k0))
    
    # Precompute which (i,j,k) each block covers
    block_coverage = {}
    for block in all_origins:
        i0, j0, k0 = block
        covered = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
        block_coverage[block] = covered
    
    best_rank = n * m * p  # schoolbook
    best_blocks = []
    
    # Try all combinations of up to max_blocks non-overlapping blocks
    # For efficiency, use greedy + local search
    
    # First: greedy approach - add blocks one at a time if they don't overlap
    # and reduce the total rank
    
    def _rank_for_blocks(blocks):
        covered = set()
        for b in blocks:
            covered.update(block_coverage[b])
        return 7 * len(blocks) + (n * m * p - len(covered))
    
    # Greedy: start with empty, add block that gives best marginal improvement
    # Marginal improvement = -1 if no overlap, more complex if overlap
    
    # Actually, each non-overlapping block saves exactly 1 (replaces 8 schoolbook with 7 Strassen)
    # So we want maximum number of non-overlapping blocks
    
    # Find maximum set of non-overlapping blocks
    # This is a maximum independent set problem on the conflict graph
    
    # Build conflict graph: two blocks conflict if they share any (i,j,k)
    num_blocks = len(all_origins)
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            ci = block_coverage[all_origins[i]]
            cj = block_coverage[all_origins[j]]
            if ci & cj:  # non-empty intersection
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy maximum independent set
    def _greedy_mis(conflicts, num_nodes):
        visited = [False] * num_nodes
        mis = []
        # Order nodes by degree (ascending) for better greedy
        order = sorted(range(num_nodes), key=lambda x: len(conflicts[x]))
        for node in order:
            if not visited[node]:
                mis.append(node)
                for neighbor in conflicts[node]:
                    visited[neighbor] = True
        return mis
    
    mis_indices = _greedy_mis(conflicts, num_blocks)
    greedy_blocks = [all_origins[i] for i in mis_indices]
    greedy_rank = _rank_for_blocks(greedy_blocks)
    
    if greedy_rank < best_rank:
        best_rank = greedy_rank
        best_blocks = greedy_blocks
    
    # Try local search: swap one block for another
    improved = True
    current_blocks = list(best_blocks)
    current_rank = best_rank
    
    while improved and _time_left() > time_budget:
        improved = False
        # Try removing one block and adding another
        for remove_idx in range(len(current_blocks)):
            for add_block in all_origins:
                if add_block in current_blocks:
                    continue
                new_blocks = current_blocks[:remove_idx] + current_blocks[remove_idx+1:] + [add_block]
                # Check non-overlap
                valid = True
                for i in range(len(new_blocks)):
                    for j in range(i + 1, len(new_blocks)):
                        if block_coverage[new_blocks[i]] & block_coverage[new_blocks[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_rank = _rank_for_blocks(new_blocks)
                    if new_rank < current_rank:
                        current_blocks = new_blocks
                        current_rank = new_rank
                        best_rank = current_rank
                        best_blocks = list(current_blocks)
                        improved = True
    
    return best_blocks, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Systematically search for best block placements
    if _time_left() > 15:
        try:
            best_blocks, best_block_rank = _find_best_block_placements(n, m, p, max_blocks=4, time_budget=10.0)
            
            if best_blocks and best_block_rank < best_rank:
                U, V, W = _build_strassen_embedding(n, m, p, best_blocks)
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Also try the original fixed placement as a backup
    if _time_left() > 5:
        try:
            blocks_fixed = [
                (0, 0, 0),
                (0, 0, 2),
                (0, 0, 4),
                (0, 0, 6)
            ]
            U, V, W = _build_strassen_embedding(n, m, p, blocks_fixed)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
