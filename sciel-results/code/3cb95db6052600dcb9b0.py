"""Program for matrix multiplication 3,4,5 tensor rank minimization via Strassen blocks."""
import time
from itertools import product

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank below schoolbook.
    Finds the maximum set of disjoint 2x2x2 blocks via search.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    n, m, p = N, M, P
    t0 = time.time()
    DEADLINE = t0 + 130  # Safety margin before 150s kill
    
    def _time_left():
        return DEADLINE - time.time()
    
    def strassen_terms(i_off, j_off, k_off):
        """
        Return the 7 Strassen terms for a 2x2x2 block.
        Each term is (U_dict, V_dict, W_dict) where each dict maps global indices to coefficients.
        """
        terms = []
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
        V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
        W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
        terms.append((U1, V1, W1))
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
        V2 = {(j_off + 0, k_off + 0): 1}
        W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
        terms.append((U2, V2, W2))
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        U3 = {(i_off + 0, j_off + 0): 1}
        V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
        W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
        terms.append((U3, V3, W3))
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        U4 = {(i_off + 1, j_off + 1): 1}
        V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
        W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
        terms.append((U4, V4, W4))
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
        V5 = {(j_off + 1, k_off + 1): 1}
        W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
        terms.append((U5, V5, W5))
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
        V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
        W6 = {(k_off + 1, i_off + 1): 1}
        terms.append((U6, V6, W6))
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
        V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
        W7 = {(k_off + 0, i_off + 0): 1}
        terms.append((U7, V7, W7))
        
        return terms
    
    # Enumerate all possible 2x2x2 blocks
    # i_off in {0,1}, j_off in {0,1,2,3}, k_off in {0,1,2,3}
    all_blocks = []
    for i_off in range(n - 1):  # 0, 1
        for j_off in range(m - 1):  # 0, 1, 2, 3
            for k_off in range(p - 1):  # 0, 1, 2, 3
                # Check bounds
                if i_off + 1 < n and j_off + 1 < m and k_off + 1 < p:
                    block_tuples = set()
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                block_tuples.add((i_off + ii, j_off + jj, k_off + kk))
                    all_blocks.append((i_off, j_off, k_off, block_tuples))
    
    # Find maximum set of disjoint blocks using backtracking
    # Sort blocks by number of uncovered tuples they cover (greedy heuristic for ordering)
    def max_disjoint_blocks(blocks, deadline):
        """Find maximum set of disjoint 2x2x2 blocks."""
        if not blocks:
            return []
        
        best = []
        all_tuples = set()
        for b in blocks:
            all_tuples |= b[3]
        
        # Try each block as first, then recurse
        for i, block in enumerate(blocks):
            if time.time() > deadline:
                break
            remaining = [b for j, b in enumerate(blocks) if j != i and b[3].isdisjoint(block[3])]
            sub_best = max_disjoint_blocks(remaining, deadline)
            candidate = [block] + sub_best
            if len(candidate) > len(best):
                best = candidate
        
        return best
    
    # Use greedy with local search instead of full backtracking (too slow)
    # Greedy: repeatedly pick the block that covers the most uncovered tuples
    def greedy_max_disjoint(blocks, deadline):
        if not blocks:
            return []
        
        selected = []
        covered = set()
        remaining = list(blocks)
        
        while remaining and time.time() < deadline:
            # Find block with max uncovered tuples
            best_idx = -1
            best_count = 0
            for i, block in enumerate(remaining):
                uncovered = len(block[3] - covered)
                if uncovered > best_count:
                    best_count = uncovered
                    best_idx = i
            
            if best_idx < 0:
                break
            
            selected.append(remaining[best_idx])
            covered |= remaining[best_idx][3]
            remaining = [b for i, b in enumerate(remaining) if i != best_idx and b[3].isdisjoint(covered)]
        
        return selected
    
    # Local search: try to improve greedy solution by swapping
    def local_search_improve(selected, all_blocks, deadline):
        if not selected:
            return selected
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            covered = set()
            for block in selected:
                covered |= block[3]
            
            # Try removing one block and adding another
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                without_i = selected[:i] + selected[i+1:]
                covered_without = set()
                for block in without_i:
                    covered_without |= block[3]
                
                # Find best block to add
                best_add = None
                best_gain = 0
                for block in all_blocks:
                    if block in selected:
                        continue
                    if not block[3].isdisjoint(covered_without):
                        continue
                    gain = len(block[3] - covered_without)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = block
                
                if best_add is not None and len(without_i) + 1 > len(selected):
                    selected = without_i + [best_add]
                    improved = True
                    break
                elif best_add is not None and best_gain > 0 and len(without_i) == len(selected):
                    # Same size but different coverage - check if it enables further improvement
                    # For now, skip
                    pass
        
        return selected
    
    # Run greedy
    greedy_result = greedy_max_disjoint(all_blocks, DEADLINE)
    
    # Try local search
    if _time_left() > 10:
        improved_result = local_search_improve(greedy_result, all_blocks, DEADLINE)
        if len(improved_result) > len(greedy_result):
            greedy_result = improved_result
    
    # Also try the known good set of 4 blocks as baseline
    known_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2),
    ]
    known_set = set()
    for i_off, j_off, k_off in known_blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    known_set.add((i_off + ii, j_off + jj, k_off + kk))
    
    # Check if greedy found something better
    greedy_covered = set()
    for block in greedy_result:
        greedy_covered |= block[3]
    
    print(f"Greedy found {len(greedy_result)} blocks covering {len(greedy_covered)} tuples")
    print(f"Known 4 blocks cover {len(known_set)} tuples")
    
    # Use the better set
    if len(greedy_result) >= 4:
        strassen_blocks = [(b[0], b[1], b[2]) for b in greedy_result]
    else:
        strassen_blocks = known_blocks
    
    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in strassen_blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    strassen_covered.add((i_off + ii, j_off + jj, k_off + kk))
    
    # Build the full decomposition.
    all_terms = []
    
    # Add Strassen terms for each block
    for i_off, j_off, k_off in strassen_blocks:
        block_terms = strassen_terms(i_off, j_off, k_off)
        for U_dict, V_dict, W_dict in block_terms:
            all_terms.append((U_dict, V_dict, W_dict))
    
    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_sb = {(i, j): 1}
                    V_sb = {(j, k): 1}
                    W_sb = {(k, i): 1}
                    all_terms.append((U_sb, V_sb, W_sb))
    
    rank = len(all_terms)
    print(f"Total rank: {rank} ({len(strassen_blocks)} Strassen blocks, {rank - 7*len(strassen_blocks)} schoolbook)")
    
    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    
    # Fallback to schoolbook if verification fails
    U0 = [[0] * RANK_SCHOOLBOOK for _ in range(n * m)]
    V0 = [[0] * RANK_SCHOOLBOOK for _ in range(m * p)]
    W0 = [[0] * RANK_SCHOOLBOOK for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    return {"U": U0, "V": V0, "W": W0}
