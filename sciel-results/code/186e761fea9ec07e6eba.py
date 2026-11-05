import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Generate all possible 2x2x2 blocks
    # Block is (i0, i1, j0, j1, k0, k1)
    all_blocks = []
    for i0, i1 in itertools.combinations(range(n), 2):
        for j0, j1 in itertools.combinations(range(m), 2):
            for k0, k1 in itertools.combinations(range(p), 2):
                all_blocks.append((i0, i1, j0, j1, k0, k1))
                
    # Precompute covers
    block_covers = []
    for b in all_blocks:
        i0, i1, j0, j1, k0, k1 = b
        covers = set()
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covers.add((ii, jj, kk))
        block_covers.append(frozenset(covers))
        
    # Check for 7 disjoint blocks
    # We use a backtracking search
    # To speed up, we can order blocks and prune early
    # A block is compatible with a set of selected blocks if it's disjoint from all of them
    
    # Heuristic: Try blocks that cover "central" or "hard" parts first?
    # Or just random restarts?
    # Given small size, we can try all combinations of 7? 
    # C(378, 7) is too large.
    # But backtracking with pruning should work.
    
    # Let's try to find 7 blocks.
    # If found, rank = 77.
    # If not, rank = 78.
    
    found_7 = False
    best_blocks_7 = None
    
    # To make backtracking efficient, we can sort blocks by index or random
    # Let's try a few random permutations
    import random
    rng = random.Random(42)
    
    indices = list(range(len(all_blocks)))
    
    # Try to find a set of 7
    # We'll use a recursive function
    def backtrack(selected, start_idx, current_covers):
        if len(selected) == 7:
            return True, selected
        if time.time() - _t0 > _DEADLINE - 10:
            return False, None
            
        for i in range(start_idx, len(indices)):
            idx = indices[i]
            cover = block_covers[idx]
            # Check if disjoint
            if not (cover & current_covers):
                new_selected = selected + [idx]
                new_covers = current_covers | cover
                found, res = backtrack(new_selected, i + 1, new_covers)
                if found:
                    return True, res
        return False, None
    
    # Try a few random orderings
    for trial in range(100):
        if time.time() - _t0 > _DEADLINE - 10:
            break
        rng.shuffle(indices)
        found, res = backtrack([], 0, frozenset())
        if found:
            found_7 = True
            best_blocks_7 = res
            break
            
    if found_7:
        blocks_to_use = [all_blocks[idx] for idx in best_blocks_7]
        rank = 77
    else:
        # Use the known 6-block construction
        # Blocks:
        # 1. (0,1, 0,1, 0,1)
        # 2. (1,2, 2,3, 0,1)
        # 3. (0,1, 0,1, 2,3)
        # 4. (1,2, 2,3, 2,3)
        # 5. (0,1, 0,1, 4,5)
        # 6. (1,2, 2,3, 4,5)
        # Wait, let's verify disjointness of this set
        # B1: I{0,1} J{0,1} K{0,1}
        # B2: I{1,2} J{2,3} K{0,1} -> Shares I=1, K{0,1} with B1. J disjoint. OK.
        # B3: I{0,1} J{0,1} K{2,3} -> Shares I{0,1}, J{0,1} with B1. K disjoint. OK.
        # B4: I{1,2} J{2,3} K{2,3} -> Shares I=1, K{2,3} with B3. J disjoint. OK.
        # B5: I{0,1} J{0,1} K{4,5} -> Shares I{0,1}, J{0,1} with B1, B3. K disjoint. OK.
        # B6: I{1,2} J{2,3} K{4,5} -> Shares I=1, K{4,5} with B5. J disjoint. OK.
        # Also check B2 vs B4: Shares I=1, J{2,3}. K disjoint. OK.
        # B2 vs B6: Shares I=1, J{2,3}. K disjoint. OK.
        # B4 vs B6: Shares I{1,2}, J{2,3}. K disjoint. OK.
        # All disjoint.
        
        blocks_to_use = [
            (0, 1, 0, 1, 0, 1),
            (1, 2, 2, 3, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (1, 2, 2, 3, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (1, 2, 2, 3, 4, 5)
        ]
        rank = 78
        
    # Build decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    covered = set()
    
    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # Standard 7-term Strassen for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1
        W[k1*n+i1][col_start] = 1

        # M2 = (A10+A11)B00
        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1
        W[k1*n+i1][col_start+1] = -1

        # M3 = A00(B10-B00)
        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1
        W[k1*n+i1][col_start+2] = 1

        # M4 = A11(B01-B10)
        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1
        W[k0*n+i1][col_start+3] = 1

        # M5 = (A00+A01)B11
        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1
        W[k1*n+i0][col_start+4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1

        return col_start + 7

    for (i0, i1, j0, j1, k0, k1) in blocks_to_use:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0*m+j0][col] = 1
                V[j0*p+k0][col] = 1
                W[k0*n+i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
