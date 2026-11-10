"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_orthogonal_square(n, existing_squares, deadline, seed=42):
    """Find a Latin square of order n orthogonal to all existing_squares using backtracking."""
    rng = random.Random(seed)
    sq = [[-1] * n for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    ortho_used = [set() for _ in existing_squares]
    
    def get_candidates(i, j):
        cands = []
        for v in range(n):
            if v in row_used[i] or v in col_used[j]:
                continue
            ok = True
            for k, es in enumerate(existing_squares):
                e_val = es[i][j]
                if (e_val, v) in ortho_used[k]:
                    ok = False
                    break
            if ok:
                cands.append(v)
        return cands
    
    cells = [(i, j) for i in range(n) for j in range(n)]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
        
        best_ci = -1
        best_cands = None
        best_count = n + 1
        
        for ci in range(idx, n * n):
            i, j = cells[ci]
            if sq[i][j] >= 0:
                continue
            cands = get_candidates(i, j)
            if len(cands) < best_count:
                best_count = len(cands)
                best_cands = cands
                best_ci = ci
                if best_count <= 1:
                    break
        
        if best_ci == -1:
            return True
        
        if best_count == 0:
            return False
        
        i, j = cells[best_ci]
        cands = best_cands[:]
        rng.shuffle(cands)
        
        for v in cands:
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pairs_added = []
            for k, es in enumerate(existing_squares):
                e_val = es[i][j]
                pairs_added.append((k, (e_val, v)))
                ortho_used[k].add((e_val, v))
            
            if backtrack(best_ci + 1):
                return True
            
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            for k, pair in pairs_added:
                ortho_used[k].discard(pair)
        
        return False
    
    if backtrack(0):
        return [row[:] for row in sq]
    return None

def _construct_mols_10():
    """Construct MOLS of order 10 using backtracking search."""
    deadline = time.time() + 50
    
    base = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    squares = [base]
    
    for seed in range(500):
        if time.time() > deadline:
            break
        if len(squares) >= 9:
            break
        new_sq = _find_orthogonal_square(10, squares, deadline, seed=seed)
        if new_sq is not None and _is_latin(new_sq):
            if all(_is_orthogonal(new_sq, es) for es in squares):
                squares.append(new_sq)
    
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
