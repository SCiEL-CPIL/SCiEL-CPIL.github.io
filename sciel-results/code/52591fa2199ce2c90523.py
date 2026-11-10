import time
import random

DEADLINE = 50.0 # Safety margin

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if len(set(sq[i])) != n:
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if len(set(col)) != n:
            return False
    return True

def _is_orthogonal(a, b):
    n = len(a)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_square_orthogonal_to_existing(existing_squares, n, deadline, seed=42):
    """
    Find a Latin square orthogonal to all existing squares.
    Uses backtracking with MRV heuristic.
    """
    rng = random.Random(seed)
    
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    used_in_row = [set() for _ in range(n)]
    used_in_col = [set() for _ in range(n)]
    used_pairs = [set() for _ in existing_squares]
    
    # Precompute base symbols at each position
    base_symbols = [[existing_squares[k][i][j] for k in range(len(existing_squares))] 
                    for i in range(n) for j in range(n)]
    
    def get_candidates(i, j):
        """Get valid symbols for position (i,j)"""
        candidates = []
        for sym in range(n):
            if sym in used_in_row[i]:
                continue
            if sym in used_in_col[j]:
                continue
            valid = True
            for k, base_sym in enumerate(base_symbols[i * n + j]):
                if (base_sym, sym) in used_pairs[k]:
                    valid = False
                    break
            if valid:
                candidates.append(sym)
        return candidates
    
    def backtrack(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        candidates = get_candidates(i, j)
        
        # MRV heuristic: try candidates with fewer future constraints first
        # Approximate by trying symbols that appear less in remaining columns
        if len(candidates) > 1:
            # Score each candidate by how many positions it eliminates
            scored = []
            for sym in candidates:
                # Count how many future positions would be affected
                future_count = 0
                for fi in range(i, n):
                    for fj in range(n):
                        if fi == i and fj <= j:
                            continue
                        if sym not in used_in_row[fi] and sym not in used_in_col[fj]:
                            base_syms = base_symbols[fi * n + fj]
                            conflict = False
                            for k, bs in enumerate(base_syms):
                                if (bs, sym) in used_pairs[k]:
                                    conflict = True
                                    break
                            if not conflict:
                                future_count += 1
                scored.append((future_count, sym))
            scored.sort(key=lambda x: x[0])
            candidates = [s[1] for s in scored]
        
        for sym in candidates:
            # Place
            sq[i][j] = sym
            used_in_row[i].add(sym)
            used_in_col[j].add(sym)
            for k, base_sym in enumerate(base_symbols[i * n + j]):
                used_pairs[k].add((base_sym, sym))
            
            if backtrack(i, j + 1):
                return True
            
            # Undo
            sq[i][j] = -1
            used_in_row[i].remove(sym)
            used_in_col[j].remove(sym)
            for k, base_sym in enumerate(base_symbols[i * n + j]):
                used_pairs[k].remove((base_sym, sym))
        
        return False

    # Try different seeds for diversity
    for attempt_seed in [seed, seed + 1, seed + 2]:
        rng = random.Random(attempt_seed)
        # Reset state
        sq = [[-1 for _ in range(n)] for _ in range(n)]
        used_in_row = [set() for _ in range(n)]
        used_in_col = [set() for _ in range(n)]
        used_pairs = [set() for _ in existing_squares]
        
        # Symmetry breaking: fix sq[0][0] to a value that doesn't conflict
        valid_first = []
        for sym in range(n):
            valid = True
            for k, base_sym in enumerate(base_symbols[0]):
                if (base_sym, sym) in used_pairs[k]:
                    valid = False
                    break
            if valid:
                valid_first.append(sym)
        
        if not valid_first:
            continue
            
        first_sym = valid_first[0]
        sq[0][0] = first_sym
        used_in_row[0].add(first_sym)
        used_in_col[0].add(first_sym)
        for k, base_sym in enumerate(base_symbols[0]):
            used_pairs[k].add((base_sym, first_sym))
        
        if backtrack(0, 1):
            return sq
    
    return None

def construct():
    start_time = time.time()
    n = 10
    deadline = start_time + DEADLINE
    
    # S1: Cyclic
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [s1]
    
    # Try to find S2
    s2 = _find_square_orthogonal_to_existing(squares, n, deadline, seed=42)
    
    if s2 is not None:
        if _is_latin(s2) and all(_is_orthogonal(s2, sq) for sq in squares):
            squares.append(s2)
    
    # Try to find S3 if we have time
    remaining = deadline - time.time()
    if len(squares) >= 2 and remaining > 5.0:
        s3 = _find_square_orthogonal_to_existing(squares, n, deadline, seed=123)
        if s3 is not None:
            if _is_latin(s3) and all(_is_orthogonal(s3, sq) for sq in squares):
                squares.append(s3)
    
    # Convert to list of lists of ints
    final_res = []
    for sq in squares:
        new_sq = []
        for row in sq:
            new_row = [int(v) for v in row]
            new_sq.append(new_row)
        final_res.append(new_sq)
    
    return final_res
