import time
import sys
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    # 1. Create the first two Latin Squares (Cyclic)
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    S1 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    def is_valid_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            for j in range(n):
                col = [sq[k][j] for k in range(n)]
                if set(col) != set(range(n)):
                    return False
        return True

    def check_orthogonality(sq_a, sq_b):
        seen = set()
        for i in range(n):
            for j in range(n):
                p = (sq_a[i][j], sq_b[i][j])
                if p in seen:
                    return False
                seen.add(p)
        return True

    # Verify S0 and S1 are valid and orthogonal
    if not (is_valid_latin(S0) and is_valid_latin(S1) and check_orthogonality(S0, S1)):
        # Fallback to single square if something went wrong (unlikely)
        return [S0]

    result = [S0, S1]
    
    # Backtracking solver for a new square orthogonal to a list of existing squares
    def find_orthogonal_latin(existing_squares):
        sq = [[-1 for _ in range(n)] for _ in range(n)]
        
        row_masks = [0] * n
        col_masks = [0] * n
        used_pairs = [set() for _ in existing_squares]
        
        def is_consistent(i, j, val):
            if (row_masks[i] >> val) & 1:
                return False
            if (col_masks[j] >> val) & 1:
                return False
            
            for k, ex_sq in enumerate(existing_squares):
                ex_val = ex_sq[i][j]
                pair = (ex_val, val)
                if pair in used_pairs[k]:
                    return False
            return True

        def backtrack(i, j):
            if time.time() > deadline:
                return False
            
            if i == n:
                return True
            
            ni = i
            nj = j + 1
            if nj == n:
                ni = i + 1
                nj = 0
            
            candidates = []
            for val in range(n):
                if is_consistent(i, j, val):
                    candidates.append(val)
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for val in candidates:
                sq[i][j] = val
                row_masks[i] |= (1 << val)
                col_masks[j] |= (1 << val)
                for k in range(len(existing_squares)):
                    used_pairs[k].add((existing_squares[k][i][j], val))
                
                if backtrack(ni, nj):
                    return True
                
                sq[i][j] = -1
                row_masks[i] &= ~(1 << val)
                col_masks[j] &= ~(1 << val)
                for k in range(len(existing_squares)):
                    used_pairs[k].discard((existing_squares[k][i][j], val))
            
            return False

        # Try to find S2
        # Reset random seed for reproducibility in evaluation? 
        # random.seed(42) 
        if backtrack(0, 0):
            return sq
        else:
            return None

    # Try to find S2
    S2 = find_orthogonal_latin([S0, S1])
    if S2:
        if is_valid_latin(S2) and check_orthogonality(S0, S2) and check_orthogonality(S1, S2):
            result.append(S2)

    # Try to find S3 if time permits and S2 was found
    if time.time() < deadline - 10.0 and len(result) == 3:
        S3 = find_orthogonal_latin(result)
        if S3:
            if is_valid_latin(S3) and all(check_orthogonality(S3, sq) for sq in result):
                result.append(S3)

    # Final validation
    valid_result = []
    for sq in result:
        if not is_valid_latin(sq):
            continue
        ok = True
        for v_sq in valid_result:
            if not check_orthogonality(sq, v_sq):
                ok = False
                break
        if ok:
            valid_result.append(sq)
            
    if not valid_result:
        return [S0]

    # Ensure types are pure Python lists and ints
    final_result = [
        [[int(val) for val in row] for row in sq]
        for sq in valid_result
    ]
    
    return final_result
