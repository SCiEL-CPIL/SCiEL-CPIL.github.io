import time
import random

def construct():
    n = 10
    deadline = time.time() + 55
    
    # 1. Generate 4 MOLS using algebraic construction on Z2 x Z5
    # Elements are (a, b) where a in {0,1}, b in {0..4}
    # Index i = 5*a + b
    # L_k[i][j] = encode(a_i + a_j, b_i + k*b_j % 5) for k in 1..4
    
    def encode(a, b):
        return 5 * a + b

    def decode(x):
        return x // 5, x % 5

    base_squares = []
    for k in range(1, 5):
        L = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            ai, bi = decode(i)
            for j in range(n):
                aj, bj = decode(j)
                a_part = (ai + aj) % 2
                b_part = (bi + k * bj) % 5
                L[i][j] = encode(a_part, b_part)
        base_squares.append(L)
        
    # Verify base squares
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    valid_squares = []
    for sq in base_squares:
        if is_latin(sq):
            # Check orthogonality against all existing
            ortho = True
            for existing in valid_squares:
                if not are_orthogonal(sq, existing):
                    ortho = False
                    break
            if ortho:
                valid_squares.append(sq)

    # We have 4 squares. Now try to find more via backtracking.
    # We will try to find a 5th square orthogonal to the first 4.
    # Then a 6th orthogonal to the first 5, etc.
    
    def find_next_mols(existing_squares, timeout=10):
        # Backtracking search for a new Latin square orthogonal to all existing
        new_sq = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        # pair_used[k][i][j] is not needed, we check pairs on the fly or maintain sets
        # For efficiency, maintain a set of used pairs for each existing square
        pair_sets = [set() for _ in existing_squares]
        
        # Pre-compute constraints
        # For each cell (i,j), the value v must satisfy:
        # 1. v not in row_used[i]
        # 2. v not in col_used[j]
        # 3. For each existing square s, (s[i][j], v) not in pair_sets[s_idx]
        
        def is_valid(i, j, val):
            if val in row_used[i]: return False
            if val in col_used[j]: return False
            for idx, sq in enumerate(existing_squares):
                if (sq[i][j], val) in pair_sets[idx]:
                    return False
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n * n:
                return True
            i = idx // n
            j = idx % n
            vals = list(range(n))
            random.shuffle(vals)
            for val in vals:
                if is_valid(i, j, val):
                    new_sq[i][j] = val
                    row_used[i].add(val)
                    col_used[j].add(val)
                    # Add to pair sets
                    pairs_added = []
                    valid = True
                    for idx2, sq in enumerate(existing_squares):
                        p = (sq[i][j], val)
                        if p in pair_sets[idx2]:
                            valid = False
                            break
                        pair_sets[idx2].add(p)
                        pairs_added.append((idx2, p))
                    
                    if valid and backtrack(idx + 1):
                        return True
                    else:
                        # Undo
                        new_sq[i][j] = -1
                        row_used[i].remove(val)
                        col_used[j].remove(val)
                        for idx2, p in pairs_added:
                            pair_sets[idx2].remove(p)
            return False

        # Start search
        if backtrack(0):
            return new_sq
        return None

    # Try to add squares one by one
    max_attempts = 5
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        next_sq = find_next_mols(valid_squares, timeout=10)
        if next_sq:
            # Verify
            if is_latin(next_sq):
                ortho = True
                for existing in valid_squares:
                    if not are_orthogonal(next_sq, existing):
                        ortho = False
                        break
                if ortho:
                    valid_squares.append(next_sq)
                else:
                    break
            else:
                break
        else:
            break

    # Return the best set found
    if not valid_squares:
        # Fallback to 1 square
        L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [[[int(L1[i][j]) for j in range(n)] for i in range(n)]]
        
    result = []
    for sq in valid_squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    return result
