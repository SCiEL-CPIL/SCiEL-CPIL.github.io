import random
import time

N = 10
DEADLINE = 55.0

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        if sorted(sq[i][j] for i in range(n)) != list(range(n)):
            return False
    return True

def _orthogonal(a, b):
    n = len(a)
    seen = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in seen:
                return False
            seen.add(p)
    return True

def _pairwise_orthogonal(squares):
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _orthogonal(squares[i], squares[j]):
                return False
    return True

def construct():
    t0 = time.time()
    
    # Generate the first square: L_0[i][j] = (i + j) % 10
    squares = []
    L0 = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares.append(L0)
    
    # We want to find 8 more squares.
    # We use a randomized search with backtracking for each new square.
    
    def find_next_square(existing_squares, max_time=5.0):
        """
        Finds a Latin square orthogonal to all squares in existing_squares.
        Uses randomized backtracking.
        """
        start_time = time.time()
        n = N
        
        # Initialize the new square with random values?
        # Or use a constraint satisfaction approach.
        # We fill cell by cell.
        # Constraints:
        # 1. Row i must have distinct symbols.
        # 2. Col j must have distinct symbols.
        # 3. For each existing square S_k, the pair (S_k[i][j], new[i][j]) must be unique across the grid.
        
        # We will use a simple randomized greedy with backtracking.
        # To speed up, we can use a heuristic: choose the cell with the fewest valid candidates.
        
        # State:
        # new_sq: 2D list, -1 if empty
        # row_used[i]: set of symbols used in row i
        # col_used[j]: set of symbols used in col j
        # pair_used[k][p]: set of symbols used in pair (S_k[i][j], val) for each k
        
        def backtrack(new_sq, row_used, col_used, pair_used, idx):
            if time.time() - start_time > max_time:
                return None
            
            if idx == n * n:
                return new_sq
            
            i, j = divmod(idx, n)
            
            # Get candidates
            candidates = [s for s in range(n) if s not in row_used[i] and s not in col_used[j]]
            
            # Check pair constraints
            valid_candidates = []
            for s in candidates:
                ok = True
                for k, sq_k in enumerate(existing_squares):
                    p = sq_k[i][j]
                    if s in pair_used[k][p]:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(s)
            
            if not valid_candidates:
                return None
            
            # Shuffle candidates to randomize
            random.shuffle(valid_candidates)
            
            for s in valid_candidates:
                # Place s
                new_sq[i][j] = s
                row_used[i].add(s)
                col_used[j].add(s)
                for k, sq_k in enumerate(existing_squares):
                    p = sq_k[i][j]
                    pair_used[k][p].add(s)
                
                # Recurse
                res = backtrack(new_sq, row_used, col_used, pair_used, idx + 1)
                if res is not None:
                    return res
                
                # Unplace s
                new_sq[i][j] = -1
                row_used[i].remove(s)
                col_used[j].remove(s)
                for k, sq_k in enumerate(existing_squares):
                    p = sq_k[i][j]
                    pair_used[k][p].remove(s)
            
            return None

        # Initialize state
        new_sq = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = [dict() for _ in range(len(existing_squares))]
        for k in range(len(existing_squares)):
            for p in range(n):
                pair_used[k][p] = set()
        
        # We can speed up by using a fixed order or random order of cells.
        # Fixed order is fine for now.
        # To improve, we could pick the next cell dynamically (MRV), but that's more complex.
        # Given N=10, fixed order might be okay if the search space is pruned well.
        
        # Let's try a few random restarts if the first one fails.
        for attempt in range(10):
            if time.time() - start_time > max_time:
                break
            # Reset state for each attempt?
            # Actually, the backtracking above explores the whole tree for the first attempt.
            # If it fails, it returns None.
            # We should just run it. If it's too slow, we need a better heuristic.
            
            res = backtrack(new_sq, row_used, col_used, pair_used, 0)
            if res is not None:
                return res
            
            # If it failed, reset and try again?
            # The backtracking is exhaustive for the first run.
            # If it fails, it means no solution exists (unlikely) or we timed out.
            # If we timed out, we return None.
            
        return None

    # Try to find 8 more squares
    target_count = 9
    while len(squares) < target_count:
        if time.time() - t0 > DEADLINE:
            break
        
        # Find next square
        next_sq = find_next_square(squares, max_time=5.0)
        
        if next_sq is None:
            # If we can't find one, break
            break
        
        squares.append(next_sq)
        
        # Verify
        if not _is_latin(next_sq):
            squares.pop()
            break
        if not _pairwise_orthogonal(squares):
            squares.pop()
            break

    return squares
