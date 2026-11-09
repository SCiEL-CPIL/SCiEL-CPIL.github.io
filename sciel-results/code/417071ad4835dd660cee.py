import time
import random

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0  # Leave margin for 60s limit

    # S0 is the cyclic Latin square
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Function to check if two squares are orthogonal
    def is_orthogonal(A, B):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (A[i][j], B[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    # Function to count conflicts (number of pairs appearing > 1)
    def count_conflicts(A, B):
        pairs = {}
        for i in range(n):
            for j in range(n):
                p = (A[i][j], B[i][j])
                pairs[p] = pairs.get(p, 0) + 1
        conflicts = 0
        for count in pairs.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    # Generate a random Latin square
    def random_latin_square():
        # Start with cyclic and randomize row and column permutations
        L = [[(i + j) % n for j in range(n)] for i in range(n)]
        # Random permutation of rows
        rows = list(range(n))
        random.shuffle(rows)
        L = [L[i] for i in rows]
        # Random permutation of columns
        cols = list(range(n))
        random.shuffle(cols)
        L = [[L[i][j] for j in cols] for i in range(n)]
        # Random permutation of symbols
        syms = list(range(n))
        random.shuffle(syms)
        # Create mapping old->new
        mapping = {old: new for new, old in enumerate(syms)}
        L = [[mapping[L[i][j]] for j in range(n)] for i in range(n)]
        return L

    # Backtracking solver for a second square orthogonal to S0
    # This is more robust than local search for this specific constraint structure
    def find_orthogonal_square(base):
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Precompute orthogonality classes (diagonals)
        # diag[k] = list of (i, j) such that base[i][j] == k
        diags = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                k = base[i][j]
                diags[k].append((i, j))
                
        # Helper to check if placing val at (i, j) is valid
        def is_valid(i, j, val):
            # Check row
            for jj in range(n):
                if grid[i][jj] == val:
                    return False
            # Check col
            for ii in range(n):
                if grid[ii][j] == val:
                    return False
            # Check orthogonality class
            k = base[i][j]
            for (ii, jj) in diags[k]:
                if grid[ii][jj] == val:
                    return False
            return True

        # Find next empty cell
        def find_empty():
            # Heuristic: choose the cell with the fewest valid options
            best_cell = None
            min_options = n + 1
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == -1:
                        count = 0
                        for val in range(n):
                            if is_valid(i, j, val):
                                count += 1
                        if count < min_options:
                            min_options = count
                            best_cell = (i, j)
                            if min_options == 0:
                                return best_cell
            return best_cell

        def backtrack():
            if time.time() - start_time > timeout:
                return False
            
            cell = find_empty()
            if cell is None:
                return True # Success
            
            i, j = cell
            # Shuffle values for randomization to escape bad paths
            values = list(range(n))
            random.shuffle(values)
            
            for val in values:
                if is_valid(i, j, val):
                    grid[i][j] = val
                    if backtrack():
                        return True
                    grid[i][j] = -1
            return False

        if backtrack():
            return grid
        return None

    found_squares = [S0]
    
    # Try to find S1 using backtracking
    S1 = find_orthogonal_square(S0)
    if S1 is not None:
        found_squares.append(S1)
    
    # Verify orthogonality of all found squares
    valid_squares = [found_squares[0]]
    for i in range(1, len(found_squares)):
        ok = True
        for j in range(i):
            if not is_orthogonal(found_squares[i], found_squares[j]):
                ok = False
                break
        if ok:
            valid_squares.append(found_squares[i])
            
    return valid_squares
