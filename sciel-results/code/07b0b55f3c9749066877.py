import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s wall-clock

    n = 10
    symbols = list(range(n))
    random.seed(42)

    def check_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != symbols:
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != symbols:
                return False
        return True

    def is_orthogonal(a, b):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    def is_orthogonal_to_list(a, squares):
        for b in squares:
            if not is_orthogonal(a, b):
                return False
        return True

    def try_construct_search(squares, max_iters=100000):
        """Try to find a new square orthogonal to all in squares using randomized search."""
        # Fix first row and first column to standard to reduce symmetry
        new_sq = [[0]*n for _ in range(n)]
        
        # Fix first column: new_sq[i][0] = i (standard)
        for i in range(n):
            new_sq[i][0] = i
        
        # Fix first row: new_sq[0][j] = j (standard)
        for j in range(n):
            new_sq[0][j] = j

        # Now we need to fill the rest
        # For each row i (1..n-1), new_sq[i][0] is fixed to i
        # For each col j (1..n-1), new_sq[0][j] is fixed to j
        # We need to fill new_sq[i][j] for i,j in 1..n-1

        # For orthogonality with existing squares, for each cell (i,j),
        # the pair (new_sq[i][j], a[k][i][j]) must be unique across all cells
        # for each square a[k].
        
        # This is complex. Let's use a simpler approach:
        # Generate random Latin squares and check orthogonality.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Generate a random Latin square
            # Start with identity, then apply random row permutations and column permutations
            # Actually, let's use a simpler method:
            # Start with the addition table, then shuffle rows and columns randomly
            # But we need to ensure it's different from existing squares
            
            # Method: Start with a random permutation of rows of the addition table
            base = [[(i+j) % n for j in range(n)] for i in range(n)]
            
            # Apply random column permutation
            cols = list(range(n))
            random.shuffle(cols)
            new_sq = [[base[i][cols[j]] for j in range(n)] for i in range(n)]
            
            # Apply random row permutation
            rows = list(range(n))
            random.shuffle(rows)
            new_sq = [new_sq[rows[i]] for i in range(n)]
            
            # Check if it's a valid Latin square (it should be)
            # Check orthogonality with all existing squares
            if all(is_orthogonal(new_sq, s) for s in squares):
                # Check it's not identical to any existing square
                if not any(new_sq == s for s in squares):
                    return new_sq
        
        return None

    # Start with the addition table
    sq1 = [[(i+j) % n for j in range(n)] for i in range(n)]
    squares = [sq1]

    # Try to find orthogonal squares one by one
    max_squares_to_find = 9  # Maximum possible is 9 (since n-1 = 9 for n=10)
    
    for target_count in range(2, max_squares_to_find + 1):
        if time.time() > deadline - 5.0:  # Leave 5s buffer
            break
        
        found = False
        attempts = 0
        while not found and attempts < 5000:
            if time.time() > deadline - 5.0:
                break
            
            attempts += 1
            candidate = try_construct_search(squares, max_iters=100)
            if candidate is not None:
                squares.append(candidate)
                found = True
        
        if not found:
            # Try a different approach: backtracking search
            # For now, just break if we can't find more
            break

    # Convert to proper Python ints and lists
    result = []
    for sq in squares:
        new_sq = []
        for row in sq:
            new_row = [int(x) for x in row]
            new_sq.append(new_row)
        result.append(new_sq)

    return result
