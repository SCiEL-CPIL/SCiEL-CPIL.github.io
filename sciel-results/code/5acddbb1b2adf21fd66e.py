"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 10}

def is_latin_square(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
        for j in range(n):
            col = [sq[k][j] for k in range(n)]
            if sorted(col) != list(range(n)):
                return False
    return True

def is_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Pre-compute L1 values
    # L1[i][j] = (i + j) % n
    
    def try_find_l2(order_seed):
        """Try to find L2 orthogonal to L1 using backtracking with given random seed."""
        rng = random.Random(order_seed)
        
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        # Create a shuffled order of cells to fill
        cells = [(i, j) for i in range(n) for j in range(n)]
        rng.shuffle(cells)
        
        # Pre-compute L1 values for each cell in the shuffled order
        # L1[i][j] = (i+j) % n
        
        def is_valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            l1_val = (i + j) % n
            if (l1_val, val) in pair_used:
                return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n * n:
                return True
            
            i, j = cells[idx]
            l1_val = (i + j) % n
            
            # Try values in random order for this restart
            vals = list(range(n))
            rng.shuffle(vals)
            
            for val in vals:
                if is_valid(i, j, val):
                    L2[i][j] = val
                    row_used[i].add(val)
                    col_used[j].add(val)
                    pair_used.add((l1_val, val))
                    
                    if backtrack(idx + 1):
                        return True
                    else:
                        # Undo
                        L2[i][j] = -1
                        row_used[i].remove(val)
                        col_used[j].remove(val)
                        pair_used.remove((l1_val, val))
            
            return False
        
        found = backtrack(0)
        if found:
            return L2
        return None
    
    # Try multiple randomized restarts to find L2
    # Each restart uses a different random seed for cell ordering and value ordering
    max_restarts = 200
    L2 = None
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        L2 = try_find_l2(attempt)
        if L2 is not None:
            break
    
    if L2 is not None:
        # Verify L2 is a valid Latin square and orthogonal to L1
        if is_latin_square(L2) and is_orthogonal(L1, L2):
            result = []
            # Convert to lists of Python ints
            result.append([[int(L1[i][j]) for j in range(n)] for i in range(n)])
            result.append([[int(L2[i][j]) for j in range(n)] for i in range(n)])
            return result
    
    # Fallback: Return just L1
    return [[[int(L1[i][j]) for j in range(n)] for i in range(n)]]

# EVOLVE-BLOCK-END
