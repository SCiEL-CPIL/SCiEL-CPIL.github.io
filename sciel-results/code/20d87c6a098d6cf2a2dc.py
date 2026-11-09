import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Square 1: Standard cyclic Latin square
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [S1]
    
    # Helper to check orthogonality
    def is_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (s1[i][j], s2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
    
    # Helper to check Latin property
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            if sorted([sq[i][j] for i in range(n)]) != list(range(n)):
                return False
        return True
    
    # Function to find a new square orthogonal to all in `squares`
    def find_square(existing_squares, time_limit):
        # We will fill a new square S
        S = [[-1] * n for _ in range(n)]
        
        # State for backtracking
        row_masks = [0] * n
        col_masks = [0] * n
        # pair_masks[k][a] tracks which symbols b are forbidden for value a in existing_squares[k]
        # i.e., if existing_squares[k][i][j] == a, then S[i][j] cannot be b if pair_masks[k][a] has bit b set
        pair_masks = [[0] * n for _ in range(len(existing_squares))]
        
        # We fill cell by cell using MRV heuristic
        def get_available(r, c):
            """Get available symbols for cell (r,c)"""
            avail = []
            for b in range(n):
                if row_masks[r] & (1 << b):
                    continue
                if col_masks[c] & (1 << b):
                    continue
                valid = True
                for k, sq in enumerate(existing_squares):
                    a = sq[r][c]
                    if pair_masks[k][a] & (1 << b):
                        valid = False
                        break
                if valid:
                    avail.append(b)
            return avail
        
        def backtrack():
            if time.time() > time_limit:
                return False
            
            # Find unassigned cell with minimum available values (MRV)
            min_count = n + 1
            best_r, best_c = -1, -1
            best_avail = None
            
            for r in range(n):
                for c in range(n):
                    if S[r][c] == -1:
                        avail = get_available(r, c)
                        if len(avail) == 0:
                            return False  # Dead end
                        if len(avail) < min_count:
                            min_count = len(avail)
                            best_r, best_c = r, c
                            best_avail = avail
                            if min_count == 1:
                                break
                if min_count == 1:
                    break
            
            if best_r == -1:
                return True  # All cells assigned
            
            # Try values for the best cell
            random.shuffle(best_avail)
            for b in best_avail:
                # Place
                S[best_r][best_c] = b
                row_masks[best_r] |= (1 << b)
                col_masks[best_c] |= (1 << b)
                for k, sq in enumerate(existing_squares):
                    a = sq[best_r][best_c]
                    pair_masks[k][a] |= (1 << b)
                
                if backtrack():
                    return True
                
                # Undo
                S[best_r][best_c] = -1
                row_masks[best_r] ^= (1 << b)
                col_masks[best_c] ^= (1 << b)
                for k, sq in enumerate(existing_squares):
                    a = sq[best_r][best_c]
                    pair_masks[k][a] ^= (1 << b)
            
            return False
        
        # Try to find a solution with multiple restarts
        for attempt in range(20):
            if time.time() > time_limit:
                break
            # Reset state
            S = [[-1] * n for _ in range(n)]
            row_masks = [0] * n
            col_masks = [0] * n
            pair_masks = [[0] * n for _ in range(len(existing_squares))]
            
            if backtrack():
                # Verify
                if is_latin(S) and all(is_orthogonal(sq, S) for sq in existing_squares):
                    return S
        return None
    
    # Try to find 2nd square
    S2 = find_square(squares, deadline)
    if S2:
        squares.append(S2)
    
    # Ensure types are correct
    # Convert to lists of lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
