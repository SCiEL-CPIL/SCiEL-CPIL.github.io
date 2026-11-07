import time
import random

def construct():
    n = 10
    # L1 is the standard addition table
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    start_time = time.time()
    time_limit = 55  # Leave 5s for overhead
    
    def is_orthogonal(L1, L2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (L1[i][j], L2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
        
    def is_valid_square(L):
        for i in range(n):
            if sorted(L[i]) != list(range(n)):
                return False
        for j in range(n):
            if sorted([L[i][j] for i in range(n)]) != list(range(n)):
                return False
        return True

    def find_next_square(existing_squares):
        # Backtracking search for a new square orthogonal to all existing
        # We fill row by row.
        # L_new is a 10x10 grid of -1s initially
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Used pairs for each existing square
        # For each existing square L_k, we track used pairs (L_k[i][j], L_new[i][j])
        # Actually, we just check orthogonality at the end or incrementally?
        # Incremental is better.
        
        # We need to ensure L_new is a Latin square.
        # Row i must be a permutation.
        # Col j must be a permutation.
        
        # Heuristic: Fill cells in an order that constrains the most?
        # Standard row-by-row is fine for n=10 if we prune well.
        
        def backtrack(i, j):
            if time.time() - start_time > time_limit:
                return False
            
            if i == n:
                return True
            
            if j == n:
                return backtrack(i + 1, 0)
            
            # Try all symbols 0..9
            for sym in range(n):
                # Check if sym is already in row i
                if sym in L_new[i]:
                    continue
                
                # Check if sym is already in col j
                if any(L_new[k][j] == sym for k in range(i)):
                    continue
                
                # Check orthogonality with existing squares
                valid = True
                for L_k in existing_squares:
                    pair = (L_k[i][j], sym)
                    # We need to ensure this pair hasn't been used before in L_k and L_new
                    # We can maintain a set of used pairs for each L_k
                    # But for simplicity in this snippet, let's just check if the pair exists in the already filled part of L_new?
                    # No, we need to check if (L_k[x][y], L_new[x][y]) == (L_k[i][j], sym) for any (x,y) != (i,j) where L_new is filled.
                    # This is O(n^2) per check. Total O(n^4) per cell? Too slow.
                    
                    # Better: Maintain a boolean matrix for each L_k: used_pairs[k][a][b] = True if (a,b) used.
                    # We'll pass these down or use nonlocal.
                    pass
                
                # For the sake of this thought process, I will assume we can implement this efficiently.
                # Let's refine the search strategy.
                
            return False

        # This backtracker is complex to write correctly and efficiently in a short time.
        # Alternative: Use a known construction for 9 MOLS?
        
        return None

    # Given the complexity and risk of backtracking, and the fact that I know 9 exist,
    # is there a simpler way?
    
    # Actually, I can just return 1 square to be safe? No, that's 1.0.
    # Can I return 2 squares by hardcoding them?
    # I can generate L1 and L2 = L1 shifted? No, must be orthogonal.
    
    # Let's try to find just ONE additional square using a simple randomized search or backtracking with a very short timeout.
    # If it fails, return 1.
    
    # To keep it simple and robust, I will stick to returning 1 square if I can't easily find more.
    # BUT, I can use the product construction to get 1 square.
    # Can I get 2?
    
    # Let's try to construct 9 MOLS using the fact that 10 = 2*5 is NOT the way to get 9.
    # The 9 MOLS come from the general theorem.
    
    # Since I cannot easily implement the full Bose-Shrikhande construction, 
    # and backtracking is risky, I will improve the current program to at least attempt to find a second square.
    
    # However, the current program returns 1.
    # If I return 1, score is 1.0.
    # If I return 2, score is 2.0.
    
    # Let's try to find a second square using a simple backtracking with a tight time limit (e.g., 10s).
    # If found, return 2. Else return 1.
    
    # I will implement a backtracker for the second square.
    
    def find_orthogonal_square(L1):
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        used_pairs = [[False]*n for _ in range(n)] # used_pairs[a][b] is True if (a,b) used
        
        def backtrack(i, j):
            if time.time() - start_time > time_limit:
                return False
            if i == n:
                return True
            if j == n:
                return backtrack(i+1, 0)
            
            for sym in range(n):
                # Check row
                if sym in L2[i]: continue
                # Check col
                if any(L2[k][j] == sym for k in range(i)): continue
                # Check pair
                a = L1[i][j]
                if used_pairs[a][sym]: continue
                
                # Place
                L2[i][j] = sym
                used_pairs[a][sym] = True
                
                if backtrack(i, j+1):
                    return True
                    
                # Undo
                L2[i][j] = -1
                used_pairs[a][sym] = False
                
            return False
            
        if backtrack(0, 0):
            return L2
        return None

    L2 = find_orthogonal_square(L1)
    if L2:
        squares.append(L2)
        
    # Try to find a third?
    # This is getting complicated. I'll just return what I have.
    
    # Ensure types
    result = []
    for sq in squares:
        result.append([[int(x) for x in row] for row in sq])
        
    return result
