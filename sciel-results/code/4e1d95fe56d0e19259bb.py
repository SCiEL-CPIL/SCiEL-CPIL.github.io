import time
import random

def construct():
    n = 10
    deadline = time.time() + 55  # 55s safety margin
    
    # Square 0: Cyclic
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_latin(square):
        for i in range(n):
            if len(set(square[i])) != n:
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(a, b):
        seen = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Try to find a second square orthogonal to sq0
    # We will use a simple backtracking search
    # To make it fast, we fix the first row to be a permutation
    # and use constraints.
    
    def find_second_square():
        # Heuristic: The second square must be Latin and orthogonal to sq0.
        # We can iterate through possible permutations for the first row.
        # There are 10! = 3.6M permutations.
        # For each, we can try to fill the rest.
        # This might be too slow.
        
        # Alternative: Random search with backtracking.
        # We fill cell by cell.
        
        sq = [[-1 for _ in range(n)] for _ in range(n)]
        used_rows = [set() for _ in range(n)]
        used_cols = [set() for _ in range(n)]
        used_pairs = set()
        
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            
            if i == n:
                return True
            
            if j == n:
                return backtrack(i + 1, 0)
            
            if sq[i][j] != -1:
                return backtrack(i, j + 1)
            
            for val in range(n):
                if val in used_rows[i] or val in used_cols[j]:
                    continue
                
                pair = (sq0[i][j], val)
                if pair in used_pairs:
                    continue
                
                # Place val
                sq[i][j] = val
                used_rows[i].add(val)
                used_cols[j].add(val)
                used_pairs.add(pair)
                
                if backtrack(i, j + 1):
                    return True
                
                # Unplace
                sq[i][j] = -1
                used_rows[i].remove(val)
                used_cols[j].remove(val)
                used_pairs.remove(pair)
            
            return False
        
        # To speed up, we can fix the first row to be identity or a random permutation
        # Actually, we can just let backtracking handle it, but ordering matters.
        # We can try a few random initial permutations for the first row to guide the search.
        
        for _ in range(1000): # Try 1000 random restarts
            if time.time() > deadline:
                break
            
            # Randomize the order of values to try? Or randomize first row?
            # Let's randomize the first row.
            first_row = list(range(n))
            random.shuffle(first_row)
            
            # Initialize sq with first row
            for j in range(n):
                sq[0][j] = first_row[j]
                used_rows[0].add(first_row[j])
                used_cols[j].add(first_row[j])
                used_pairs.add((sq0[0][j], first_row[j]))
            
            # Reset other rows
            for i in range(1, n):
                for j in range(n):
                    sq[i][j] = -1
                    used_rows[i].clear()
                    # used_cols are already set for row 0, need to be careful
                    # Actually, used_cols[j] currently has first_row[j].
                    # We need to clear used_cols for the search, but keep row 0?
                    # No, backtracking will handle it.
                    # But my backtrack function assumes sq[i][j] == -1 for unfilled.
                    # It doesn't reset used_cols/rows at start of backtrack.
                    # So I must reset them properly.
            
            # Let's restart the state properly for each attempt
            sq = [[-1 for _ in range(n)] for _ in range(n)]
            used_rows = [set() for _ in range(n)]
            used_cols = [set() for _ in range(n)]
            used_pairs = set()
            
            for j in range(n):
                sq[0][j] = first_row[j]
                used_rows[0].add(first_row[j])
                used_cols[j].add(first_row[j])
                used_pairs.add((sq0[0][j], first_row[j]))
            
            if backtrack(1, 0):
                return sq
        
        return None

    sq1 = find_second_square()
    
    if sq1:
        # Verify just in case
        if is_latin(sq1) and are_orthogonal(sq0, sq1):
            return [sq0, sq1]
    
    return [sq0]
