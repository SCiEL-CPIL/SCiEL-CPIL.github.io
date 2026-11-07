import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0
    
    def generate_random_latin_square():
        # Generate a random Latin square of order n
        L = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        
        # Fill first row with a random permutation
        first_row = list(range(n))
        random.shuffle(first_row)
        for j in range(n):
            L[0][j] = first_row[j]
            row_used[0].add(first_row[j])
            col_used[j].add(first_row[j])
            
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n * n:
                return True
            
            i = pos // n
            j = pos % n
            
            # If cell is already filled (first row), skip
            if L[i][j] != -1:
                return backtrack(pos + 1)
                
            # Get possible values
            possible = [v for v in range(n) if v not in row_used[i] and v not in col_used[j]]
            random.shuffle(possible) # Randomize order
            
            for val in possible:
                L[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                
                if backtrack(pos + 1):
                    return True
                
                L[i][j] = -1
                row_used[i].remove(val)
                col_used[j].remove(val)
        
        # Start backtracking from index n (second row)
        if backtrack(n):
            return L
        return None

    def find_orthogonal_mate(L1):
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        def is_valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            if (L1[i][j], val) in pair_used:
                return False
            return True

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n * n:
                return True
            
            i = pos // n
            j = pos % n
            
            possible = [v for v in range(n) if is_valid(i, j, v)]
            # Heuristic: Sort by number of constraints? Or just random.
            random.shuffle(possible)
            
            for val in possible:
                L2[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                pair_used.add((L1[i][j], val))
                
                if backtrack(pos + 1):
                    return True
                
                L2[i][j] = -1
                row_used[i].remove(val)
                col_used[j].remove(val)
                pair_used.remove((L1[i][j], val))
        
        if backtrack(0):
            return L2
        return None

    # Try to find a pair
    best_result = []
    
    for attempt in range(10): # Try a few random L1s
        if time.time() > deadline:
            break
            
        L1 = generate_random_latin_square()
        if L1 is None:
            continue
            
        if not best_result:
            best_result = [L1]
            
        L2 = find_orthogonal_mate(L1)
        if L2 is not None:
            return [L1, L2]
            
    return best_result
