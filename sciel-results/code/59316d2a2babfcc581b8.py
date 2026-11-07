import time

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def solve_square(existing_squares):
        # existing_squares is a list of squares to be orthogonal to
        # Returns a new square or None
        sq = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fix first row to 0..9 to break symmetry
        for j in range(n):
            sq[0][j] = j
            
        row_sets = [set(range(n)) for _ in range(n)]
        col_sets = [set(range(n)) for _ in range(n)]
        used_pairs = [set() for _ in range(len(existing_squares))]
        
        for j in range(n):
            val = j
            row_sets[0].remove(val)
            col_sets[j].remove(val)
            for k, esq in enumerate(existing_squares):
                used_pairs[k].add((esq[0][j], val))
                
        def bt(idx):
            if time.time() > deadline:
                return False
            if idx == n * n:
                return True
            i = idx // n
            j = idx % n
            
            # If i == 0, we already filled it. 
            # Since we start idx at n, i is always >= 1.
            
            for val in row_sets[i]:
                if val not in col_sets[j]:
                    continue
                valid = True
                for k, esq in enumerate(existing_squares):
                    if (esq[i][j], val) in used_pairs[k]:
                        valid = False
                        break
                if not valid:
                    continue
                
                sq[i][j] = val
                row_sets[i].remove(val)
                col_sets[j].remove(val)
                for k, esq in enumerate(existing_squares):
                    used_pairs[k].add((esq[i][j], val))
                
                if bt(idx + 1):
                    return True
                
                row_sets[i].add(val)
                col_sets[j].add(val)
                for k, esq in enumerate(existing_squares):
                    used_pairs[k].remove((esq[i][j], val))
                    
            return False
            
        if bt(n):
            return sq
        return None

    L2 = solve_square([L1])
    if L2 is None:
        return [L1]
        
    L3 = solve_square([L1, L2])
    if L3 is None:
        return [L1, L2]
        
    return [L1, L2, L3]
