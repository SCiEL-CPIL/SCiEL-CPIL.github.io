import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # L1: standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_latin(square):
        for i in range(n):
            if sorted(square[i]) != list(range(n)):
                return False
            col = [square[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
    
    def are_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((s1[i][j], s2[i][j]))
        return len(pairs) == n * n
    
    # Try known pairs first - hardcoded valid MOLS(10) if available
    # The cyclic square L1 has no orthogonal mate (it's isotopic to Z_10),
    # so we need a non-cyclic L1.
    
    # Generate a random Latin square that is NOT isotopic to cyclic
    def generate_random_latin_square(n, seed=None):
        """Generate a random Latin square using backtracking."""
        if seed is not None:
            random.seed(seed)
        
        square = [[-1] * n for _ in range(n)]
        
        def backtrack(r, c):
            if c == n:
                c = 0
                r += 1
            if r == n:
                return True
            
            symbols = list(range(n))
            random.shuffle(symbols)
            
            for sym in symbols:
                # Check row constraint
                if sym in square[r]:
                    continue
                # Check column constraint
                if any(square[i][c] == sym for i in range(r)):
                    continue
                # Check pair constraint with L1 (will be checked separately for orthogonality)
                
                square[r][c] = sym
                if backtrack(r, c + 1):
                    return True
                square[r][c] = -1
            
            return False
        
        if backtrack(0, 0):
            return square
        return None
    
    # Better approach: use a known construction or try multiple random seeds
    # For n=10, we know MOLS exist (max is 9), but finding them is non-trivial.
    
    # Strategy: Try to find a valid L2 orthogonal to a given L1 using backtracking
    def find_orthogonal_square(L1, n, time_limit=30.0):
        """Find a Latin square L2 orthogonal to L1 using backtracking."""
        L2 = [[-1] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        def backtrack(r, c):
            if c == n:
                c = 0
                r += 1
            if r == n:
                return True
            
            if time.time() > time_limit:
                return False
            
            # Get candidate symbols
            candidates = [s for s in range(n) 
                         if s not in row_used[r] and s not in col_used[c]]
            
            # Sort by most constrained first (fewest available pairs)
            def constraint_count(sym):
                return sum(1 for s2 in range(n) if (L1[r][c], s2) not in pair_used)
            
            candidates.sort(key=lambda s: constraint_count(s))
            
            random.shuffle(candidates)  # Add randomness to break ties
            
            for sym in candidates:
                if (L1[r][c], sym) in pair_used:
                    continue
                
                L2[r][c] = sym
                row_used[r].add(sym)
                col_used[c].add(sym)
                pair_used.add((L1[r][c], sym))
                
                if backtrack(r, c + 1):
                    return True
                
                L2[r][c] = -1
                row_used[r].remove(sym)
                col_used[c].remove(sym)
                pair_used.remove((L1[r][c], sym))
            
            return False
        
        # Try multiple random restarts
        for attempt in range(20):
            random.seed(attempt * 7919 + 42)
            L2 = [[-1] * n for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            pair_used = set()
            
            t0 = time.time()
            if backtrack(0, 0):
                return L2
            if time.time() > start_time + deadline - 5:
                break
        
        return None
    
    # Since L1 is cyclic (isotopic to Z_10), it has NO orthogonal mate.
    # We MUST use a non-cyclic L1.
    
    # Generate a non-cyclic Latin square
    def generate_non_cyclic_latin(n):
        """Generate a Latin square that is not isotopic to the cyclic group."""
        # Start with a random permutation of symbols in each row
        # and random row/column permutations
        base = [[(i + j) % n for j in range(n)] for i in range(n)]
        
        # Apply random row permutation
        row_perm = list(range(n))
        random.shuffle(row_perm)
        
        # Apply random column permutation  
        col_perm = list(range(n))
        random.shuffle(col_perm)
        
        # Apply random symbol permutation
        sym_perm = list(range(n))
        random.shuffle(sym_perm)
        
        result = [[base[row_perm[i]][col_perm[j]] for j in range(n)] for i in range(n)]
        result = [[sym_perm[s] for s in row] for row in result]
        
        # Check if it's actually different from cyclic (should be with high prob)
        # Verify it's a valid Latin square
        for i in range(n):
            if sorted(result[i]) != list(range(n)):
                return None
            col = [result[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return None
        
        return result
    
    # Try to find a pair of MOLS
    best_result = [L1]  # Fallback: just L1
    
    # Try several random non-cyclic Latin squares and find orthogonal mates
    for seed in range(50):
        if time.time() > start_time + deadline - 10:
            break
        
        random.seed(seed)
        L1_candidate = generate_non_cyclic_latin(n)
        
        if L1_candidate is None:
            continue
        
        # Verify L1_candidate is valid
        if not is_latin(L1_candidate):
            continue
        
        # Try to find orthogonal mate
        L2_candidate = find_orthogonal_square(L1_candidate, n, 
                                              time_limit=start_time + deadline - 5 - time.time())
        
        if L2_candidate is not None:
            # Verify both are valid and orthogonal
            if is_latin(L2_candidate) and are_orthogonal(L1_candidate, L2_candidate):
                best_result = [L1_candidate, L2_candidate]
                # Try to find a third square
                if time.time() < start_time + deadline - 15:
                    L3_candidate = find_orthogonal_to_both(L1_candidate, L2_candidate, n,
                                                           time_limit=start_time + deadline - 5 - time.time())
                    if L3_candidate is not None:
                        if is_latin(L3_candidate) and are_orthogonal(L1_candidate, L3_candidate) and \
                           are_orthogonal(L2_candidate, L3_candidate):
                            best_result = [L1_candidate, L2_candidate, L3_candidate]
                break
    
    # Final validation
    result = []
    for sq in best_result:
        # Ensure all values are Python ints
        validated = [[int(sq[i][j]) for j in range(n)] for i in range(n)]
        result.append(validated)
    
    if not result:
        result = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    return result

def find_orthogonal_to_both(L1, L2, n, time_limit=30.0):
    """Find a Latin square L3 orthogonal to both L1 and L2."""
    L3 = [[-1] * n for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair1_used = set()
    pair2_used = set()
    
    def backtrack(r, c):
        if c == n:
            c = 0
            r += 1
        if r == n:
            return True
        
        if time.time() > time_limit:
            return False
        
        candidates = [s for s in range(n) 
                     if s not in row_used[r] and s not in col_used[c]]
        
        random.shuffle(candidates)
        
        for sym in candidates:
            if (L1[r][c], sym) in pair1_used:
                continue
            if (L2[r][c], sym) in pair2_used:
                continue
            
            L3[r][c] = sym
            row_used[r].add(sym)
            col_used[c].add(sym)
            pair1_used.add((L1[r][c], sym))
            pair2_used.add((L2[r][c], sym))
            
            if backtrack(r, c + 1):
                return True
            
            L3[r][c] = -1
            row_used[r].remove(sym)
            col_used[c].remove(sym)
            pair1_used.remove((L1[r][c], sym))
            pair2_used.remove((L2[r][c], sym))
        
        return False
    
    for attempt in range(10):
        random.seed(attempt * 104729 + 7)
        L3 = [[-1] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair1_used = set()
        pair2_used = set()
        
        if backtrack(0, 0):
            return L3
    
    return None
