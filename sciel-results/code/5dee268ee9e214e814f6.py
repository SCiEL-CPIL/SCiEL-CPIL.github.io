import time

def _is_latin(sq, n=10):
    for r in range(n):
        row = set(sq[r])
        if len(row) != n:
            return False
    for c in range(n):
        col = set(sq[r][c] for r in range(n))
        if len(col) != n:
            return False
    return True

def _orthogonal(a, b, n=10):
    seen = set()
    for r in range(n):
        for c in range(n):
            p = (a[r][c], b[r][c])
            if p in seen:
                return False
            seen.add(p)
    return True

def _try_find_second(deadline):
    # Search for a Latin square orthogonal to the cyclic square
    # Using randomized local search / simulated annealing style
    import random
    import math
    
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Start with a random permutation of columns and rows
    best_score = -1
    best_sq = None
    
    # Use a simpler approach: try all permutations of a base structure
    # Since 10! is too large, use random restarts with local search
    
    def score(sq):
        # Count distinct pairs with base
        pairs = set()
        conflicts = 0
        for r in range(n):
            for c in range(n):
                p = (base[r][c], sq[r][c])
                if p in pairs:
                    conflicts += 1
                pairs.add(p)
        # Also check Latin property
        for r in range(n):
            if len(set(sq[r])) != n:
                conflicts += 100
        for c in range(n):
            if len(set(sq[r][c] for r in range(n))) != n:
                conflicts += 100
        return -conflicts
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random initialization
        sq = [[random.randint(0, n-1) for _ in range(n)] for _ in range(n)]
        
        # Local search: swap elements to reduce conflicts
        cur_score = score(sq)
        for step in range(200):
            if time.time() > deadline:
                break
            if cur_score == 0:
                return sq
            
            # Pick two random positions to swap
            r1, c1 = random.randint(0, n-1), random.randint(0, n-1)
            r2, c2 = random.randint(0, n-1), random.randint(0, n-1)
            if r1 == r2 and c1 == c2:
                continue
            
            sq[r1][c1], sq[r2][c2] = sq[r2][c2], sq[r1][c1]
            new_score = score(sq)
            
            if new_score >= cur_score:
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_sq = [row[:] for row in sq]
            else:
                sq[r1][c1], sq[r2][c2] = sq[r2][c2], sq[r1][c1]
    
    if best_sq is not None and best_score == 0:
        return best_sq
    return None

def construct():
    n = 10
    deadline = time.time() + 80  # Safety margin
    
    # Construct the cyclic Latin square
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second orthogonal Latin square
    second = _try_find_second(deadline)
    
    if second is not None:
        # Verify orthogonality
        if _is_latin(second, n) and _orthogonal(square, second, n):
            return [square, second]
    
    return [square]
