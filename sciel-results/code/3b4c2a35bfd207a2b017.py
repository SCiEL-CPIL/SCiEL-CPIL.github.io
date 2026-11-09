"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True

def _is_orth_to_all(sq, existing):
    for ex in existing:
        if not _are_orthogonal(sq, ex):
            return False
    return True

def construct():
    n = 10
    import time
    start_time = time.time()
    deadline = start_time + 85

    # Square 1: standard cyclic
    s1 = [[(i + j) % 10 for j in range(n)] for i in range(n)]
    squares = [s1]

    # Generate candidate Latin squares and greedily add orthogonal ones
    candidates = []
    
    # Type 1: (a*i + b*j) % 10
    for a in range(1, 10):
        for b in range(1, 10):
            sq = [[(a*i + b*j) % 10 for j in range(n)] for i in range(n)]
            if _is_latin(sq):
                candidates.append(sq)

    # Type 2: (a*i + b*j + c*i*j) % 10
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                sq = [[(a*i + b*j + c*i*j) % 10 for j in range(n)] for i in range(n)]
                if _is_latin(sq):
                    candidates.append(sq)

    # Type 3: (a*i + b*j + c) % 10 (constant shift, same as type 1 essentially)
    # Skip - redundant with type 1

    # Greedily add orthogonal squares
    for sq in candidates:
        if time.time() > deadline:
            break
        if len(squares) >= 9:  # Max possible is n-1=9, but known max is 3
            break
        if not _is_orth_to_all(sq, squares):
            continue
        dup = False
        for ex in squares:
            if ex == sq:
                dup = True
                break
        if not dup:
            squares.append(sq)

    # Ensure at least 1
    if not squares:
        squares = [s1]

    # Convert to pure Python int lists
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result
# EVOLVE-BLOCK-END
