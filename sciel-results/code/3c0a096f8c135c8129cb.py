"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    squares = []
    
    # Generate linear squares L_k[i][j] = (i + k*j) % n
    # Check for orthogonality
    candidates = []
    for k in range(1, 10):
        sq = []
        for i in range(n):
            row = [(i + k * j) % n for j in range(n)]
            sq.append(row)
        candidates.append(sq)
        
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[r][i] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
        
    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n*n
        
    valid = []
    for sq in candidates:
        if not is_latin(sq):
            continue
        ok = True
        for vsq in valid:
            if not is_orthogonal(sq, vsq):
                ok = False
                break
        if ok:
            valid.append(sq)
            
    if not valid:
        # Fallback to 1 square
        sq = []
        for i in range(n):
            row = [(i + j) % n for j in range(n)]
            sq.append(row)
        return [sq]
        
    return valid

# EVOLVE-BLOCK-END
