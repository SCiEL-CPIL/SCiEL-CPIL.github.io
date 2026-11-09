import time

def construct():
    n = 10
    deadline = time.time() + 50  # Leave 10s safety margin
    
    # Validation helpers
    def is_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            col = [sq[k][i] for k in range(n)]
            if set(col) != set(range(n)):
                return False
        return True

    def is_orthogonal(sq_a, sq_b):
        seen_pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq_a[i][j], sq_b[i][j])
                if pair in seen_pairs:
                    return False
                seen_pairs.add(pair)
        return True

    def is_valid_set(sqs):
        for sq in sqs:
            if not is_latin(sq):
                return False
        for i in range(len(sqs)):
            for j in range(i+1, len(sqs)):
                if not is_orthogonal(sqs[i], sqs[j]):
                    return False
        return True

    # Try different constructions and return the best valid set
    
    # Construction 1: Use Z_2 x Z_5 with careful mapping
    # Represent i,j as (i1,i2) where i1 in {0,1}, i2 in {0,1,2,3,4}
    # Map to 5*i1 + i2
    # For square k, use: S_k(i,j) = 5*((i1 + a_k*j1) % 2) + ((i2 + b_k*j2) % 5)
    # where (a_k, b_k) are chosen to ensure orthogonality
    
    # We need for any k != l: the pair (S_k, S_l) is injective
    # S_k = 5*((i1 + a_k*j1)%2) + ((i2 + b_k*j2)%5)
    # S_l = 5*((i1 + a_l*j1)%2) + ((i2 + b_l*j2)%5)
    
    # Try all combinations of a in {0,1}, b in {1,2,3,4}
    candidates = []
    for a in range(2):
        for b in range(1, 5):
            sq = []
            for i in range(n):
                row = []
                for j in range(n):
                    i1 = i // 5
                    i2 = i % 5
                    j1 = j // 5
                    j2 = j % 5
                    val = 5 * ((i1 + a * j1) % 2) + ((i2 + b * j2) % 5)
                    row.append(val)
                sq.append(row)
            candidates.append((a, b, sq))
    
    # Greedily select orthogonal squares
    selected = []
    for a, b, sq in candidates:
        if time.time() > deadline - 5:
            break
        if not is_latin(sq):
            continue
        orthogonal = True
        for sel_sq in selected:
            if not is_orthogonal(sq, sel_sq):
                orthogonal = False
                break
        if orthogonal:
            selected.append(sq)
    
    if len(selected) > 0:
        result = [
            [[int(val) for val in row] for row in sq]
            for sq in selected
        ]
        return result
    
    # Fallback: return a single valid Latin square
    sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [[[int(val) for val in row] for row in sq]]
