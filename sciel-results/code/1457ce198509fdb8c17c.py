"""Construct mutually orthogonal Latin squares of order 10."""
# EVOLVE-BLOCK-START
import time

N = 10

def generate_square(k):
    """Generate a Latin square of order 10 using Z5 x Z2 structure.
    
    Rows: i = 2*a + b where a in Z5, b in Z2
    Cols: j = 2*c + d where c in Z5, d in Z2
    Symbol: (e, f) where e = (a + k*c) % 5, f = (b + d + k*(a+c) % 2) % 2
    Encoded as: e*2 + f
    """
    S = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            a = i // 2
            b = i % 2
            c = j // 2
            d = j % 2
            sym_a = (a + k * c) % 5
            sym_b = (b + d + (k * (a + c)) % 2) % 2
            S[i][j] = sym_a * 2 + sym_b
    return S

def check_latin(S):
    for i in range(N):
        if sorted(S[i]) != list(range(N)):
            return False
    for j in range(N):
        col = [S[i][j] for i in range(N)]
        if sorted(col) != list(range(N)):
            return False
    return True

def check_orthogonal(S1, S2):
    pairs = set()
    for i in range(N):
        for j in range(N):
            p = (S1[i][j], S2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    start = time.time()
    deadline = start + 50  # Safety margin
    
    # Try multiple construction families and pick the best set
    best_squares = []
    
    # Family 1: sym_b = (b + d + (k*(a+c)) % 2) % 2
    for k in [1, 2, 3, 4]:
        if time.time() > deadline:
            break
        S = generate_square(k)
        if not check_latin(S):
            continue
        orthogonal = True
        for S2 in best_squares:
            if not check_orthogonal(S, S2):
                orthogonal = False
                break
        if orthogonal:
            best_squares.append(S)
    
    # If we have at least 1 square, return it
    if best_squares:
        result = []
        for S in best_squares:
            result.append([list(row) for row in S])
        return result
    
    # Fallback: trivial Latin square
    return [[[(i+j)%N for j in range(N)] for i in range(N)]]
# EVOLVE-BLOCK-END
