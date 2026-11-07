"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 10}

# We know that a maximum set of mutually orthogonal Latin squares (MOLS) of order 10
# has size 9 (since 10 is not a prime power, but we can construct 9 MOLS of order 10).
# Actually, it is known that the maximum number of MOLS of order 10 is 9.
# Wait, let's verify. The maximum number of MOLS of order n is n-1.
# For n=10, is it possible to have 9 MOLS?
# The existence of 9 MOLS of order 10 is a known result.
# In fact, it was shown that there are 9 MOLS of order 10.
# Let's try to construct them or find a known construction.

# A simple construction for prime powers p^k gives p^k - 1 MOLS.
# 10 is not a prime power.
# However, it is known that there exist 9 MOLS of order 10.
# One way to generate them is via a specific permutation of a known set.

# Let's use a known construction or search.
# Since we have a 60s limit, we can try a heuristic search or use a known explicit construction.

# Explicit construction for 9 MOLS of order 10:
# This is non-trivial to hardcode without a reference.
# Let's try to generate them using a backtracking or greedy algorithm with random restarts.
# Or, we can use the fact that we can combine two orthogonal Latin squares from a prime power construction?
# No, that's for product orders.

# Let's try a simple greedy search.
# Start with a random Latin square.
# Try to add another random Latin square that is orthogonal to all existing ones.
# Repeat until no more can be added or time is up.

def is_latin_square(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
        for j in range(n):
            col = [sq[k][j] for k in range(n)]
            if sorted(col) != list(range(n)):
                return False
    return True

def is_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True

def generate_random_latin_square(n):
    # Start with the cyclic Latin square
    sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    # Shuffle rows
    random.shuffle(sq)
    # Shuffle columns
    cols = list(range(n))
    random.shuffle(cols)
    sq = [[sq[i][cols[j]] for j in range(n)] for i in range(n)]
    # Shuffle symbols
    symbols = list(range(n))
    random.shuffle(symbols)
    sq = [[symbols[sq[i][j]] for j in range(n)] for i in range(n)]
    return sq

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_squares = []
    
    # We know that at least 1 square is easy.
    # Let's try to find a large set.
    # We will use a randomized greedy approach.
    # For each attempt, we start with a random square and try to add more.
    
    max_attempts = 2000
    attempt = 0
    
    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        squares = []
        # Start with a random square
        sq = generate_random_latin_square(n)
        squares.append(sq)
        
        # Try to add more squares
        progress = True
        while progress and time.time() < deadline:
            progress = False
            # Try a few random squares
            for _ in range(50):
                if time.time() > deadline:
                    break
                cand = generate_random_latin_square(n)
                orthogonal_to_all = True
                for s in squares:
                    if not is_orthogonal(s, cand):
                        orthogonal_to_all = False
                        break
                if orthogonal_to_all:
                    squares.append(cand)
                    progress = True
                    break
        
        if len(squares) > len(best_squares):
            best_squares = squares
            if len(best_squares) == n - 1:  # Max possible is 9
                break
    
    # Ensure we return at least 1 square
    if not best_squares:
        best_squares = [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    # Convert to lists of ints to be safe
    result = []
    for sq in best_squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result

# EVOLVE-BLOCK-END
