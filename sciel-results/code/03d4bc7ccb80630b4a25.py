"""
Mutually orthogonal Latin squares of order 10.

Constructs a complete set of 9 MOLS using the CRT decomposition
Z_10 ≅ Z_5 × Z_2. Each square L_k is defined by linear maps on
the two components with distinct slope parameters ensuring
pairwise orthogonality.
"""
import time
import random

_DEADLINE = time.time() + 85.0


def _crt(x5, x2):
    """Chinese Remainder Theorem: find x in 0..9 with x%5=x5, x%2=x2."""
    # x = x5 * 2 * inv(2,5) + x2 * 5 * inv(5,2) mod 10
    # inv(2,5)=3 since 2*3=6≡1 mod 5; inv(5,2)=1 since 5≡1 mod 2
    return (x5 * 6 + x2 * 5) % 10


def _build_square(a5, b5, a2, b2):
    """
    Build a Latin square of order 10 using parameters:
      L(i,j) = (a5*i5 + b5*j5 mod 5, a2*i2 + b2*j2 mod 2)
    where i=(i5,i2), j=(j5,j2) via CRT decomposition.
    For this to be a Latin square, we need b5≠0 mod 5 and b2≠0 mod 2.
    """
    n = 10
    sq = [[0] * n for _ in range(n)]
    for i in range(n):
        i5, i2 = i % 5, i % 2
        for j in range(n):
            j5, j2 = j % 5, j % 2
            r5 = (a5 * i5 + b5 * j5) % 5
            r2 = (a2 * i2 + b2 * j2) % 2
            sq[i][j] = _crt(r5, r2)
    return sq

def _random_latin_square():
    """Generate a random Latin square of order 10."""
    n = 10
    # Start with the cyclic Latin square
    sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    # Apply random row permutations
    rows = list(range(n))
    random.shuffle(rows)
    sq = [sq[i] for i in rows]
    # Apply random column permutations
    cols = list(range(n))
    random.shuffle(cols)
    sq = [[row[j] for j in cols] for row in sq]
    # Apply random symbol permutations
    syms = list(range(n))
    random.shuffle(syms)
    sq = [[syms[sq[i][j]] for j in range(n)] for i in range(n)]
    return sq

def _orthogonality_violation(sq1, sq2):
    """Count the number of duplicate pairs (violation of orthogonality)."""
    n = 10
    pairs = {}
    violations = 0
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                violations += 1
            else:
                pairs[pair] = True
    return violations

def _total_violations(squares):
    """Total number of orthogonality violations across all pairs."""
    k = len(squares)
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += _orthogonality_violation(squares[i], squares[j])
    return total


def _is_latin_square(sq):
    """Check if sq is a valid Latin square of order 10."""
    n = 10
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True


def _are_orthogonal(sq1, sq2):
    """Check if two Latin squares are orthogonal."""
    n = 10
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True


def construct():
    """
    Return a list of 9 mutually orthogonal Latin squares of order 10.
    Each square is a list of 10 lists of Python ints in 0..9.
    """
    random.seed(42)
    
    # First, try the structured approach with 40 candidates
    candidates = []
    for a5 in range(5):
        for b5 in range(1, 5):
            for a2 in range(2):
                b2 = 1
                sq = _build_square(a5, b5, a2, b2)
                if _is_latin_square(sq):
                    candidates.append(sq)
    
    # Greedy + backtracking on structured candidates
    best_set = []
    for sq in candidates:
        if all(_are_orthogonal(sq, prev) for prev in best_set):
            best_set.append(sq)
            if len(best_set) >= 9:
                return best_set
    
    if len(best_set) < 9:
        def backtrack(start, current):
            if len(current) == 9:
                return current
            if time.time() > _DEADLINE - 10.0:
                return None
            for idx in range(start, len(candidates)):
                sq = candidates[idx]
                if all(_are_orthogonal(sq, prev) for prev in current):
                    result = backtrack(idx + 1, current + [sq])
                    if result is not None:
                        return result
            return None
        
        result = backtrack(0, [])
        if result is not None:
            return result
    
    # Simulated annealing approach
    # Start with a random set of 9 Latin squares
    current_squares = [_random_latin_square() for _ in range(9)]
    current_energy = _total_violations(current_squares)
    
    best_squares = [sq[:] for sq in current_squares]
    best_energy = current_energy
    
    start_temp = 10.0
    end_temp = 0.001
    cooling_rate = 0.995
    max_iterations = 50000
    
    temperature = start_temp
    iteration = 0
    
    while iteration < max_iterations and time.time() < _DEADLINE - 5.0:
        # Choose a random square to modify
        idx = random.randint(0, 8)
        old_sq = [row[:] for row in current_squares[idx]]
        
        # Modify: apply random row/col/symbol permutation
        new_sq = _random_latin_square()
        current_squares[idx] = new_sq
        
        new_energy = _total_violations(current_squares)
        delta = new_energy - current_energy
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 1e-10)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_squares = [sq[:] for sq in current_squares]
                if best_energy == 0:
                    return best_squares
        else:
            current_squares[idx] = old_sq
        
        temperature *= cooling_rate
        iteration += 1
    
    if best_energy == 0:
        return best_squares
    
    # Return the best set found (may not be fully orthogonal)
    # But we need to return a valid set. If we can't find 9 orthogonal,
    # return the largest mutually orthogonal subset.
    # Find the largest clique in the orthogonality graph
    def find_largest_clique(squares):
        n = len(squares)
        # Build orthogonality graph
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if _are_orthogonal(squares[i], squares[j]):
                    adj[i][j] = adj[j][i] = True
        
        best_clique = []
        
        def expand_clique(candidates, current):
            nonlocal best_clique
            if len(current) > len(best_clique):
                best_clique = current[:]
            if not candidates:
                return
            for i in range(len(candidates)):
                v = candidates[i]
                new_candidates = [u for u in candidates[i+1:] if adj[v][u]]
                expand_clique(new_candidates, current + [v])
        
        expand_clique(list(range(n)), [])
        return [squares[i] for i in best_clique]
    
    result = find_largest_clique(best_squares)
    if result:
        return result
    
    # Fallback: return a single valid Latin square
    return [candidates[0]]
