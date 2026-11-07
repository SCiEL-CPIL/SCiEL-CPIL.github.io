# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers in 2..40 maximizing count such that
    all subset products are distinct.
    
    Strategy: 
    The condition is equivalent to the numbers being multiplicatively independent,
    which means their prime exponent vectors are linearly independent over Q.
    We use a greedy algorithm to select the largest linearly independent subset
    from the candidates 2..40.
    """
    candidates = list(range(2, 41))
    
    # 1. Sieve primes up to 40
    is_prime = [True] * 41
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(40**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 41, i):
                is_prime[j] = False
    primes = [p for p in range(2, 41) if is_prime[p]]
    
    # 2. Helper to get exponent vector
    def get_vector(n):
        vec = []
        for p in primes:
            cnt = 0
            while n % p == 0:
                cnt += 1
                n //= p
            vec.append(cnt)
        return tuple(vec)
    
    # 3. Helper to check linear independence
    # We maintain a list of selected vectors.
    # To check if a new vector is independent, we can add it to the matrix
    # and check if the rank increases.
    # Since dimensions are small (12), we can use a simple rank check.
    
    def rank(matrix):
        """Calculate rank of a list of tuples (vectors) using Gaussian elimination with Fractions."""
        if not matrix:
            return 0
        from fractions import Fraction
        # Convert to list of lists of Fractions
        m = [[Fraction(x) for x in vec] for vec in matrix]
        rows = len(m)
        cols = len(m[0])
        rank_val = 0
        for col in range(cols):
            # Find pivot
            pivot_row = None
            for r in range(rank_val, rows):
                if m[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            # Swap
            m[rank_val], m[pivot_row] = m[pivot_row], m[rank_val]
            # Eliminate
            for r in range(rows):
                if r != rank_val and m[r][col] != 0:
                    factor = m[r][col] / m[rank_val][col]
                    for c in range(col, cols):
                        m[r][c] -= factor * m[rank_val][c]
            rank_val += 1
            if rank_val == rows:
                break
        return rank_val

    selected_numbers = []
    selected_vectors = []
    
    # Greedy selection
    # Iterating through candidates in increasing order.
    # Primes will be selected, composites will be skipped if dependent.
    for n in candidates:
        v = get_vector(n)
        # Check independence
        current_rank = rank(selected_vectors)
        new_rank = rank(selected_vectors + [v])
        if new_rank > current_rank:
            selected_numbers.append(n)
            selected_vectors.append(v)
            
    return selected_numbers

# EVOLVE-BLOCK-END
