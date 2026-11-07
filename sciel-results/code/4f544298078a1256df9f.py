import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 55
    N = 40
    MAX_K = 18
    candidates = list(range(2, N + 1))

    # Precompute vectors (prime exponents) for each number 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    p_index = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    vectors = {}
    for x in range(2, 41):
        v = [0] * n_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[x] = tuple(v)

    # Fast validity check: vectors are linearly independent over Q
    # Using fraction-free Gaussian elimination
    def is_independent(vecs):
        """Check if list of integer vectors are linearly independent over Q."""
        if not vecs:
            return True
        n = len(vecs[0])
        # Work with a copy as list of lists
        rows = [list(v) for v in vecs]
        m = len(rows)
        rank = 0
        for col in range(n):
            # Find pivot
            pivot = -1
            for i in range(rank, m):
                if rows[i][col] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue
            # Swap rows
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            # Eliminate below
            for i in range(rank + 1, m):
                factor = rows[i][col]
                if factor == 0:
                    continue
                # Fraction-free: multiply row i by pivot_val, subtract factor * pivot_row
                pivot_val = rows[rank][col]
                for j in range(col, n):
                    rows[i][j] = rows[i][j] * pivot_val - factor * rows[rank][j]
            rank += 1
            if rank == m:
                break
        return rank == m

    best_set = []
    best_size = 0

    # Backtracking search with fast independence check
    def search_order(order):
        nonlocal best_set, best_size
        best_local = []
        best_size_local = 0
        
        def dfs(idx, current_set, current_vecs):
            nonlocal best_local, best_size_local
            if time.time() > deadline - 2:
                return
            
            if len(current_set) > best_size_local:
                best_size_local = len(current_set)
                best_local = current_set[:]
                if best_size_local >= 18:
                    return

            if idx >= len(order):
                return

            x = order[idx]
            v = vectors[x]
            
            # Check if adding v keeps the set linearly independent
            if is_independent(current_vecs + [v]):
                if time.time() > deadline - 2:
                    return
                dfs(idx + 1, current_set + [x], current_vecs + [v])
            
            dfs(idx + 1, current_set, current_vecs)

        dfs(0, [], [])
        return best_local, best_size_local

    start_time = time.time()
    
    # Try multiple orderings
    orders = []
    # Ascending
    orders.append(sorted(candidates))
    # Primes first
    primes_in_range = [x for x in candidates if x in p_index]
    composites_in_range = [x for x in candidates if x not in p_index]
    orders.append(sorted(primes_in_range) + sorted(composites_in_range))
    # Descending
    orders.append(sorted(candidates, reverse=True))
    # Composites first (may allow more total elements)
    orders.append(sorted(composites_in_range) + sorted(primes_in_range))
    
    # Add random orders
    for _ in range(10):
        r = candidates[:]
        random.shuffle(r)
        orders.append(r)

    for order in orders:
        if time.time() > deadline - 5:
            break
        s, sz = search_order(order)
        if sz > best_size:
            best_size = sz
            best_set = s[:]
        if best_size >= 18:
            break

    # More random restarts if needed
    if best_size < 16:
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            r = candidates[:]
            random.shuffle(r)
            s, sz = search_order(r)
            if sz > best_size:
                best_size = sz
                best_set = s[:]
            if best_size >= 18:
                break

    # Fallback
    if not best_set:
        best_set = [2, 3]
        best_size = 2

    return [int(x) for x in best_set]
