import time
import numpy as np

def construct():
    """
    Finds a small tournament where every 4 vertices have a common dominator.
    Uses local search to minimize the number of 'bad' 4-sets.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # We aim for n around 60-70. 
    # The known smallest is 67. Let's try to find one <= 67.
    # If we can't find a valid one, we return the best we have (lowest bad count).
    # But the task requires a VALID tournament. If we can't find one, we return the seed (83) 
    # or the best attempt if it became valid.
    # Actually, the prompt says "Minimize n". A valid tournament on n vertices certifies f(4) <= n.
    # The evaluator measures the raw value (n). Lower is better.
    # If we return an invalid tournament, it scores 0.
    # So we MUST return a valid tournament.
    # The seed program returns a valid Paley tournament on 83 vertices.
    # Strategy: Try to find a valid tournament with n < 83.
    # If we succeed, return it. If not, return the Paley 83 tournament.
    
    # Helper to generate Paley tournament
    def _paley_orientation(p):
        squares = np.array([(x * x) % p for x in range(1, p)], dtype=np.int8)
        sq_set = set(squares)
        # Create adjacency matrix
        # i -> j if (j - i) % p is in sq_set
        # Note: In Paley tournament on p (prime 3 mod 4), i->j if j-i is quadratic residue.
        # Wait, standard definition: i -> j iff j - i is a quadratic residue mod p.
        # Let's verify direction for the bit array.
        # Bit (i, j) with i < j: 0 means i -> j, 1 means j -> i.
        # If i -> j, bit is 0. This happens if (j - i) % p is a square.
        # If j -> i, bit is 1. This happens if (i - j) % p is a square (which means (j-i)%p is non-square).
        
        # We need to return a list of bits for pairs (i, j) with i < j in lexicographic order.
        # Total pairs: n*(n-1)/2.
        
        bits = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in sq_set:
                    bits.append(0) # i -> j
                else:
                    bits.append(1) # j -> i
        return bits

    # Helper to check validity and count bad 4-sets
    def _count_bad_4sets(adj_matrix, n):
        """
        adj_matrix: n x n boolean matrix where adj[i][j] is True if i -> j.
        A 4-set {a,b,c,d} is bad if no vertex dominates the other 3.
        Vertex v dominates {a,b,c,d} if v -> a, v -> b, v -> c, v -> d.
        So a 4-set is good if there exists v in {a,b,c,d} such that v -> u for all u in {a,b,c,d} \ {v}.
        """
        bad_count = 0
        # Vectorized approach for counting bad 4-sets is complex.
        # For n <= 70, n^4 is ~ 24 million. In Python, a loop is too slow.
        # We need a faster check or a smaller n.
        
        # Let's use a different perspective:
        # A 4-set is bad if for every vertex v in the set, there is at least one vertex u in the set such that v does NOT point to u (i.e., u -> v).
        # Let A be the adjacency matrix.
        # For a specific 4-set S, let M_S be the 4x4 submatrix of A.
        # S is good if there is a row in M_S that is all 1s (except diagonal).
        
        # We can iterate through all 4-sets using numpy combinations? No, itertools is better for iteration, but we need speed.
        # Let's try to optimize the check.
        # If n is small (e.g., 60), C(60, 4) = 487,635. This is small enough to check in a tight loop if optimized, or vectorized.
        
        # Let's use numpy to vectorize the check for all 4-sets?
        # Generating all 4-sets indices: shape (K, 4).
        # This might use too much memory if K is large.
        # For n=70, K ~ 916,895. 916k * 4 * 8 bytes ~ 30 MB. Acceptable.
        
        import itertools
        
        # Pre-compute out-neighborhoods as bitmasks?
        # For each vertex i, mask[i] is a bitmask of vertices i points to.
        # Then for a set {a,b,c,d}, a dominates if mask[a] & (1<<b) & (1<<c) & (1<<d) is non-zero?
        # No, mask[a] needs to contain b, c, AND d.
        # So (mask[a] & (1<<b)) and (mask[a] & (1<<c)) and (mask[a] & (1<<d)).
        
        masks = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j]:
                    masks[i] |= (1 << j)
        
        # Iterate 4-sets
        # This is still a Python loop over ~1M items. ~1-2 seconds in PyPy, maybe 5-10s in CPython.
        # We have 140s. We can do many iterations of local search.
        
        count = 0
        for a, b, c, d in itertools.combinations(range(n), 4):
            # Check if a dominates b,c,d
            if (masks[a] & (1 << b)) and (masks[a] & (1 << c)) and (masks[a] & (1 << d)):
                continue
            # Check if b dominates a,c,d
            if (masks[b] & (1 << a)) and (masks[b] & (1 << c)) and (masks[b] & (1 << d)):
                continue
            # Check if c dominates a,b,d
            if (masks[c] & (1 << a)) and (masks[c] & (1 << b)) and (masks[c] & (1 << d)):
                continue
            # Check if d dominates a,b,c
            if (masks[d] & (1 << a)) and (masks[d] & (1 << b)) and (masks[d] & (1 << c)):
                continue
            count += 1
        return count

    # Local Search
    def _local_search(n, max_iter=500):
        # Start with random tournament
        # adj[i][j] = 1 if i->j
        # Random orientation
        adj = np.random.randint(0, 2, (n, n), dtype=np.int8)
        # Ensure anti-symmetry
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0
                    adj[j, i] = 1
        adj[np.diag_indices(n)] = 0

        best_bad_count = _count_bad_4sets(adj, n)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Pick a random edge to flip
            i, j = np.random.choice(n, 2, replace=False)
            
            # Calculate change in bad count?
            # Flipping edge (i, j) affects all 4-sets containing both i and j.
            # Number of such sets is C(n-2, 2) = (n-2)(n-3)/2.
            # For n=60, that's 58*57/2 = 1653 sets.
            # We can recalculate the bad count for these specific sets.
            
            old_adj_ij = adj[i, j]
            old_adj_ji = adj[j, i]
            
            # Flip
            adj[i, j] = 1 - old_adj_ij
            adj[j, i] = 1 - old_adj_ji
            
            # Recalculate bad count for sets containing i and j
            # We need to check only the sets {i, j, k, l}
            # Let's collect indices k, l
            others = [k for k in range(n) if k != i and k != j]
            
            new_local_bad = 0
            # We need masks for current state
            # Updating masks is O(n).
            # Let's just recompute masks for i and j?
            # No, masks for k and l also matter for domination.
            # Actually, the condition "v dominates others" depends on edges from v to others.
            # Flipping (i, j) changes:
            # 1. Whether i dominates j (and vice versa).
            # 2. For any k, l, the domination status of i or j within {i, j, k, l} might change.
            
            # This is getting complicated to do incrementally.
            # Given n is small (<70), maybe full recalculation is fast enough if we optimize the loop?
            # _count_bad_4sets takes ~1s for n=60.
            # If we do 50 iterations, that's 50s.
            # We can afford ~100 full recalculations.
            
            # Let's just do full recalculation for simplicity and robustness.
            current_bad_count = _count_bad_4sets(adj, n)
            
            if current_bad_count < best_bad_count:
                best_bad_count = current_bad_count
                if best_bad_count == 0:
                    return adj, True
            else:
                # Revert (or accept with probability p for simulated annealing)
                # Simple hill climbing: revert
                adj[i, j] = old_adj_ij
                adj[j, i] = old_adj_ji

        return adj, (best_bad_count == 0)

    # Try various n
    # Start from small n, go up.
    # If we find a valid one, return it.
    # If we run out of time, return the best valid one found (or seed if none).
    
    # The seed is 83. We want < 83.
    # Let's try n from 60 to 82.
    # If we find a valid tournament for n, we can try smaller n?
    # Actually, if we find valid for n, we return n.
    # We want the MINIMUM n.
    # So we should try n=31, 32, ... up to 83?
    # But searching for n=31 is very hard (likely impossible).
    # The known lower bound is 31.
    # Random search is unlikely to find a valid one for small n.
    # Paley tournament for p=83 is valid.
    # Are there smaller Paley tournaments?
    # p must be prime 3 mod 4.
    # 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83.
    # Let's check if Paley tournaments for smaller p satisfy the property.
    
    def _check_paley(p):
        if p < 5: return False
        # Generate Paley
        squares = set((x * x) % p for x in range(1, p))
        # Build adj
        adj = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(p):
                if i != j:
                    if (j - i) % p in squares:
                        adj[i, j] = 1
                    else:
                        adj[i, j] = 0
        # Check 4-sets
        # Use the same counter
        # If bad count is 0, it's valid.
        bad = _count_bad_4sets(adj, p)
        return bad == 0

    # Check small primes
    valid_n = None
    for p in [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79]:
        if time.time() > deadline - 5.0:
            break
        if _check_paley(p):
            valid_n = p
            break
            
    if valid_n:
        # Generate the orientation for this valid_n
        p = valid_n
        squares = set((x * x) % p for x in range(1, p))
        bits = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    bits.append(0)
                else:
                    bits.append(1)
        return {'n': p, 'orientation': bits}

    # If no small Paley works, try local search on n=60..70?
    # Or just return the 83 Paley tournament to be safe.
    # Let's try local search on n=67 (the known smallest size, maybe we can find one?).
    # Actually, if a valid tournament of size 67 exists, it's not necessarily Paley.
    # But finding one via random+local search in 100s is a shot in the dark.
    
    # Let's try local search for n=60, 61, ... up to 70.
    # If we find a valid one, return it.
    for n in range(60, 71):
        if time.time() > deadline - 20.0:
            break
        # Run local search
        adj, is_valid = _local_search(n, max_iter=50)
        if is_valid:
            # Convert to bits
            bits = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 1:
                        bits.append(0) # i -> j
                    else:
                        bits.append(1) # j -> i
            return {'n': n, 'orientation': bits}

    # Fallback: Return the 83 Paley tournament
    p = 83
    squares = set((x * x) % p for x in range(1, p))
    bits = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits.append(0)
            else:
                bits.append(1)
    return {'n': p, 'orientation': bits}

# EVOLVE-BLOCK-END
