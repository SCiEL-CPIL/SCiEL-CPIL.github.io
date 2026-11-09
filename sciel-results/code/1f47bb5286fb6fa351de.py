import time
import numpy as np

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
# We only need to check if factors are a subset of target_primes
# So we can precompute a bitmask of small primes? 
# No, simpler: just factorize and check.
# But for speed in graph construction, we can precompute a boolean array for specific prime sets.

def get_prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Define prime sets to try
    # k=2: {2} -> sums are powers of 2
    # k=3: {2, 3, 5} -> sums are 5-smooth
    prime_sets = [
        [2],
        [2, 3, 5],
        [2, 3, 5, 7],
    ]
    
    # Precompute validity of sums for each prime set
    # valid_sums[k][s] = True if s is k-smooth
    valid_sums_cache = {}
    for k, primes in enumerate(prime_sets):
        target = set(primes)
        valid = [False] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            # Check if s is smooth
            n = s
            is_smooth = True
            while n > 1:
                p = spf[n]
                if p not in target:
                    is_smooth = False
                    break
                while n % p == 0:
                    n //= p
            if is_smooth:
                valid[s] = True
        valid_sums_cache[k] = valid

    # Tomita's Maximum Clique Algorithm with Bitmasks
    # Graph nodes: 1 to 10000
    # We will map valid nodes to indices 0..N-1 for efficient bitmasks?
    # Actually, 10000 bits fits in ~156 64-bit words. Python ints are arbitrary precision.
    # Using 0-indexed or 1-indexed? 
    # Let's use the actual value (1..10000) as the bit index.
    # Bit i represents node i.
    
    def find_clique_bitmask(valid_sum_arr, clique_size, time_limit):
        # 1. Determine active nodes
        # A node i is active if it has at least clique_size-1 potential neighbors
        # Potential neighbor j: 1 <= j <= 10000, j != i, and valid_sum_arr[i+j] is True
        
        # To speed up, we can filter nodes.
        # But first, let's just build the adjacency for all 1..10000.
        # N = 10000
        N = 10000
        
        # adj[i] is a bitmask of neighbors of i
        # We only care about neighbors j > i to avoid redundancy? 
        # Tomita's algorithm usually works on the whole graph but orders vertices.
        # Let's store adj[i] as a bitmask of all j such that i+j is valid and j != i.
        
        adj = [0] * (N + 1)
        
        # Optimization: Only include nodes that have high degree?
        # Let's compute degrees first to filter.
        degrees = [0] * (N + 1)
        
        # Iterate i from 1 to N
        # For each i, check j from 1 to N
        # This is 100M operations. Might be too slow in Python.
        # Vectorize with NumPy?
        
        # Let's use NumPy to build the adjacency matrix or at least filter nodes.
        # valid_sum_arr is a boolean array of size MAX_SUM+1.
        
        # We want to find a clique of size 12.
        # If the graph is dense enough, we can just pick random valid nodes and expand?
        # No, clique search is safer.
        
        # Let's try to construct the graph more efficiently.
        # For a fixed i, neighbors j satisfy valid_sum_arr[i+j].
        # j ranges from 1 to N.
        # i+j ranges from i+1 to i+N.
        
        # We can precompute for each i the list of j's?
        # Or use bit operations on the valid_sum_arr?
        
        # Let's convert valid_sum_arr to a numpy array of uint8 or bool
        vs = np.array(valid_sum_arr, dtype=bool)
        
        # For each i, the mask of valid j's is vs[i+1 : i+N+1]
        # We can't easily make this a Python int bitmask directly from numpy slice without conversion.
        # Conversion of 10000 bits is fast.
        
        # However, building 10000 such masks takes 10000 * (slice + conversion).
        # Let's try it.
        
        adj_masks = [0] * (N + 1)
        
        # Pre-filter: Nodes with degree < clique_size-1 cannot be in the clique.
        # We'll compute degrees roughly first.
        
        # Actually, let's just build the masks and then filter.
        for i in range(1, N + 1):
            # Get slice of valid sums
            # j is 1..N, so i+j is i+1 .. i+N
            # Check bounds
            lo = i + 1
            hi = i + N
            if hi > MAX_SUM:
                # Pad or slice
                # vs is size MAX_SUM+1
                # We need vs[lo:hi+1]
                # If hi > MAX_SUM, we just take up to MAX_SUM
                # But j must be <= N.
                # If i+N > MAX_SUM, then for j > MAX_SUM - i, i+j > MAX_SUM.
                # Since MAX_SUM = 20000 and N=10000, i+N <= 20000 always.
                pass
            
            # Slice
            # Note: vs is 0-indexed. Index s corresponds to sum s.
            # We need j such that vs[i+j] is True.
            # j ranges 1 to N.
            # Let's create a boolean array of length N for neighbors of i.
            # neighbor_mask[j] = vs[i+j]
            
            # Using numpy:
            # sums = np.arange(i+1, i+N+1)
            # neighbors = vs[sums]
            # This creates 10000 arrays of size 10000. Total 100M bools. 100MB.
            # Might be okay.
            
            # Faster:
            # We can use the fact that vs is a boolean array.
            # Let's just do a loop if it's not too slow, or use numpy.
            
            # Let's use numpy to generate the mask for all i at once?
            # No, let's just build the masks.
            
            # Optimization: If time is tight, skip building full graph and use heuristic.
            # But let's try the bitset approach properly.
            
            # To save time, we can compute degrees first using numpy convolution or similar?
            # Degree of i = sum(vs[i+1 : i+N+1])
            
            # Let's compute degrees for all i using numpy.
            # This is a sliding window sum on vs.
            # vs is length 20001.
            # We want sum of vs[i+1 ... i+10000] for i=1..10000.
            
            # cumsum = np.cumsum(vs)
            # degree[i] = cumsum[i+N] - cumsum[i]  (careful with indices)
            
            # Let's do this.
            pass
        
        # Let's restart the graph building with the cumsum trick for degrees.
        # This allows us to filter nodes quickly.
        
        cumsum = np.cumsum(vs.astype(np.int32))
        # deg[i] = number of j in 1..N such that vs[i+j] is True
        # j=1 -> i+1, j=N -> i+N
        # sum = cumsum[i+N] - cumsum[i]
        # Note: cumsum[k] = sum(vs[0..k])
        # We want sum(vs[i+1..i+N]) = cumsum[i+N] - cumsum[i]
        
        # i ranges 1..10000
        # i+N ranges 10001..20000
        # i ranges 1..10000
        
        degs = np.zeros(N + 1, dtype=np.int32)
        # Vectorized calculation
        # i_arr = np.arange(1, N+1)
        # degs[i_arr] = cumsum[i_arr + N] - cumsum[i_arr]
        
        i_arr = np.arange(1, N + 1)
        degs[i_arr] = cumsum[i_arr + N] - cumsum[i_arr]
        
        # Filter nodes with deg >= clique_size - 1
        min_deg = clique_size - 1
        valid_nodes = i_arr[degs[i_arr] >= min_deg]
        
        if len(valid_nodes) < clique_size:
            return None
        
        # Map valid nodes to 0..M-1 for compact bitmask?
        # Or just use the original indices in a sparse way?
        # Python ints can handle bits up to 10000.
        # Let's use the original indices (1..10000) as bit positions.
        
        # Build adjacency masks for valid nodes only?
        # We need adj[i] to only contain bits for valid nodes j.
        
        adj_masks = [0] * (N + 1)
        
        # For each valid node i, determine which valid nodes j are neighbors.
        # j is a neighbor if j is in valid_nodes and vs[i+j] is True.
        
        # To speed this up, we can iterate through valid_nodes for i and j.
        # M = len(valid_nodes)
        # If M is small (e.g., < 1000), M^2 is 1M, which is fast.
        # If M is large, we need to be careful.
        
        M = len(valid_nodes)
        
        # Create a set of valid nodes for O(1) lookup
        valid_set = set(valid_nodes)
        
        # Build adj_masks
        # We can iterate i in valid_nodes
        # For each i, we need to find j in valid_nodes such that vs[i+j] is True.
        
        # Optimization:
        # For a given i, the possible sums are i+j.
        # We can check vs[i+j] for each j in valid_nodes.
        
        # Let's try to build it.
        for i in valid_nodes:
            mask = 0
            # Iterate through all valid nodes j
            # This is O(M) per i, total O(M^2)
            # If M is 1000, 1M ops. Fast.
            # If M is 10000, 100M ops. Might be slow in Python.
            
            # Let's check M first.
            if M > 2000:
                # Use a different strategy? 
                # Maybe sample? Or use numpy?
                # Let's stick to Python loop but break early if we have enough neighbors?
                # No, we need the full mask for clique search.
                
                # Alternative: 
                # For each i, the neighbors are a subset of valid_nodes.
                # We can construct the mask by checking vs[i+j] for j in valid_nodes.
                
                # Let's try to optimize the inner loop.
                # We can't easily vectorize the bit setting.
                
                # Let's just run it. 100M simple ops in Python might take ~10-20s.
                # We have 85s.
                
                pass
            
            for j in valid_nodes:
                if i == j:
                    continue
                if vs[i + j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        # Now we have adj_masks for all valid nodes.
        
        # Tomita's Algorithm
        # We need to find a clique of size clique_size.
        
        # Order nodes by degree in the subgraph induced by valid_nodes?
        # Or just use the order they appear?
        # Tomita's algorithm benefits from good ordering.
        
        # Let's sort valid_nodes by degree (degs[i]) descending.
        # This helps in finding a large clique quickly for the bound.
        
        ordered_nodes = sorted(valid_nodes, key=lambda x: degs[x], reverse=True)
        
        # Map node value to index in ordered_nodes?
        # Or just use the node values as keys in a dict?
        # The bitmask uses node values as bit indices.
        
        # State:
        # P: bitmask of candidates
        # X: bitmask of excluded (not strictly needed for max clique if we just return True/False)
        # current_clique: list of nodes
        # depth: len(current_clique)
        
        best_clique = []
        found = False
        
        # Upper bound for clique size
        # We just want to know if size >= clique_size exists.
        
        def expand(clique, P, depth):
            nonlocal found, best_clique
            
            if found:
                return
            
            if len(clique) == clique_size:
                found = True
                best_clique = clique[:]
                return
            
            # Upper bound: len(clique) + popcount(P)
            if len(clique) + bin(P).count('1') < clique_size:
                return
            
            # If P is empty, return
            if P == 0:
                return
            
            # Select a pivot vertex u from P
            # u should maximize the number of neighbors in P
            # To speed up, we can just pick the first one, or one with high degree.
            # For simplicity, pick the lowest set bit.
            # Better: pick the one with max degree in P.
            
            # Getting max degree in P requires iterating bits.
            # Let's just pick the first available bit for now to keep it simple and fast.
            # Actually, Tomita's algorithm says:
            # Partition P into P1 (neighbors of u) and P2 (non-neighbors of u).
            # Recurse on P1.
            # Then remove u from P and recurse on P2.
            
            # Pick u
            u_bit = P & (-P)
            u = u_bit.bit_length() - 1
            
            # Neighbors of u in P
            N_u = adj_masks[u] & P
            
            # P1 = N_u
            # P2 = P & ~N_u (excluding u? No, u is in N_u? No, u is not neighbor of itself)
            # Wait, adj_masks[u] does NOT include u.
            # So N_u contains neighbors of u in P.
            # P1 = N_u
            # P2 = P & ~N_u  (This includes u? No, u is not in N_u. 
            # But u is in P. So u will be in P2?
            # If u is in P2, we will consider it in the second branch.
            # But we already "used" u as pivot.
            # Standard Tomita:
            # 1. Expand(clique + [u], N_u, depth+1)
            # 2. P = P - {u}
            # 3. Expand(clique, P & ~N_u, depth)  -- Wait, N_u was calculated with u in P?
            #    N_u = adj[u] & P. Since adj[u] doesn't have u, N_u doesn't have u.
            #    So P & ~N_u contains u.
            #    We should remove u from P before the second call?
            #    Yes.
            
            # Branch 1: Include u
            new_clique = clique + [u]
            expand(new_clique, N_u, depth + 1)
            
            if found:
                return
            
            # Branch 2: Exclude u
            # Remove u from P
            P = P & ~u_bit
            # Recurse on non-neighbors of u (excluding u)
            # N_u is neighbors of u in original P.
            # We want P_new = P (which is P_orig - {u}) intersected with (NOT neighbors of u).
            # Non-neighbors of u in P_orig = P_orig & ~adj_masks[u]
            # Since u is not in adj_masks[u], u is in Non-neighbors.
            # But we already removed u from P.
            # So P_new = P & ~adj_masks[u]
            
            P2 = P & ~adj_masks[u]
            
            expand(clique, P2, depth)
        
        # Initial P: bitmask of all valid nodes
        initial_P = 0
        for node in valid_nodes:
            initial_P |= (1 << node)
        
        # We need to order the expansion.
        # The recursive call above picks the lowest bit.
        # To improve, we can sort the bits in initial_P by degree.
        # But the algorithm picks `u_bit = P & (-P)` which is the lowest bit.
        # This is equivalent to processing nodes in increasing order of their value.
        # This might be suboptimal.
        
        # Let's modify `expand` to pick the node with highest degree in P.
        # This requires iterating through bits of P.
        
        def expand_opt(clique, P, depth):
            nonlocal found, best_clique
            
            if found:
                return
            
            if len(clique) == clique_size:
                found = True
                best_clique = clique[:]
                return
            
            if P == 0:
                return
            
            # Upper bound
            if len(clique) + bin(P).count('1') < clique_size:
                return
            
            # Find node u in P with max degree in P
            # Iterate through bits
            max_deg = -1
            u = 0
            temp_P = P
            while temp_P:
                bit = temp_P & (-temp_P)
                node = bit.bit_length() - 1
                # Degree of node in P
                # deg_in_P = bin(adj_masks[node] & P).count('1')
                # This is expensive.
                # Let's just use the precomputed global degree as a heuristic?
                # Or just pick the first one?
                # Let's try picking the one with max global degree first.
                # We can pre-sort the bits? No, P changes.
                
                # Let's stick to lowest bit for speed, but maybe randomize?
                # Actually, for small cliques (12), lowest bit is fine if graph is dense.
                pass
            
            # Let's just use the lowest bit for now.
            u_bit = P & (-P)
            u = u_bit.bit_length() - 1
            
            N_u = adj_masks[u] & P
            
            # Branch 1
            expand_opt(clique + [u], N_u, depth + 1)
            if found:
                return
            
            # Branch 2
            P = P & ~u_bit
            P2 = P & ~adj_masks[u]
            expand_opt(clique, P2, depth)

        # To make it faster, let's try a greedy initialization first.
        # Pick nodes with high degree.
        
        # Run the search
        # To avoid recursion depth issues (though depth is only 12), 
        # and to allow pruning, we use the function above.
        
        # One issue: `bin(P).count('1')` is O(bits). 
        # For 10000 bits, it's fast enough.
        
        # Let's add a time check inside the recursion?
        # Python recursion overhead is high.
        # Let's limit the search space by only considering top M nodes by degree.
        
        # If M is too large, the search might explode.
        # But we filtered by degree >= 11.
        
        # Let's try.
        
        # To speed up, we can use a non-recursive version or just trust the depth 12.
        
        # Let's run it.
        
        # If M is very large, we might timeout.
        # Let's add a hard stop.
        
        # For k=2 (powers of 2), the graph is very sparse.
        # Degree of i is number of j such that i+j is power of 2.
        # Max power of 2 is 16384.
        # For a fixed i, j must be 2^k - i.
        # There are at most log2(20000) ~ 14 such j's.
        # So degree is very small (< 15).
        # So M (nodes with deg >= 11) will be small!
        
        # For k=3 (5-smooth), degree is higher.
        
        # So the graph is sparse, and M will be manageable.
        
        expand_opt([], initial_P, 0)
        
        if found:
            return best_clique
        else:
            return None

    # Try k=2
    # valid_sums_cache[0] is for [2]
    clique = find_clique_bitmask(valid_sums_cache[0], 12, deadline)
    if clique:
        return clique

    # Try k=3
    clique = find_clique_bitmask(valid_sums_cache[1], 12, deadline)
    if clique:
        return clique

    # Try k=4 (if time permits)
    if time.time() < deadline - 10:
        clique = find_clique_bitmask(valid_sums_cache[2], 12, deadline)
        if clique:
            return clique

    # Fallback: Local Search
    import random
    random.seed(42)
    
    best_score = 999
    best_arr = list(range(1, 13))
    
    def count_distinct_primes(arr):
        all_primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                all_primes |= get_prime_factors(s)
        return len(all_primes)
        
    best_score = count_distinct_primes(best_arr)
    current_arr = best_arr[:]
    current_score = best_score
    
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        score = count_distinct_primes(current_arr)
        
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
            
    return best_arr

# Helper for fallback
def get_prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors
