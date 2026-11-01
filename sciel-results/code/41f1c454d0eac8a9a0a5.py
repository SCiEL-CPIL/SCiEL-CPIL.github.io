import random
import time
import itertools
import numpy as np

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n > 4, use heuristic optimization with precomputed overlaps
    def get_standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    std_word = get_standard_superperm(n)
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Get standard order of permutations
    std_order = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        w = tuple(std_word[i:i+n])
        if w not in seen:
            seen.add(w)
            std_order.append(w)
            
    if len(std_order) != len(perms):
        return "".join(map(str, std_word))
    
    N = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Use a faster method: for each pair, compute overlap
    # But N=5040, so N^2 = 25M pairs. We need to be fast.
    
    # Alternative: For each permutation, build a trie of its prefixes, and for each other permutation,
    # check its suffixes against the trie. But that's complex.
    
    # Let's use a hash-based approach:
    # For each permutation p, store all its suffixes in a dict: suffix_tuple -> length
    # Then for overlap(p, q), check if q's prefix of length k matches p's suffix of length k.
    # But we need the max k.
    
    # Better: Precompute using numpy for speed.
    # Convert perms to a numpy array of shape (N, n)
    perm_array = np.array(perms, dtype=np.int8)
    
    # Compute overlap matrix
    # For each i, j, we want max k in [0, n-1] such that perm_array[i, n-k:] == perm_array[j, :k]
    # This is O(n) per pair, so O(N^2 * n) total = 25M * 7 = 175M ops. Might be slow in Python.
    
    # Let's use a smarter approach:
    # For each k from n-1 down to 0, create a hash map from the suffix of length k to the list of indices that have that suffix.
    # Then for each pair (i, j), check if the prefix of j of length k matches the suffix of i of length k.
    
    # Actually, let's just compute it directly but with early termination.
    overlap_matrix = np.zeros((N, N), dtype=np.int8)
    
    # For each i, for each j, compute overlap
    # To speed up, we can vectorize partially
    for i in range(N):
        p_i = perm_array[i]
        for k in range(n - 1, -1, -1):
            if k == 0:
                # All pairs have at least overlap 0
                continue
            suffix_i = tuple(p_i[n-k:])
            # Find all j such that prefix of j of length k equals suffix_i
            # We can precompute a dict for this
            # But let's just do it simply for now
    
    # This is getting complex. Let's use a different strategy:
    # Precompute for each permutation its "suffix signatures" and "prefix signatures"
    # For each k, create a dict: suffix_k_tuple -> set of indices
    # Then for each pair (i, j), the overlap is the max k such that suffix_k(i) == prefix_k(j)
    
    # Build suffix dicts
    suffix_dicts = [dict() for _ in range(n)]
    prefix_dicts = [dict() for _ in range(n)]
    
    for idx, p in enumerate(perms):
        for k in range(1, n):
            suffix = tuple(p[n-k:])
            prefix = tuple(p[:k])
            if suffix not in suffix_dicts[k]:
                suffix_dicts[k][suffix] = []
            suffix_dicts[k][suffix].append(idx)
            if prefix not in prefix_dicts[k]:
                prefix_dicts[k][prefix] = []
            prefix_dicts[k][prefix].append(idx)
    
    # Now compute overlap matrix
    # For each k from n-1 down to 1, for each suffix in suffix_dicts[k],
    # the indices that have this suffix can overlap with indices that have the same prefix
    # But we need overlap[i][j] for specific i, j pairs.
    
    # Let's just compute it directly with a fast inner loop
    # Use the fact that overlap is at most n-1
    
    # Alternative: Use a lookup table based on the last element and first element
    # If p_i[-1] != p_j[0], overlap is 0
    # This prunes many cases
    
    # Let's precompute last elements and first elements
    last_elems = perm_array[:, -1]
    first_elems = perm_array[:, 0]
    
    # Build overlap matrix
    # Initialize to 0
    # For k from n-1 down to 1:
    #   For each i, if there exists j such that p_i[n-k:] == p_j[:k], set overlap[i][j] = max(overlap[i][j], k)
    
    # This is still O(N^2 * n) in the worst case.
    
    # Let's just compute it and hope it's fast enough with numpy optimizations
    # Actually, let's use a simpler approach: compute overlap on the fly but cache it
    
    # For now, let's use a dict to cache overlaps
    overlap_cache = {}
    
    def get_overlap(i, j):
        key = (i, j)
        if key in overlap_cache:
            return overlap_cache[key]
        p_i = perms[i]
        p_j = perms[j]
        max_k = n - 1
        for k in range(max_k, -1, -1):
            if k == 0:
                result = 0
                break
            if p_i[n-k:] == p_j[:k]:
                result = k
                break
        overlap_cache[key] = result
        return result
    
    # This is O(n) per new pair, but cached after first access.
    # In local search, we'll access the same pairs many times.
    
    # Current order as indices
    current_order_idx = [perm_to_idx[p] for p in std_order]
    
    def calc_cost(order_idx):
        total = 0
        for i in range(len(order_idx) - 1):
            ov = get_overlap(order_idx[i], order_idx[i+1])
            total += n - ov
        return total
    
    current_cost = calc_cost(current_order_idx)
    
    start_time = time.time()
    time_limit = 110.0
    
    # Local search with 2-opt moves
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try 2-opt: reverse a segment
        # For i < j, reverse order[i..j]
        # The affected edges are (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
        # Actually, only edges (i-1, i), (j, j+1), and the internal edges change
        # But reversing changes all internal edges
        
        # Let's try adjacent swaps first (2-opt with j=i+1)
        for i in range(N - 1):
            if time.time() - start_time > time_limit:
                break
            # Swap i and i+1
            a = current_order_idx[i]
            b = current_order_idx[i+1]
            
            # Old cost of affected edges
            old_cost = 0
            if i > 0:
                old_cost += n - get_overlap(current_order_idx[i-1], a)
            old_cost += n - get_overlap(a, b)
            if i + 1 < N - 1:
                old_cost += n - get_overlap(b, current_order_idx[i+2])
            
            # New cost
            new_cost = 0
            if i > 0:
                new_cost += n - get_overlap(current_order_idx[i-1], b)
            new_cost += n - get_overlap(b, a)
            if i + 1 < N - 1:
                new_cost += n - get_overlap(a, current_order_idx[i+2])
            
            if new_cost < old_cost:
                current_order_idx[i] = b
                current_order_idx[i+1] = a
                current_cost += new_cost - old_cost
                improved = True
        
        # Try 2-opt: reverse segment [i, j] for random i, j
        if not improved:
            for _ in range(200):
                if time.time() - start_time > time_limit:
                    break
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                
                # Reverse segment [i, j]
                segment = current_order_idx[i:j+1]
                reversed_segment = segment[::-1]
                
                # Compute old cost of affected edges
                old_cost = 0
                # Edge (i-1, i)
                if i > 0:
                    old_cost += n - get_overlap(current_order_idx[i-1], current_order_idx[i])
                # Internal edges
                for k in range(i, j):
                    old_cost += n - get_overlap(current_order_idx[k], current_order_idx[k+1])
                # Edge (j, j+1)
                if j < N - 1:
                    old_cost += n - get_overlap(current_order_idx[j], current_order_idx[j+1])
                
                # New cost
                new_cost = 0
                if i > 0:
                    new_cost += n - get_overlap(current_order_idx[i-1], reversed_segment[0])
                for k in range(len(reversed_segment) - 1):
                    new_cost += n - get_overlap(reversed_segment[k], reversed_segment[k+1])
                if j < N - 1:
                    new_cost += n - get_overlap(reversed_segment[-1], current_order_idx[j+1])
                
                if new_cost < old_cost:
                    current_order_idx[i:j+1] = reversed_segment
                    current_cost += new_cost - old_cost
                    improved = True
    
    # Reconstruct word
    word = list(perms[current_order_idx[0]])
    for i in range(1, N):
        p_prev = perms[current_order_idx[i-1]]
        p_curr = perms[current_order_idx[i]]
        ov = get_overlap(current_order_idx[i-1], current_order_idx[i])
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
