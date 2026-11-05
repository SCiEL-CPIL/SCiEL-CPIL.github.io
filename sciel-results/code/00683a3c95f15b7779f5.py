import time
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We aim for n=20. 
    # If we fail, we return the best valid graph found.
    # A graph with n < 6 is trivially valid.
    # We know n=20 is possible.
    
    best_n = 5
    best_edges = []
    
    # Try to construct a valid graph for n=20 down to 6
    for target_n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        # Backtracking search
        # adj[i] is a bitmask of neighbors for vertex i
        adj = [0] * target_n
        # degrees[i] is the current degree of vertex i
        degrees = [0] * target_n
        
        # We will decide edges (i, j) in a fixed order.
        # Order: all pairs (i, j) with i < j.
        # However, standard backtracking on edges is too slow.
        # Better: Decide the neighborhood of vertex k (edges from k to k+1...n-1) one by one?
        # Or: Decide vertex by vertex.
        
        # Let's try a different backtracking strategy:
        # We build the graph by adding vertices 0, 1, ..., n-1.
        # When adding vertex k, we decide its connections to 0, ..., k-1.
        # Constraint: No induced subgraph of size 6 is regular.
        # A subset S is regular if all vertices in S have the same degree within S.
        
        # Optimization: 
        # When adding vertex k, we only need to check subsets of size 6 that include k.
        # Let S be a subset of {0, ..., k-1} of size 5.
        # The subset S U {k} must NOT be regular.
        # Let d_v be the degree of v in S (within S).
        # Let e_v be 1 if v is connected to k, 0 otherwise.
        # The degree of v in S U {k} is d_v + e_v.
        # The degree of k in S U {k} is sum(e_v for v in S).
        # The subset is regular if d_v + e_v is constant for all v in S, AND equal to sum(e_v).
        
        # We can precompute the "pattern" of degrees for every subset of size 5 from {0, ..., k-1}.
        # But this is expensive to update.
        
        # Alternative Fast Constraint Check:
        # For a fixed new vertex k and a fixed subset S of size 5 from previous vertices:
        # We want to choose a bitmask E (edges from k to S) such that S U {k} is NOT regular.
        # This is a constraint on E.
        
        # Let's implement a simple backtracking on the edges of the last vertex.
        # Actually, let's just do a simple DFS on the adjacency matrix, but with strong pruning.
        
        # State:
        # adj[0..k-1] are fixed.
        # We are choosing adj[k] (edges to 0..k-1).
        
        # To make it faster, we can use a heuristic:
        # Randomize the order of vertices? No, deterministic is better for reproducibility.
        # But maybe a specific order helps.
        
        # Let's try a different approach: **Simulated Annealing with a very fast verifier** 
        # but starting from a "smart" initial guess.
        # The previous program failed because the verifier was too slow or the search was weak.
        
        # Let's try a **Genetic Algorithm** or **Memetic Algorithm**?
        # No, let's stick to **Backtracking** but optimized.
        
        # Optimization for Backtracking:
        # When adding vertex k, we iterate through all 2^k possible connection patterns to previous vertices.
        # For each pattern, we check if it creates a regular induced subgraph of size 6.
        # If it does, we skip.
        # If we run out of patterns, we backtrack.
        
        # This is still potentially 2^20 which is 1M states. With pruning, it might be fast.
        # The check for "creates regular induced subgraph of size 6" involving k:
        # We need to check all C(k, 5) subsets S of previous vertices.
        # For each S, we check if S U {k} is regular given the current connection pattern.
        
        # This check is O(k^5) per pattern? No, O(C(k,5)) which is ~ k^5/120.
        # For k=20, C(20,5) = 15504.
        # If we try 2^20 patterns, and each takes 15k checks... too slow.
        
        # We need to prune earlier.
        
        # Let's try a different strategy: **Pre-computed Valid Graph**.
        # Since n=20 is known to exist, and random search is hard, maybe we can use a known construction?
        # Or, we can use a **SAT Solver** approach? We don't have a SAT solver.
        
        # Let's try **Local Search** again but with a **much faster verifier** using NumPy.
        # The bottleneck in the previous program was likely the Python loop for verification.
        
        def verify_fast(n, adj_masks):
            """
            Verify if graph with adj_masks has no regular induced subgraph of size >= 6.
            Returns True if valid.
            Uses bit operations for speed.
            """
            # We only need to check size 6? 
            # If a graph has a regular induced subgraph of size k > 6, does it necessarily have one of size 6?
            # No. Example: A 7-vertex regular graph (degree 3) might not have a 6-vertex regular induced subgraph.
            # However, for the purpose of finding a valid graph, minimizing size-6 violations is a good proxy.
            # If we find a graph with 0 size-6 violations, we must still check larger sizes.
            # But checking larger sizes is rare if size-6 is clean.
            
            # Check size 6
            # Iterate over all combinations of 6 vertices
            # There are C(n, 6) of them.
            # For n=20, C(20,6) = 38760.
            # This is small enough to do in a tight loop.
            
            for subset in itertools.combinations(range(n), 6):
                # Calculate degrees of each vertex in the subset
                # deg[v] = popcount(adj_masks[v] & mask_subset)
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask_subset).bit_count())
                
                # Check if all degrees are equal
                if len(set(degs)) == 1:
                    return False
            
            # If no size-6 violation, check larger sizes?
            # For safety, let's check size 7 to n.
            # But this is much more expensive.
            # C(20, 7) = 77520.
            # C(20, 20) = 1.
            # Total subsets is 2^20 - sum(C(20,k) for k<6) ~ 1M.
            # Checking 1M subsets in Python is ~1-2 seconds.
            # We have 115 seconds. We can afford a few full checks.
            
            for k in range(7, n + 1):
                for subset in itertools.combinations(range(n), k):
                    mask_subset = 0
                    for v in subset:
                        mask_subset |= (1 << v)
                    
                    degs = []
                    for v in subset:
                        degs.append((adj_masks[v] & mask_subset).bit_count())
                    
                    if len(set(degs)) == 1:
                        return False
            
            return True

        # Local Search with Fast Verifier
        import random
        random.seed(42)
        
        found = False
        # Try a few random restarts
        for restart in range(100):
            if time.time() > deadline:
                break
            
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            if verify_fast(target_n, adj_masks):
                # Found a valid graph!
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
                break
            
            # If not found, do a few local search steps?
            # Actually, random restarts with a fast verifier might be enough.
            # If verify_fast is ~1s, 100 restarts = 100s. Too slow.
            # We need to be smarter.
            
            # Let's optimize verify_fast to exit early.
            # The code above does exit early on the first violation.
            # Most random graphs will have violations.
            # So verify_fast should be fast on average.
            
        if found:
            return {'n': best_n, 'edges': best_edges}
        
        # If we didn't find one for this target_n, try smaller
        continue

    # If we failed to find n=20..6, return n=5 (trivially valid)
    # But we should return the best valid graph we found.
    # Since we loop from 20 down, if we didn't break, we didn't find any.
    # So return n=5.
    
    return {'n': best_n, 'edges': best_edges}

# To ensure we don't timeout, we can add a faster check for size 6 only first.
# But the current verify_fast checks size 6 first and exits early.

# Let's refine the strategy:
# 1. Check size 6 only. If valid, check larger sizes.
# 2. Use NumPy for vectorized degree calculation?
#    For a fixed subset, calculating degrees is a matrix-vector product?
#    No, it's just bit counts.

# Let's try to make the random generation smarter.
# Instead of p=0.5, maybe p=0.4 or 0.6?
# Regular graphs are more likely to appear in dense or sparse graphs?
# Actually, random graphs with p=0.5 are "most random" and least likely to have structure?
# Or are they?
# It's a balance.

# Let's stick to the plan: Fast verifier + Random Restarts.
# The key is the speed of verify_fast.

# One more optimization:
# Precompute all masks of size 6?
# There are 38760 masks.
# Storing them is fine.
# Then verify_fast iterates over precomputed masks.

def _precompute_masks(n, k):
    masks = []
    for subset in itertools.combinations(range(n), k):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        masks.append(mask)
    return masks

# But we need the vertices to calculate degrees.
# We can store (mask, vertices_list)

# Actually, the current verify_fast is probably fast enough if it exits early.
# Let's assume the first few subsets are checked quickly.

# Final check on the code structure.
# The `construct` function is defined.
# It returns a dict.

# One potential issue: `itertools.combinations` is slow in Python.
# For n=20, C(20,6) is 38k.
# 38k iterations is ~0.01s.
# So verify_fast is ~0.01s if it finds a violation early.
# If it doesn't find a violation, it takes ~1s.
# So random restarts are very cheap if they fail early.
# We can do thousands of restarts.

# Let's increase the number of restarts.

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 5
    best_edges = []
    
    for target_n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        import random
        random.seed(42)
        
        found = False
        restarts = 0
        max_restarts = 5000
        
        while restarts < max_restarts and time.time() < deadline:
            restarts += 1
            
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Fast Check
            is_valid = True
            
            # Check size 6
            for subset in itertools.combinations(range(target_n), 6):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask_subset).bit_count())
                
                if len(set(degs)) == 1:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Check larger sizes
            for k in range(7, target_n + 1):
                for subset in itertools.combinations(range(target_n), k):
                    mask_subset = 0
                    for v in subset:
                        mask_subset |= (1 << v)
                    
                    degs = []
                    for v in subset:
                        degs.append((adj_masks[v] & mask_subset).bit_count())
                    
                    if len(set(degs)) == 1:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
        
        continue

    return {'n': best_n, 'edges': best_edges}

# This looks solid. The key is that `verify` is fast on failure.
# We are betting that random graphs fail fast.

# Let's add a small optimization: 
# If target_n < 6, return immediately.
# But we start from 20.

# One more thing: The problem says "Maximize n".
# If we find n=20, we return.
# If we don't, we try 19.
# This is correct.

# Final Code Block

import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 5
    best_edges = []
    
    for target_n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        random.seed(42)
        
        found = False
        restarts = 0
        max_restarts = 10000
        
        while restarts < max_restarts and time.time() < deadline:
            restarts += 1
            
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            is_valid = True
            
            # Check size 6
            for subset in itertools.combinations(range(target_n), 6):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask_subset).bit_count())
                
                if len(set(degs)) == 1:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Check larger sizes
            for k in range(7, target_n + 1):
                for subset in itertools.combinations(range(target_n), k):
                    mask_subset = 0
                    for v in subset:
                        mask_subset |= (1 << v)
                    
                    degs = []
                    for v in subset:
                        degs.append((adj_masks[v] & mask_subset).bit_count())
                    
                    if len(set(degs)) == 1:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
        
        continue

    return {'n': best_n, 'edges': best_edges}
