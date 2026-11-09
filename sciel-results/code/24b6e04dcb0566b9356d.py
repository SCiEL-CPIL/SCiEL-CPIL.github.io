import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = 120  # Safety margin from 150s
    
    # We will try n=21, then 20, then 19...
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        # Precompute subsets of size 6, 7, ..., n
        # For speed, we start with size 6 only, then expand if needed
        # But to be safe, let's include 6 and 7 in the primary search if n is small
        # For n=21, size 6 is 54k subsets. Size 7 is 116k.
        # Let's stick to size 6 for the main loop to be fast.
        
        subsets_6 = [mask for mask in (sum(1 << v for v in comb) for comb in itertools.combinations(range(n), 6))]
        # Note: The generator expression above is a bit complex, let's simplify
        subsets_6 = []
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_6.append(mask)
            
        # For checking larger sizes, we will do it in a separate function
        # because iterating over all of them in the inner loop is too slow.
        
        def is_regular_subset(A, mask):
            # Check if induced subgraph on mask is regular
            # Get vertices in mask
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            if not verts:
                return True
            
            degs = []
            for v in verts:
                # Count neighbors of v within mask
                degs.append(bin(A[v] & mask).count('1'))
            
            return len(set(degs)) == 1

        def count_violations(A, subsets):
            count = 0
            for mask in subsets:
                if is_regular_subset(A, mask):
                    count += 1
            return count

        def find_valid_graph(n, max_time):
            # Local search
            best_graph = None
            best_penalty = float('inf')
            
            for restart in range(50): # Limit restarts
                if time.time() > max_time:
                    break
                
                # Initialize random graph
                A = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            A[u] |= (1 << v)
                            A[v] |= (1 << u)
                
                # Initial penalty
                # We only count size 6 violations for speed in the loop
                penalty = count_violations(A, subsets_6)
                
                if penalty == 0:
                    # Check larger sizes
                    larger_violation = False
                    for k in range(7, n + 1):
                        for comb in itertools.combinations(range(n), k):
                            mask = 0
                            for v in comb:
                                mask |= (1 << v)
                            if is_regular_subset(A, mask):
                                larger_violation = True
                                break
                        if larger_violation:
                            break
                    
                    if not larger_violation:
                        return A
                
                # Local search steps
                stuck_count = 0
                for step in range(1000):
                    if time.time() > max_time:
                        break
                    
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    
                    # Flip edge (u, v)
                    edge_exists = (A[u] >> v) & 1
                    if edge_exists:
                        A[u] ^= (1 << v)
                        A[v] ^= (1 << u)
                    else:
                        A[u] ^= (1 << v)
                        A[v] ^= (1 << u)
                    
                    # Calculate delta penalty
                    # Only subsets containing u or v are affected
                    # This is still expensive if we check all 54k subsets.
                    # Optimization: Precompute for each vertex u, the list of subsets_6 containing u.
                    # subsets_containing[u] = [mask for mask in subsets_6 if mask & (1<<u)]
                    
                    # Let's precompute this once outside the loop
                    # But we are inside find_valid_graph. Let's do it there.
                    
                    # Actually, recalculating penalty from scratch for 54k subsets is ~0.1s.
                    # 1000 steps * 0.1s = 100s. Too slow.
                    
                    # We MUST use incremental update or a faster penalty.
                    # Let's precompute subsets_containing for size 6.
                    pass # This logic is flawed for speed.
                
                # Fallback: If simple local search is too slow, use a heuristic
                # that checks only a sample of subsets? No, must be exact.
                
            return None

        # Precompute subsets containing each vertex for speed
        subsets_containing = [[] for _ in range(n)]
        for mask in subsets_6:
            for v in range(n):
                if mask & (1 << v):
                    subsets_containing[v].append(mask)

        def get_penalty(A):
            # Count regular size-6 subgraphs
            # We can cache? No, graph changes.
            # Let's just count.
            c = 0
            for mask in subsets_6:
                if is_regular_subset(A, mask):
                    c += 1
            return c

        # Revised Local Search with Incremental Update is complex.
        # Let's try a simpler approach:
        # 1. Random restarts.
        # 2. For each restart, do a few steps of "greedy" improvement.
        #    To make it fast, we only check if the *new* penalty is lower by
        #    checking ONLY the affected subsets.
        
        def calculate_delta_penalty(A, u, v, old_penalty):
            # This is hard to implement correctly without global state of "which subsets are currently regular".
            # Let's store a set of currently regular masks.
            pass

        # Given the complexity, let's use a different strategy:
        # Use a known 20-vertex graph if we can find one via a quick search.
        # If not, return a random 20-vertex graph? No, it must be valid.
        
        # Let's try to find a valid graph on n=20 using a simpler heuristic:
        # The complement of a random graph?
        
        # Actually, let's just run a local search that is slightly slower but correct.
        # We will limit the number of steps.
        
        def search(n):
            subsets_containing = [[] for _ in range(n)]
            for mask in subsets_6:
                for v in range(n):
                    if mask & (1 << v):
                        subsets_containing[v].append(mask)
            
            for restart in range(20):
                if time.time() > deadline - 5:
                    return None
                
                A = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            A[u] |= (1 << v)
                            A[v] |= (1 << u)
                
                # Initialize set of regular masks
                regular_masks = set()
                for mask in subsets_6:
                    if is_regular_subset(A, mask):
                        regular_masks.add(mask)
                
                current_penalty = len(regular_masks)
                
                for step in range(500):
                    if time.time() > deadline - 5:
                        return None
                    
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    
                    # Flip edge
                    edge_exists = (A[u] >> v) & 1
                    A[u] ^= (1 << v)
                    A[v] ^= (1 << u)
                    
                    # Update regular_masks
                    # Affected masks are those containing u or v
                    affected = set(subsets_containing[u]) | set(subsets_containing[v])
                    
                    new_penalty = current_penalty
                    
                    # Remove masks that are no longer regular
                    for mask in list(regular_masks):
                        if mask in affected:
                            if not is_regular_subset(A, mask):
                                regular_masks.remove(mask)
                                new_penalty -= 1
                    
                    # Add masks that are now regular
                    for mask in affected:
                        if mask not in regular_masks:
                            if is_regular_subset(A, mask):
                                regular_masks.add(mask)
                                new_penalty += 1
                    
                    if new_penalty < current_penalty:
                        current_penalty = new_penalty
                    else:
                        # Revert
                        A[u] ^= (1 << v)
                        A[v] ^= (1 << u)
                        # Restore regular_masks? 
                        # This is getting complicated. 
                        # Simpler: Just recalculate penalty from scratch every K steps?
                        # Or accept if equal (random walk)?
                        pass

            return None

        # This is getting too complex for a short script.
        # Let's use a simpler, robust method:
        # 1. Try to construct a graph on n=20 using a known property.
        # 2. If we can't find it fast, return a valid small graph.
        
        # Actually, the prompt says "Graphs with 20 vertices are known".
        # We can try to generate one.
        
        # Let's try a simple local search on n=20 with a time limit.
        # If it finds one, return it.
        # If not, try n=19.
        
        pass # Placeholder for actual implementation below

    # Fallback: Return a valid graph on n=5 (the seed)
    # But we want to do better.
    # Let's implement a simple hill climbing for n=20.
    
    # For the sake of the solution, I will implement a direct search for n=20.
    # If it fails, I will return n=5.
    
    # Actually, I can just return a hardcoded 20-vertex graph if I find one?
    # I don't have one hardcoded.
    
    # Let's try to run the local search in the main function.
    
    # Due to the complexity of the incremental update, I will use a 
    # "batch" approach: 
    # 1. Random graph.
    # 2. Check all size-6 subsets.
    # 3. If any regular, pick one, and try to break it by flipping an edge inside it.
    # 4. Repeat.
    
    def try_find(n):
        subsets_6 = []
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_6.append(mask)
        
        # Precompute vertices for each mask for faster degree check?
        # is_regular_subset is the bottleneck.
        
        for restart in range(10):
            if time.time() > deadline - 2:
                return None
                
            A = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        A[u] |= (1 << v)
                        A[v] |= (1 << u)
            
            # Find a violating subset
            violating_mask = None
            for mask in subsets_6:
                if is_regular_subset(A, mask):
                    violating_mask = mask
                    break
            
            if violating_mask is None:
                # Check larger sizes
                valid = True
                for k in range(7, n + 1):
                    for comb in itertools.combinations(range(n), k):
                        mask = 0
                        for v in comb:
                            mask |= (1 << v)
                        if is_regular_subset(A, mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return A
            
            # If we found a violation, try to fix it
            # Pick a random edge within the violating mask and flip it
            verts = [v for v in range(n) if violating_mask & (1 << v)]
            for _ in range(20): # Try 20 random flips
                if time.time() > deadline - 2:
                    return None
                u, v = random.sample(verts, 2)
                A[u] ^= (1 << v)
                A[v] ^= (1 << u)
                
                # Check if this fixed the specific violation
                if not is_regular_subset(A, violating_mask):
                    # Good, accept this step and restart the search for violations
                    break
            
    # Run try_find for n=21, 20, 19
    for n in [21, 20, 19]:
        if time.time() > deadline - 5:
            break
        res = try_find(n)
        if res:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (res[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback
    return {'n': 5, 'edges': []}
