import time
import random
import itertools
import sys

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6+ vertices.
    Maximizes n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # We aim for n=21 first, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline:
            break
        
        # Precompute subsets for this n
        # We will focus on size 6 for incremental updates
        # and check size 7+ when size 6 is clean.
        
        subsets_6 = list(itertools.combinations(range(target_n), 6))
        subsets_7 = list(itertools.combinations(range(target_n), 7))
        
        # Map each subset to an index
        # For incremental update, we need to know which subsets contain a vertex.
        # subsets_with[i] = list of indices of subsets containing i
        subsets_with = [[] for _ in range(target_n)]
        
        # Store subset vertices for quick access
        # And for each subset, store the index of each vertex in the tuple?
        # Or just the tuple itself.
        
        # To speed up degree updates, we can store for each subset:
        # - The tuple of vertices
        # - The current degrees of these vertices within the subset
        # - A flag if it's regular
        
        # This structure is heavy. Let's try a simpler approach first:
        # Fast full check for size 6 using bit_count.
        
        # If n=21, binom(21,6) = 54264.
        # binom(21,7) = 116280.
        # Total ~170k subsets.
        # If we can check 1000 graphs/sec, we can do 100k checks.
        
        # Let's try a highly optimized full check.
        
        def count_violations_fast(adj):
            """
            Returns number of regular induced subgraphs of size 6 and 7.
            """
            count = 0
            # Size 6
            for subset in subsets_6:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    count += 1
            # Size 7
            if count > 0: # Optimization: if many 6s, skip 7s? No, we need to minimize total.
                # But 7s are more expensive.
                pass
            for subset in subsets_7:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    count += 1
            return count

        # Local Search with Fast Check
        best_adj = None
        best_violations = float('inf')
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random init
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            curr_violations = count_violations_fast(adj)
            if curr_violations == 0:
                # Verify full validity (sizes 8+)
                if is_fully_valid(adj, target_n):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
            
            if curr_violations < best_violations:
                best_violations = curr_violations
                best_adj = adj[:]
            
            # Hill Climbing
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Pick random edge to flip
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                
                # Flip
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_violations = count_violations_fast(adj)
                
                if new_violations == 0:
                    if is_fully_valid(adj, target_n):
                        edges = []
                        for k in range(target_n):
                            for l in range(k + 1, target_n):
                                if adj[k] & (1 << l):
                                    edges.append([k, l])
                        return {'n': target_n, 'edges': edges}
                
                if new_violations <= curr_violations:
                    curr_violations = new_violations
                    if curr_violations < best_violations:
                        best_violations = curr_violations
                        best_adj = adj[:]
                else:
                    # Revert
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

    # Fallback to best found or n=5
    if best_adj is not None:
        n = target_n # Note: target_n is the last one tried. We should track best_n.
        # We need to track which n gave the best_adj.
        # Let's restructure to return the first valid one found.
        # If no valid one found, we return a known valid small graph.
        # For now, return n=5 empty graph.
        pass
    
    return {'n': 5, 'edges': []}

def is_fully_valid(adj, n):
    """
    Checks if graph has no regular induced subgraph of size >= 6.
    """
    # Check 6 and 7 are done in fast check. 
    # Here we check 8..n
    for size in range(8, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
            if len(set(degs)) == 1:
                return False
    return True
