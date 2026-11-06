import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    def is_valid(n, adj_masks):
        # adj_masks: list of n ints
        # Check subsets of size k >= 6
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                degs = []
                for v in subset:
                    deg = (adj_masks[v] & mask).bit_count()
                    degs.append(deg)
                
                # Check if all degrees are equal
                if all(d == degs[0] for d in degs):
                    return False
        return True

    def generate_random_graph(n, p):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try multiple random instances
        # More attempts for smaller n, fewer for larger n if time is tight
        # But let's just loop until time runs out for this n
        while time.time() - start_time < time_limit - 5:
            adj_masks = generate_random_graph(n, 0.5)
            if is_valid(n, adj_masks):
                best_n = n
                best_edges = get_edges_from_masks(n, adj_masks)
                # If we found a large n, we can stop early?
                # The task says "Maximize n".
                # If we found n=20, we should return it immediately?
                # Yes, 20 is the max known. 21 is possible but unlikely to find randomly.
                # If we find 20, we return it.
                if n == 20:
                    return {'n': n, 'edges': best_edges}
                # If we find 19, we might still want to try 20?
                # But we are iterating downwards.
                # So if we are at n=19, we already failed 20.
                # So we should return the best found so far if we find one?
                # No, we want the MAXIMUM.
                # Since we go 20, 19, 18..., if we find one at 19, it's the best we have.
                # But maybe we can find a 20 later? No, we passed 20.
                # So we should break out of the n-loop if we find a valid graph?
                # Yes, because we are going downwards.
                break 
        
        # If we didn't find a valid graph for this n, continue to n-1
        # If we did, best_n is updated, and we break the inner while.
        # Then we break the outer for loop?
        # Yes, because any subsequent n is smaller.
        if best_n == n:
            break

    # Fallback: if no graph found for n>=6, return a trivial one
    if best_n < 6:
        # Construct a valid graph on 5 vertices (any graph works)
        # Empty graph on 5 vertices
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
