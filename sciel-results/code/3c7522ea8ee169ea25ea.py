import time
import math

def construct():
    """
    Finds a nonempty subset of [2, 256] with lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    """
    N = 256
    nodes = list(range(2, N + 1))
    weights = {x: 1.0 / x for x in nodes}
    
    # Precompute neighbors
    # neighbors[x] is the set of y in nodes such that lcm(x, y) <= 256
    neighbors = {x: set() for x in nodes}
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if math.lcm(a, b) <= N:
                neighbors[a].add(b)
                neighbors[b].add(a)
                
    # Sort nodes by weight descending (i.e., value ascending)
    # This helps in finding good solutions early
    sorted_nodes = sorted(nodes, key=lambda x: -weights[x])
    
    global_best = 0.0
    best_set = []
    
    start_time = time.time()
    # Use 80s deadline to be safe within 90s limit, allowing for overhead
    deadline = start_time + 80.0 
    
    def greedy_bound(candidates):
        """
        Calculates a tight upper bound on the max weight independent set
        of the subgraph induced by candidates.
        Greedily picks the max weight node, adds its weight, and removes
        it and its neighbors.
        """
        if not candidates:
            return 0.0
        bound = 0.0
        remaining = set(candidates)
        while remaining:
            # Find node with max weight in remaining
            v = max(remaining, key=lambda x: weights[x])
            bound += weights[v]
            # Remove v and its neighbors from consideration
            remaining -= neighbors[v]
            remaining.discard(v)
        return bound

    def dfs(candidates, current_sum, current_set):
        nonlocal global_best, best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > global_best:
                global_best = current_sum
                best_set = current_set[:]
            return
        
        # Prune if upper bound is not better than current best
        # We use greedy_bound which is tighter than sum(weights)
        upper_bound = greedy_bound(candidates)
        if current_sum + upper_bound <= global_best:
            return
        
        # Pick the node with the highest weight from candidates
        # Since we want to explore high weight nodes first
        v = max(candidates, key=lambda x: weights[x])
        
        # Branch 1: Include v
        # New candidates are those not v and not neighbors of v
        new_candidates = [x for x in candidates if x != v and x not in neighbors[v]]
        dfs(new_candidates, current_sum + weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        new_candidates = [x for x in candidates if x != v]
        dfs(new_candidates, current_sum, current_set)

    dfs(sorted_nodes, 0.0, [])
    
    # If no set was found (should not happen as single elements are valid), 
    # return the single element with max weight (2)
    if not best_set:
        return [2]
        
    return best_set
