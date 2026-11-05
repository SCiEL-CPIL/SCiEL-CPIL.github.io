import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy: Randomized local search.
    1. Generate random 5-uniform hypergraphs on n=9 or n=10 vertices.
    2. Check for non-2-colorability using bitmask enumeration (fast for n<=20).
    3. If non-2-colorable, greedily prune edges.
    4. Keep the best (minimum edge count) result found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_edges = None
    best_n = 9
    
    # Helper: Fast check for non-2-colorability using bitmasks
    # Only efficient for small n (<= 20 or so)
    def is_non_2colorable_fast(edges, n):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        i.e., For every mask (subset of red vertices), there is an edge e 
        such that |e & mask| == 0 or |e & mask| == 5.
        """
        # Precompute edge masks
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        
        # Iterate over all possible colorings (masks)
        # mask bit i is 1 if vertex i is red
        for mask in range(1 << n):
            valid_coloring = True
            for em in edge_masks:
                # Count red vertices in this edge
                # bin(x).count('1') is fast for small x
                if bin(em & mask).count('1') in (0, 5):
                    valid_coloring = False
                    break
            if valid_coloring:
                # Found a coloring where all edges have both colors
                return False
        return True

    # Helper: Greedy pruning
    def prune_edges(edges, n, time_limit):
        """
        Try to remove edges while preserving non-2-colorability.
        Returns the pruned list of edges.
        """
        current_edges = list(edges)
        # Shuffle to avoid order bias
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            # Try removing current_edges[i]
            candidate = current_edges[:i] + current_edges[i+1:]
            
            # Check if candidate is still non-2-colorable
            if is_non_2colorable_fast(candidate, n):
                current_edges = candidate
                # Do not increment i, check the next edge in the new list at the same index
            else:
                i += 1
                
        return current_edges

    # Main search loop
    # We try n=9 and n=10. n=9 has fewer vertices, so 2^n checks are faster,
    # but fewer total possible edges (126). n=10 has 252 possible edges.
    # Known best is 51 edges.
    
    for n in [9, 10, 11]:
        if time.time() > deadline - 10.0:
            break
            
        # Number of all possible edges
        total_edges = len(list(combinations(range(n), 5)))
        
        # Try several random instances
        # We start with a reasonable number of edges, say 50-60
        # If we find a non-2-colorable one, we prune it.
        
        for attempt in range(1000): # Max attempts per n, but limited by time
            if time.time() > deadline - 5.0:
                break
                
            # Choose a target number of edges for initialization
            # We want to find small hypergraphs. 
            # If we start too small, it's likely colorable.
            # If we start too large, it's likely non-2-colorable but pruning takes time.
            # Let's start around 50-55.
            k_init = random.randint(45, 60)
            
            # Generate random distinct edges
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            initial_edges = [list(e) for e in all_possible[:k_init]]
            
            # Check if initial is non-2-colorable
            if is_non_2colorable_fast(initial_edges, n):
                # It is non-2-colorable. Try to prune.
                # Set a sub-deadline for pruning to allow restarts
                prune_deadline = time.time() + 2.0
                if time.time() + 2.0 > deadline:
                    prune_deadline = deadline - 1.0
                    
                pruned = prune_edges(initial_edges, n, prune_deadline)
                
                if len(pruned) < (best_edges is None and 999 or len(best_edges) if best_edges else 999):
                    best_edges = pruned
                    best_n = n
                    
                    # If we found a very good solution, we can keep searching for better
                    # But if we hit the known lower bound or close to it, it's great.
                    # The known lower bound is 29.
                    if len(pruned) <= 30:
                        # Unlikely, but if we find it, we can stop early or keep trying
                        pass
                        
    # If no solution found (should not happen as random dense graphs are non-2-colorable),
    # fall back to a known construction or the complete graph on 9 vertices.
    if best_edges is None:
        # Fallback: Complete 5-graph on 9 vertices
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
