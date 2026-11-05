import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy:
      1. Use a fast bitmask-based checker for n <= 16 (brute force 2^n).
      2. Multi-start random search over n=13, 14, 15.
      3. Aggressive greedy pruning.
      4. Safety: always return a valid non-2-colorable hypergraph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_edges = None
    best_n = 14

    def is_non_2colorable_fast(edges, n):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        Uses brute force over all 2^n masks. Efficient for n <= 16.
        """
        if not edges:
            return False

        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        
        # Iterate over all possible colorings
        for mask in range(1 << n):
            valid_coloring = True
            for em in edge_masks:
                # If edge is entirely red (em & mask == em) or entirely blue (em & mask == 0),
                # then this coloring is invalid for Property B (it has a monochromatic edge).
                # We want a coloring where NO edge is monochromatic.
                # So if we find a mask where ALL edges are non-mono, the graph IS 2-colorable.
                if (em & mask) == em or (em & mask) == 0:
                    valid_coloring = False
                    break
            if valid_coloring:
                return False
        return True

    def prune_edges(edges, n, time_limit):
        """
        Greedy pruning with random restarts.
        """
        current_edges = list(edges)
        
        # Single pass
        current_edges = _prune_pass(current_edges, n, time_limit)
        
        # Random restarts
        max_restarts = 10
        for _ in range(max_restarts):
            if time.time() > time_limit:
                break
            if len(current_edges) <= 51: # Known lower bound area
                break
            
            shuffled = current_edges[:]
            random.shuffle(shuffled)
            
            new_edges = _prune_pass(shuffled, n, time_limit)
            
            if len(new_edges) < len(current_edges):
                current_edges = new_edges
            else:
                # Try a kick: remove a random edge, if fails, put back.
                # This helps escape local minima.
                if len(current_edges) > 1:
                    idx = random.randint(0, len(current_edges) - 1)
                    kick_edges = current_edges[:idx] + current_edges[idx+1:]
                    if is_non_2colorable_fast(kick_edges, n):
                        current_edges = kick_edges
                    else:
                        break # Stuck
                else:
                    break
                    
        return current_edges

    def _prune_pass(edges, n, time_limit):
        """
        Single pass of greedy pruning.
        """
        current_edges = list(edges)
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            candidate = current_edges[:i] + current_edges[i+1:]
            
            if is_non_2colorable_fast(candidate, n):
                current_edges = candidate
            else:
                i += 1
        return current_edges

    # Pre-compute all possible edges for small n
    all_possible_edges = {}
    for n in [13, 14, 15]:
        all_possible_edges[n] = list(combinations(range(n), 5))
        
    # Search loop
    for n in [13, 14, 15]:
        if time.time() > deadline - 10.0:
            break
            
        total_edges = len(all_possible_edges[n])
        attempts = 500 if n == 13 else 200
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Start with a dense random graph
            k_init = random.randint(60, 80)
            
            all_possible = all_possible_edges[n][:]
            random.shuffle(all_possible)
            initial_edges = [list(e) for e in all_possible[:k_init]]
            
            if is_non_2colorable_fast(initial_edges, n):
                prune_deadline = time.time() + 2.0
                if time.time() + 2.0 > deadline:
                    prune_deadline = deadline - 1.0
                    
                pruned = prune_edges(initial_edges, n, prune_deadline)
                
                if best_edges is None or len(pruned) < len(best_edges):
                    best_edges = pruned
                    best_n = n
                    
                    if len(pruned) <= 51:
                        # We found the known best or better. 
                        # We can stop early if we hit 51 to save time, 
                        # but we can also try to see if 50 is possible.
                        # Given time constraints, 51 is a very good result.
                        # Let's continue searching a bit more to see if we can go lower.
                        pass

    # Fallback: Complete 5-graph on 9 vertices (126 edges)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
