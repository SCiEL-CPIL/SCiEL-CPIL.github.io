import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    Uses a highly optimized backtracking 2-colorability checker and greedy edge removal
    starting from the complete 5-uniform hypergraph on 9 vertices (K9^(5)).
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    def is_2_colorable(n, edges):
        """
        Check if hypergraph is 2-colorable using backtracking with unit propagation.
        Optimized with bitmasks for speed.
        """
        if not edges:
            return True
        
        # Preprocess: distinct edges, sorted
        edges = list(set(tuple(sorted(e)) for e in edges))
        if not edges:
            return True
            
        n_edges = len(edges)
        
        # Map vertex to bit position for fast edge membership check?
        # Or precompute edge masks.
        # n <= 26, so we can use Python integers as bitsets.
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # Vertex degrees for ordering
        deg = [0] * n
        for e in edges:
            for v in e:
                deg[v] += 1
        
        vertex_order = sorted(range(n), key=lambda v: -deg[v])
        
        # State: color[v] = -1, 0, 1
        color = [-1] * n
        
        # For each edge, we need to know counts of 0, 1, -1
        # We will recompute counts on the fly in the propagation step for simplicity,
        # or maintain them. Maintaining is faster.
        
        # Let's use a recursive solver with bitset checks.
        
        def propagate(edge_indices):
            """
            Unit propagation.
            Returns False if contradiction, True otherwise.
            Modifies color array.
            """
            changed = True
            while changed:
                changed = False
                for i in edge_indices:
                    e = edges[i]
                    c0 = 0
                    c1 = 0
                    unc = 0
                    last_unc_v = -1
                    for v in e:
                        cv = color[v]
                        if cv == 0:
                            c0 += 1
                        elif cv == 1:
                            c1 += 1
                        else:
                            unc += 1
                            last_unc_v = v
                            
                    if unc == 0:
                        if c0 == 0 or c1 == 0:
                            return False
                    elif unc == 1:
                        if c0 > 0 and c1 == 0:
                            # Must be 1
                            if color[last_unc_v] != -1: # Should be -1
                                if color[last_unc_v] == 0: return False
                            color[last_unc_v] = 1
                            changed = True
                        elif c1 > 0 and c0 == 0:
                            # Must be 0
                            if color[last_unc_v] != -1:
                                if color[last_unc_v] == 1: return False
                            color[last_unc_v] = 0
                            changed = True
            return True

        def backtrack(pos):
            if pos == n:
                # All assigned. Verify all edges are bichromatic.
                for i in range(n_edges):
                    c0 = 0
                    c1 = 0
                    for v in edges[i]:
                        if color[v] == 0: c0 += 1
                        elif color[v] == 1: c1 += 1
                    if c0 == 0 or c1 == 0:
                        return False
                return True
                
            v = vertex_order[pos]
            if color[v] != -1:
                return backtrack(pos + 1)
            
            # Try 0
            color[v] = 0
            if propagate(range(n_edges)) and backtrack(pos + 1):
                return True
            color[v] = -1
            
            # Try 1
            color[v] = 1
            if propagate(range(n_edges)) and backtrack(pos + 1):
                return True
            color[v] = -1
            
            return False
            
        return backtrack(0)

    def minimize_edges(n, edges, rng, deadline):
        """
        Greedy removal of edges that are redundant.
        """
        current_edges = edges[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            rng.shuffle(current_edges)
            for i in range(len(current_edges)):
                if time.time() > deadline:
                    break
                trial_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, trial_edges):
                    current_edges = trial_edges
                    changed = True
                    break
        return current_edges

    # Strategy: Start with K9^(5) which is definitely not 2-colorable.
    # K9^(5) has 126 edges. We need to get to 51.
    # This requires removing 75 edges.
    # Each removal check is expensive.
    # We need a faster checker or a smarter removal.
    
    # Let's try to find a smaller n first? 
    # m(5) = 51. The minimal n is likely 10 or less.
    # K9^(5) is a good starting point because it's dense and symmetric.
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = all_edges[:]
    
    # Verify it's not 2-colorable (should be true)
    if is_2_colorable(n, current_edges):
        # If K9 is 2-colorable (it shouldn't be), we have a problem.
        # But K9^(5) contains K7^(5)? No.
        # It is known that K9^(5) is not 2-colorable.
        pass
        
    rng = random.Random(42)
    
    # Aggressive minimization
    min_edges = minimize_edges(n, current_edges, rng, deadline)
    
    # If we didn't reach 51, try other n values or different seeds?
    # The greedy removal on K9^(5) is known to get close to 51.
    # If time remains, we can try to restart with different shuffles.
    
    best_edges = min_edges
    best_n = n
    
    # If we have time, try to improve further or verify 51
    if len(best_edges) > 51 and time.time() < deadline - 10:
        # Try a few more random restarts of the greedy process
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            trial_edges = all_edges[:]
            rng2 = random.Random(rng.randint(0, 10000))
            min_edges2 = minimize_edges(n, trial_edges, rng2, deadline - 5)
            if len(min_edges2) < len(best_edges):
                best_edges = min_edges2
                best_n = n
                
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}

# EVOLVE-BLOCK-END
