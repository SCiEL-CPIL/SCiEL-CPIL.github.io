"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges, deadline=None):
    """
    Optimized 2-colorability check using bitsets and unit propagation.
    Returns True if 2-colorable, False if not, None if timeout.
    """
    if not edges:
        return True
        
    m = len(edges)
    # Convert edges to bitmasks
    masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        masks.append(mask)
        
    # Precompute vertex-edge incidence for variable selection
    # Since n <= 26, we can use a list of lists
    vedges = [[] for _ in range(n)]
    for i, mask in enumerate(masks):
        v = 0
        temp = mask
        while temp:
            low = temp & (-temp)
            v = low.bit_length() - 1
            vedges[v].append(i)
            temp ^= low
            v += 1
            
    # Precompute popcount for small numbers? Or just use bit_count()
    # Python 3.8+ has int.bit_count()
    
    def popcount(x):
        return x.bit_count()

    # State: c0 (vertices colored 0), c1 (vertices colored 1)
    # We use a recursive solver with unit propagation.
    
    def solve(c0, c1, depth=0):
        if deadline is not None and time.time() > deadline:
            return None # Timeout
        
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                mask = masks[i]
                c0e = c0 & mask
                c1e = c1 & mask
                
                # Check if edge is monochromatic
                if (c0e == mask) or (c1e == mask):
                    return False
                
                # Check if edge has 4 of one color and 1 unassigned
                pop0 = popcount(c0e)
                pop1 = popcount(c1e)
                
                if pop0 == 4:
                    # The remaining vertex must be 1
                    unassigned = mask & ~(c0 | c1)
                    if unassigned:
                        # If unassigned is already 0, conflict
                        if unassigned & c0:
                            return False
                        # If unassigned is already 1, no change needed (but pop1 would be 1? No, pop1 counts assigned 1s)
                        # Wait, if pop0=4, there is 1 vertex left.
                        # If that vertex is assigned 1, pop1 would be 1.
                        # If that vertex is unassigned, we assign it 1.
                        if unassigned & c1:
                            pass # Already assigned 1, consistent
                        else:
                            c1 |= unassigned
                            changed = True
                            
                elif pop1 == 4:
                    unassigned = mask & ~(c0 | c1)
                    if unassigned:
                        if unassigned & c1:
                            return False
                        if unassigned & c0:
                            pass
                        else:
                            c0 |= unassigned
                            changed = True

        # Check if all vertices assigned
        if (c0 | c1) == (1 << n) - 1:
            return True

        # Variable Selection: Choose unassigned vertex with highest degree in remaining graph
        unassigned = (1 << n) - 1
        unassigned &= ~(c0 | c1)
        
        best_v = -1
        best_score = -1
        
        # To speed up, we can just iterate through unassigned vertices
        # n is small (<= 26)
        for v in range(n):
            if unassigned & (1 << v):
                # Count how many edges containing v are not yet satisfied/conflicted
                # Actually, just counting edges containing v that have other unassigned vertices
                score = 0
                for i in vedges[v]:
                    # If the edge has any other unassigned vertex, it's "active"
                    if masks[i] & unassigned:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_v = v
                    if best_score == m: # Max possible
                        break

        if best_v == -1:
            # All assigned? Should have been caught above
            return True

        bit = 1 << best_v
        
        # Try 0 first
        res = solve(c0 | bit, c1, depth + 1)
        if res is not None:
            return res
            
        # Try 1
        res = solve(c0, c1 | bit, depth + 1)
        if res is not None:
            return res
            
        return False

    return solve(0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # Known 51-edge construction
    known_51 = [
        [0,1,2,3,4],[0,1,2,3,5],[0,1,2,4,5],[0,1,3,4,5],[0,2,3,4,5],[1,2,3,4,5],
        [0,1,2,3,6],[0,1,2,3,7],[0,1,2,4,6],[0,1,2,4,7],[0,1,3,4,6],[0,1,3,4,7],
        [0,2,3,4,6],[0,2,3,4,7],[0,2,4,5,6],[0,2,4,5,7],[0,3,4,5,6],[0,3,4,5,7],
        [1,2,3,4,6],[1,2,3,4,7],[1,2,4,5,6],[1,2,4,5,7],[1,3,4,5,6],[1,3,4,5,7],
        [2,3,4,5,6],[2,3,4,5,7],[3,4,5,6,7],
        [0,1,2,3,8],[0,1,2,4,8],[0,1,3,5,8],[0,2,3,5,8],[0,2,4,5,8],
        [1,2,3,4,8],[1,2,3,5,8],[1,2,4,5,8],[1,3,4,5,8],[2,3,4,5,8],
        [0,1,2,3,9],[0,1,2,4,9],[0,1,3,5,9],[0,2,3,5,9],[0,2,4,5,9],
        [1,2,3,4,9],[1,2,3,5,9],[1,2,4,5,9],[1,3,4,5,9],[2,3,4,5,9],
        [0,1,2,3,10],[0,1,2,4,10],[0,1,3,5,10],[0,2,3,5,10],[0,2,4,5,10],
        [1,2,3,4,10],[1,2,3,5,10],[1,2,4,5,10],[1,3,4,5,10],[2,3,4,5,10],
    ]
    
    # Verify known 51-edge construction
    is_51_valid = _is_2_colorable_fast(11, known_51, deadline)
    if is_51_valid is False:
        best_edges = [list(e) for e in known_51]
        best_n = 11
        best_count = 51
    else:
        best_edges = None
        best_n = None
        best_count = 200 # Initialize high

    # Randomized Local Search to try to find a smaller non-2-colorable hypergraph
    # Strategy:
    # 1. Start with a random set of edges (or the known one)
    # 2. Try to remove edges one by one if non-2-colorability is preserved
    # 3. Try to add edges if it helps in a different context? No, we want to minimize.
    #    So we start with a superset and prune.
    #    Or start with a random non-2-colorable graph and prune.
    
    # Since we don't have a random non-2-colorable generator, we start with the known 51-edge one.
    # But we can also try random subsets of the complete hypergraph on small n.
    
    # Let's try to improve the 51-edge solution by aggressive pruning.
    # The current program does a simple greedy pass. We can do a more thorough local search.
    # Randomly shuffle edges and try to remove them.
    
    if best_edges is not None and time.time() < deadline:
        # Make a copy
        current_edges = [list(e) for e in best_edges]
        current_n = best_n
        
        # Shuffle to break symmetry in pruning order
        random.seed(42) # Deterministic for reproducibility in tests, but we can vary
        random.shuffle(current_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_edges)):
                if time.time() > deadline:
                    break
                # Try removing edge i
                candidate_edges = current_edges[:i] + current_edges[i+1:]
                if not candidate_edges:
                    break # Empty is 2-colorable
                res = _is_2_colorable_fast(current_n, candidate_edges, deadline)
                if res is False:
                    # Successfully removed
                    current_edges = candidate_edges
                    improved = True
                    # Restart the loop to check from beginning again, as removal might enable further removals
                    break
        
        if len(current_edges) < best_count:
            # Final verification
            res = _is_2_colorable_fast(current_n, current_edges, deadline)
            if res is False:
                best_edges = current_edges
                best_count = len(current_edges)

    # Try other small n values?
    # If we have time, try to construct from scratch for n=8, 9, 10?
    # For n=8, max edges = 56. If we can find a non-2-colorable one with < 51 edges, that's better.
    # But 2-colorability is hard to break for small n.
    # Known lower bound is 29.
    
    # Let's try a random search for n=9, 10, 11
    # Start with random edges, check if non-2-colorable.
    # This is unlikely to hit quickly.
    
    # Alternative: Use the known 51-edge graph but permute vertices to see if a different pruning order helps?
    # The graph structure is fixed. Permutation doesn't change the property.
    
    # What if we try to find a 50-edge graph?
    # We can try to add a vertex and edges to the 51-edge graph? No, we want to minimize edges.
    
    # Let's try a different random strategy:
    # Pick a random subset of the 51 edges. If it's non-2-colorable, prune it.
    # This might find a different minimal core.
    
    if time.time() < deadline:
        for trial in range(10):
            if time.time() > deadline:
                break
            # Random subset of size 45-51
            size = random.randint(45, 51)
            subset = random.sample(best_edges, size)
            res = _is_2_colorable_fast(best_n, subset, deadline)
            if res is False:
                # Prune this subset
                current_edges = [list(e) for e in subset]
                random.shuffle(current_edges)
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(current_edges)):
                        if time.time() > deadline:
                            break
                        candidate_edges = current_edges[:i] + current_edges[i+1:]
                        if not candidate_edges:
                            break
                        res = _is_2_colorable_fast(best_n, candidate_edges, deadline)
                        if res is False:
                            current_edges = candidate_edges
                            improved = True
                            break
                if len(current_edges) < best_count:
                    res = _is_2_colorable_fast(best_n, current_edges, deadline)
                    if res is False:
                        best_edges = current_edges
                        best_count = len(current_edges)

    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Complete hypergraph on 9 vertices (126 edges)
    n9 = 9
    return {'n': n9, 'edges': [list(e) for e in combinations(range(n9), 5)]}

# EVOLVE-BLOCK-END
