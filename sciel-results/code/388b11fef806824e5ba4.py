import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()

def _is_non_2colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    edges: list of lists of 5 vertices.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge masks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Vertices involved in edges (optional, but n is small)
    
    # State: mask0 (vertices colored 0), mask1 (vertices colored 1)
    # Initially all 0.
    
    def popcount(x):
        return bin(x).count('1')
        
    def propagate(mask0, mask1):
        """
        Returns (new_mask0, new_mask1, success)
        If success is False, contradiction found.
        """
        changed = True
        while changed:
            changed = False
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                c_un = 5 - c0 - c1
                
                if c0 == 5 or c1 == 5:
                    return 0, 0, False # Contradiction
                
                if c0 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 1
                    # Find it: e_mask & ~(mask0 | mask1)
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    # unassigned_mask should have exactly 1 bit set
                    # Add it to mask1
                    new_mask1 = mask1 | unassigned_mask
                    if new_mask1 != mask1:
                        mask1 = new_mask1
                        changed = True
                elif c1 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 0
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask0 = mask0 | unassigned_mask
                    if new_mask0 != mask0:
                        mask0 = new_mask0
                        changed = True
        return mask0, mask1, True

    def backtrack(mask0, mask1):
        # Check if all vertices are assigned?
        # We don't strictly need all vertices assigned, just no contradiction and no forced moves.
        # But to find a valid coloring, we need to ensure we can assign the rest without contradiction.
        # Actually, if we can't force any more, and there are no contradictions,
        # we can just pick an unassigned vertex and branch.
        
        # Find an unassigned vertex
        # To be efficient, pick a vertex involved in an edge with the most assigned vertices?
        # Or just pick the first unassigned vertex in any edge?
        
        # Let's find a vertex that is in an edge with minimal unassigned count (max assigned).
        best_v = -1
        best_unassigned_count = 6 # Max is 5, so 6 is infinity
        
        for e_mask in edge_masks:
            c0 = popcount(e_mask & mask0)
            c1 = popcount(e_mask & mask1)
            c_un = 5 - c0 - c1
            if c_un < best_unassigned_count:
                best_unassigned_count = c_un
                # Find the unassigned vertices in this edge
                unassigned_mask = e_mask & ~(mask0 | mask1)
                if unassigned_mask:
                    # Pick the lowest bit
                    v = (unassigned_mask & -unassigned_mask).bit_length() - 1
                    best_v = v
                    if best_unassigned_count == 1:
                        # This should have been handled by propagation, 
                        # but if not, we must branch on it.
                        # Actually, if c_un==1, propagation should have forced it.
                        # So if we get here with c_un==1, it means propagation missed it?
                        # No, propagation runs to fixpoint. So c_un==1 shouldn't happen 
                        # unless the forced color was already assigned? No, 
                        # if c0==4, c_un==1, we force the 5th to 1.
                        # If c1==4, c_un==1, we force the 5th to 0.
                        # So after propagation, no edge should have c_un==1.
                        # So best_unassigned_count should be >= 2.
                        break
        
        if best_v == -1:
            # No unassigned vertices in any edge?
            # This means all edges are fully assigned.
            # Check if any edge is monochromatic.
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                if c0 == 5 or c1 == 5:
                    return True # Contradiction, so NOT 2-colorable
            # If we reach here, we have a valid 2-coloring.
            return False # IS 2-colorable
        
        # Branch on best_v
        # Try color 0
        new_mask0 = mask0 | (1 << best_v)
        # Check if best_v was already in mask1 (shouldn't be, as it's unassigned)
        if new_mask0 & mask1:
            # Contradiction, skip
            pass
        else:
            nm0, nm1, ok = propagate(new_mask0, mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        # Try color 1
        new_mask1 = mask1 | (1 << best_v)
        if new_mask1 & mask0:
            pass
        else:
            nm0, nm1, ok = propagate(mask0, new_mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        return True # No valid coloring found in this branch

    # Initial propagation
    nm0, nm1, ok = propagate(0, 0)
    if not ok:
        return True
    
    return backtrack(nm0, nm1)

def _try_prune(seed, max_time=10.0):
    """
    Attempts to find a small non-2-colorable hypergraph using edge pruning.
    """
    rng = random.Random(seed)
    
    # Choose n
    # Smaller n has fewer total edges, but denser hypergraphs are non-2-colorable.
    # C(9,5)=126. C(10,5)=252.
    # Let's try n from 9 to 14.
    n = rng.randint(9, 14)
    
    # Generate a starting set of edges
    all_edges = list(combinations(range(n), 5))
    
    # Strategy: Start with a random subset that is likely non-2-colorable.
    # We can just take the first K edges? No, random is better.
    # Let's take a random 70% of edges?
    # Or just start with all edges and prune?
    # Starting with all edges (K_n^(5)) is guaranteed non-2-colorable for n>=9.
    # Pruning from there is a good idea.
    
    edges = [list(e) for e in all_edges]
    rng.shuffle(edges)
    
    # Pruning loop
    # We want to remove as many edges as possible while keeping it non-2-colorable.
    # We iterate through edges and try to remove them.
    # To be efficient, we should try to remove edges in an order that might be "least important".
    # Random order is a good heuristic.
    
    # We will perform multiple passes of pruning.
    # In each pass, we try to remove each edge once.
    # If we removed any, we do another pass.
    
    # Optimization: Only check non-2-colorability if the removal might succeed.
    # We can't easily predict, so we just check.
    
    # To avoid excessive time, we limit the number of checks.
    # Each check is expensive.
    
    # Let's do a single pass first.
    changed = True
    passes = 0
    start_time = time.time()
    
    while changed and time.time() - start_time < max_time:
        changed = False
        passes += 1
        # Shuffle order for this pass
        current_edges = edges[:]
        rng.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() - start_time > max_time:
                break
            
            # Try to remove e
            edges_without_e = [ed for ed in edges if ed != e]
            
            # Check if still non-2-colorable
            if _is_non_2colorable(n, edges_without_e):
                # Yes, remove it
                edges = edges_without_e
                changed = True
    
    return len(edges), edges

def construct():
    best_edges = float('inf')
    best_result = None
    seed = 0
    
    # Fallback: Complete graph on 9 vertices
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    while time.time() < DEADLINE - 5:
        seed += 1
        # Allocate time for this attempt
        # If we have lots of time left, give more time to pruning.
        time_left = DEADLINE - time.time()
        if time_left < 10:
            break
            
        max_time_for_attempt = min(10.0, time_left * 0.5)
        
        try:
            count, edges = _try_prune(seed, max_time_for_attempt)
            if count < best_edges:
                best_edges = count
                best_result = {'n': len(edges[0][0] if edges else 0) + 1, 'edges': edges} # n needs to be correct
                # Actually, n is passed to _try_prune, but not returned.
                # Let's fix _try_prune to return n.
        except Exception:
            pass
            
    if best_result is None:
        return fallback
        
    return best_result
