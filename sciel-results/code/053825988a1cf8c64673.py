import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # The optimal solution is a subset of {51, ..., 100}
    # Any element < 51 creates tighter constraints on the larger elements
    # without adding enough "value" (cardinality) to compensate, 
    # because the target sum 200 is small relative to the range.
    
    candidates = list(range(51, 101))
    
    # 1. Generate all "bad" subsets (triples) that sum to 200
    # Since min element is 51, max size of a subset summing to 200 is 3.
    # (51+52+53 = 156, 51+52+53+54 > 200? No, 51+52+53+54 = 210 > 200. 
    #  Wait, 51+52+53 = 156. 51+52+53+54 is not a sum, it's a subset.
    # Can 4 elements sum to 200?
    # Min sum of 4 distinct elements from {51..100} is 51+52+53+54 = 210.
    # So NO subset of size >= 4 can sum to 200.
    # Max size is 3.
    # Can size 2 sum to 200?
    # Max sum of 2 is 99+100=199.
    # So NO subset of size 2 can sum to 200.
    # Thus, only triples are the constraints.
    
    bad_triples = []
    n = len(candidates)
    # Map value to index for fast lookup if needed, but values are unique 51..100
    # We can just work with values directly.
    
    for i in range(n):
        a = candidates[i]
        for j in range(i+1, n):
            b = candidates[j]
            s2 = a + b
            if s2 >= 200:
                break # c would be <= 0 or smaller than b, but we need c > b
            c_needed = 200 - s2
            # c must be > b and <= 100
            if c_needed > b and c_needed <= 100:
                # Check if c is in candidates (it is, since 51..100)
                # Ensure c is distinct from a, b (guaranteed by c > b > a)
                bad_triples.append((a, b, c_needed))
                
    # 2. Greedy Vertex Cover to minimize removed elements
    # We want to remove a set R such that every triple in bad_triples has at least one element in R.
    # This is Minimum Vertex Cover in a 3-uniform hypergraph.
    # Greedy: Pick vertex in most uncovered edges.
    
    # For each vertex, count how many triples it participates in
    # And keep track of which triples are "covered" (broken)
    
    # State:
    # selected: set of vertices to KEEP
    # removed: set of vertices to REMOVE
    # We start with all selected.
    selected = set(candidates)
    removed = set()
    
    # Precompute incidence: vertex -> list of triple indices
    vertex_triples = {v: [] for v in candidates}
    for idx, t in enumerate(bad_triples):
        for v in t:
            vertex_triples[v].append(idx)
            
    # active_triples: boolean array, True if not yet covered
    active = [True] * len(bad_triples)
    
    def is_valid(selected_set):
        # Check if any active triple is fully contained in selected_set
        # We can just check all triples
        for a, b, c in bad_triples:
            if a in selected_set and b in selected_set and c in selected_set:
                return False
        return True

    # Greedy Removal
    # We remove vertices until no active triple is fully in selected.
    # Actually, simpler:
    # While there exists a triple fully in selected:
    #   Pick the vertex in that triple that has the highest degree (or just any)
    #   Remove it.
    
    # To make it robust, we use a priority queue or just scan.
    # Given small size, scanning is fine.
    
    # We will run a local search loop.
    # 1. Greedy pass to get a valid set.
    # 2. Local search to add back removed vertices if possible.
    
    # Greedy Pass
    current_selected = set(candidates)
    current_removed = set()
    
    while True:
        # Find a conflict
        conflict = None
        for a, b, c in bad_triples:
            if a in current_selected and b in current_selected and c in current_selected:
                conflict = (a, b, c)
                break
        
        if conflict is None:
            break
            
        # Resolve conflict by removing one element
        # Heuristic: Remove the element that appears in the most OTHER conflicts?
        # Or just the one with the highest global degree.
        # Let's use global degree as a static heuristic, or dynamic count of active conflicts.
        
        # Dynamic count: how many active triples does each vertex in conflict belong to?
        # Since we don't maintain 'active' strictly in this simple loop, 
        # we can just count how many bad_triples contain each vertex that are still "valid" (fully in current_selected).
        
        counts = {v: 0 for v in conflict}
        for a, b, c in bad_triples:
            if a in current_selected and b in current_selected and c in current_selected:
                for v in conflict:
                    if v in (a, b, c):
                        counts[v] += 1
        
        # Remove the one with highest count
        v_to_remove = max(counts, key=counts.get)
        current_selected.remove(v_to_remove)
        current_removed.add(v_to_remove)
        
        if time.time() > deadline - 5:
            break

    # 3. Local Search / Shake
    # Try to add back removed vertices
    # Order of removal might matter. We can try multiple random orders or restarts.
    
    best_size = len(current_selected)
    best_set = set(current_selected)
    
    # Perform a few random restarts of the greedy process
    # Or just a simple hill climbing on the current solution
    
    # Hill Climbing:
    # Try to add each removed vertex back. If it causes a conflict, try to remove one of the conflicting vertices instead?
    # This is getting complex.
    
    # Simpler: Run the greedy process with different tie-breaking strategies.
    
    def run_greedy(seed):
        random.seed(seed)
        sel = set(candidates)
        rem = set()
        
        # Shuffle candidates for tie-breaking? No, the conflict selection is deterministic based on order.
        # We can shuffle the list of bad_triples to change which conflict is found first.
        
        triples_copy = bad_triples[:]
        random.shuffle(triples_copy)
        
        while True:
            conflict = None
            for a, b, c in triples_copy:
                if a in sel and b in sel and c in sel:
                    conflict = (a, b, c)
                    break
            
            if conflict is None:
                break
                
            # Pick one of the 3 vertices to remove.
            # Randomly pick one? Or pick the one with highest degree?
            # Let's try random picking to explore different covers.
            v_to_remove = random.choice(conflict)
            sel.remove(v_to_remove)
            rem.add(v_to_remove)
            
            if time.time() > deadline - 2:
                break
                
        # Try to add back
        for v in list(rem):
            # Check if adding v back creates a conflict
            sel.add(v)
            if is_valid(sel):
                rem.remove(v)
            else:
                sel.remove(v)
                
        return sel

    # Run a few iterations
    best_set = current_selected
    best_size = len(best_set)
    
    for i in range(100):
        if time.time() > deadline - 2:
            break
        s = run_greedy(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return list(best_set)
# EVOLVE-BLOCK-END
