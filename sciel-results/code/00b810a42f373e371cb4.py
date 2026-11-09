"""Sidon set with differences 1..30, minimize diameter."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    NEEDED = 31  # differences 1..30
    
    # We'll use backtracking search with iterative deepening on diameter
    # State: sorted list of elements, set of covered differences
    # Precompute collision info on the fly
    
    best_result = None
    best_diameter = float('inf')
    
    # First, try a fast greedy to get an initial upper bound
    def greedy_construct():
        a = [0]
        covered = set()
        # Precompute all differences
        def all_diffs(s):
            d = []
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    d.append(s[j] - s[i])
            return d
        
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            # Try adding x = v + d or x = v - d for each v in a
            candidates = []
            for v in a:
                for x in (v + d, v - d):
                    if x >= 0 and x not in a:
                        candidates.append(x)
            candidates = sorted(set(candidates))
            for x in candidates:
                trial = sorted(a + [x])
                diffs = all_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    covered = set(diffs)
                    found = True
                    break
            if not found:
                # Add two new elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    diffs = all_diffs(trial)
                    if len(set(diffs)) == len(diffs):
                        a = trial
                        covered = set(diffs)
                        break
                    y += 1
        return a
    
    greedy_sol = greedy_construct()
    best_diameter = max(greedy_sol)
    best_result = greedy_sol
    
    # Now do backtracking search with iterative deepening
    # We search for solutions with diameter <= bound
    # Use time-limited backtracking
    
    def backtracking_search(bound, time_limit):
        """Search for Sidon set with max element <= bound, covering diffs 1..30."""
        global best_result, best_diameter
        start = time.time()
        
        # State: elements (sorted), covered diffs set
        # We build the set incrementally
        # Key optimization: precompute which differences are "needed"
        
        elements = [0]
        covered = set()
        needed = set(range(1, 31))
        
        # For fast collision checking, maintain a set of all current differences
        current_diffs = set()
        
        # Precompute: for each possible new element x, which existing elements
        # would create a collision? We'll compute on the fly.
        
        # Move ordering: prioritize moves that cover the most needed differences
        # and keep the set "compact"
        
        def check_sidon(elements, new_x):
            """Check if adding new_x to elements maintains Sidon property."""
            for v in elements:
                d = abs(new_x - v)
                if d in current_diffs:
                    return False
            return True
        
        def get_new_diffs(elements, new_x):
            """Get new differences introduced by adding new_x."""
            return set(abs(new_x - v) for v in elements)
        
        def search(elements, covered, current_diffs, depth):
            nonlocal best_result, best_diameter
            
            if time.time() - start > time_limit:
                return False
            
            if not needed - covered:
                # Found a solution
                diameter = max(elements)
                if diameter < best_diameter:
                    best_diameter = diameter
                    best_result = list(elements)
                return True
            
            if len(elements) >= 32:
                return False
            
            if max(elements) > bound:
                return False
            
            # Generate candidate moves
            # Option 1: Add x = v + d or v - d for v in elements, d in needed
            # Option 2: Add two elements y, y+d for some d in needed
            # Option 3: Add a single new element that covers multiple needed diffs
            
            candidates = set()
            
            # For each needed difference d and each existing element v:
            for d in needed:
                for v in elements:
                    x1 = v + d
                    if 0 <= x1 <= bound and x1 not in elements:
                        candidates.add(x1)
                    x2 = v - d
                    if 0 <= x2 <= bound and x2 not in elements:
                        candidates.add(x2)
            
            # Also consider adding elements that might cover multiple needed diffs
            # This is harder to enumerate, so we'll rely on the above + backtracking
            
            # Score candidates by how many needed diffs they cover
            def score_candidate(x):
                if x in elements:
                    return -1
                if not check_sidon(elements, x):
                    return -1
                new_diffs = get_new_diffs(elements, x)
                return len(new_diffs & needed)
            
            scored = [(score_candidate(x), x) for x in candidates if score_candidate(x) >= 0]
            scored.sort(key=lambda t: (-t[0], abs(t[1] - max(elements))))  # Most coverage first, then closest to max
            
            for _, x in scored:
                if time.time() - start > time_limit:
                    return False
                
                if x in elements:
                    continue
                if not check_sidon(elements, x):
                    continue
                
                new_diffs = get_new_diffs(elements, x)
                new_covered = covered | new_diffs
                new_diffs_set = current_diffs | new_diffs
                new_elements = sorted(elements + [x])
                
                # Recurse
                if search(new_elements, new_covered, new_diffs_set, depth + 1):
                    return True
                
                # Undo
                # (we're using immutable sets, so no undo needed)
            
            # Try adding two elements at once: y and y+d for d in needed
            for d in needed:
                if time.time() - start > time_limit:
                    return False
                # Try y values
                for y in range(0, bound + 1, 1):
                    if time.time() - start > time_limit:
                        return False
                    y2 = y + d
                    if y2 > bound:
                        break
                    if y in elements or y2 in elements:
                        continue
                    # Check Sidon for both
                    if not check_sidon(elements, y):
                        continue
                    # Check y2 against elements + y
                    temp_diffs = current_diffs | get_new_diffs(elements, y)
                    for v in elements + [y]:
                        dd = abs(y2 - v)
                        if dd in temp_diffs:
                            break
                    else:
                        # Both are valid
                        new_diffs_y = get_new_diffs(elements, y)
                        new_diffs_y2 = get_new_diffs(elements + [y], y2)
                        new_covered = covered | new_diffs_y | new_diffs_y2
                        new_diffs_set = current_diffs | new_diffs_y | new_diffs_y2
                        new_elements = sorted(elements + [y, y2])
                        
                        if search(new_elements, new_covered, new_diffs_set, depth + 1):
                            return True
            
            return False
        
        # Start search
        search(elements, covered, current_diffs, 0)
        return best_result
    
    # Iterative deepening: try smaller bounds first
    # Start from a reasonable lower bound
    # Theoretical minimum: need 31 differences, each element adds at most (n-1) diffs
    # With n elements, max diffs = n*(n-1)/2. Need 31, so n >= 9 (9*8/2=36)
    # Minimum diameter for a (n,k) Golomb ruler...
    
    # Let's try bounds from a reasonable minimum up to the greedy result
    # Start from max(greedy) - 1 and go down? No, let's go up from a lower bound.
    
    # Actually, let's do iterative deepening from a lower bound upward
    # Lower bound: at least 30 (since diff 30 must exist)
    
    lb = 30
    ub = best_diameter - 1  # Try to beat the greedy
    
    # Try decreasing bounds to find the minimum
    # But that's expensive. Instead, let's do a few targeted searches.
    
    # Try a few bounds below the greedy result
    for bound in range(ub, lb, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        # Time limit for this bound: give more time to better (smaller) bounds
        time_limit = min(30.0, DEADLINE - (time.time() - start_time) - 5)
        if time_limit <= 0:
            break
        backtracking_search(bound, time_limit)
        if best_diameter <= bound:
            # Found a solution with this bound, try smaller
            ub = bound - 1
        else:
            # No solution found with this bound, but we might have missed it
            # Move to next (larger) bound... wait, we're going down
            # Actually, if we go from ub down to lb, and find a solution at bound,
            # we should try bound-1 next
            pass
    
    # Final check: make sure we have a valid result
    if best_result is None:
        best_result = greedy_sol
    
    # Validate
    def validate(sol):
        if len(sol) > 32:
            return False
        if 0 not in sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        diffs = []
        for i in range(len(sol)):
            for j in range(i+1, len(sol)):
                diffs.append(sol[j] - sol[i])
        if len(set(diffs)) != len(diffs):
            return False
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    if not validate(best_result):
        best_result = greedy_sol
    
    return best_result
# EVOLVE-BLOCK-END
