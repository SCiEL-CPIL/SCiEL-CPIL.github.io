def construct():
    """Construct a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.
    
    Uses greedy insertion with lookahead, then local improvement via simulated annealing.
    """
    import random
    import math
    
    random.seed(42)
    
    def is_valid_sidon(arr):
        """Check if arr is a valid Sidon set (all pairwise sums a+b with a<=b are distinct)."""
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_conflicts(arr, candidate, existing_sums):
        """Check if adding candidate to arr creates any sum conflicts."""
        new_sums = set()
        for x in arr:
            s = x + candidate
            new_sums.add(s)
        new_sums.add(2 * candidate)
        
        # Check for conflicts with existing sums
        for s in new_sums:
            if s in existing_sums:
                return True
        # Check for conflicts within new sums themselves
        if len(new_sums) != len(arr) + 1:
            return True
        return False
    
    def compute_sums(arr):
        """Compute all pairwise sums for arr."""
        sums = set()
        for i, x in enumerate(arr):
            for y in arr[i:]:
                sums.add(x + y)
        return sums
    
    def greedy_construct():
        """Greedy construction: add candidates in order that minimizes future conflicts."""
        a = [1, 2, 5, 11]
        sums = compute_sums(a)
        
        candidates = list(range(12, 285))
        
        for _ in range(len(candidates)):
            best_candidate = None
            best_score = float('inf')
            
            for x in candidates:
                # Check if x can be added
                new_sums = {x + y for y in a} | {2 * x}
                if new_sums & sums:
                    continue
                if len(new_sums) != len(a) + 1:
                    continue
                
                # Score: prefer candidates that leave more room (fewer conflicts with remaining)
                # Use a simple heuristic: prefer smaller candidates first, but check feasibility
                # Count how many remaining candidates would conflict with x
                conflict_count = 0
                for y in candidates:
                    if y == x:
                        continue
                    # Check if x + y would conflict with any current or future sum
                    # Simple heuristic: just count direct conflicts
                    if (x + y) in sums or (2 * y) in sums:
                        conflict_count += 1
                
                score = conflict_count
                if score < best_score or (score == best_score and (best_candidate is None or x < best_candidate)):
                    best_score = score
                    best_candidate = x
            
            if best_candidate is None:
                # Try any valid candidate
                for x in candidates:
                    new_sums = {x + y for y in a} | {2 * x}
                    if not (new_sums & sums) and len(new_sums) == len(a) + 1:
                        best_candidate = x
                        break
            
            if best_candidate is None:
                break
            
            a.append(best_candidate)
            sums = compute_sums(a)
            candidates.remove(best_candidate)
            a.sort()
        
        return a
    
    def local_improve(arr, max_iterations=5000):
        """Local improvement: try replacing elements or adding new ones."""
        arr = sorted(arr)
        best_arr = arr[:]
        best_size = len(arr)
        
        for iteration in range(max_iterations):
            sums = compute_sums(arr)
            improved = False
            
            # Try adding a new element
            for x in range(1, 285):
                if x in arr:
                    continue
                new_sums = {x + y for y in arr} | {2 * x}
                if new_sums & sums:
                    continue
                if len(new_sums) != len(arr) + 1:
                    continue
                
                new_arr = sorted(arr + [x])
                if len(new_arr) > best_size:
                    best_arr = new_arr[:]
                    best_size = len(new_arr)
                    improved = True
                    break
                else:
                    # Accept with some probability for diversity
                    if random.random() < 0.1:
                        arr = new_arr
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping an element
            for i in range(len(arr)):
                old_val = arr[i]
                for new_val in range(1, 285):
                    if new_val in arr or new_val == old_val:
                        continue
                    
                    candidate_arr = arr[:i] + [new_val] + arr[i+1:]
                    candidate_arr = sorted(candidate_arr)
                    
                    # Check validity
                    cand_sums = set()
                    valid = True
                    for j in range(len(candidate_arr)):
                        for k in range(j, len(candidate_arr)):
                            s = candidate_arr[j] + candidate_arr[k]
                            if s in cand_sums:
                                valid = False
                                break
                            cand_sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Check if required elements are still present
                        if all(req in candidate_arr for req in [1, 2, 5, 11]):
                            arr = candidate_arr
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved and iteration > max_iterations // 2:
                # Try random restart with perturbation
                if random.random() < 0.05:
                    arr = best_arr[:]
                    # Remove a random non-required element
                    non_required = [x for x in arr if x not in [1, 2, 5, 11]]
                    if non_required:
                        to_remove = random.choice(non_required)
                        arr.remove(to_remove)
        
        return best_arr
    
    # Step 1: Greedy construction
    arr = greedy_construct()
    
    # Step 2: Local improvement
    arr = local_improve(arr, max_iterations=3000)
    
    # Step 3: Verify final result
    final = sorted(set(arr))
    if not all(x in final for x in [1, 2, 5, 11]):
        # Fallback to simple greedy
        final = [1, 2, 5, 11]
        sums = compute_sums(final)
        for x in range(12, 285):
            new_sums = {x + y for y in final} | {2 * x}
            if not (new_sums & sums) and len(new_sums) == len(final) + 1:
                final.append(x)
                sums = compute_sums(final)
        final = sorted(final)
    
    return final
