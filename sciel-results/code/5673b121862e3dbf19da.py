import time
import sys

def construct():
    deadline = time.time() + 145
    
    def backtrack(elements, diffs, missing_mask, k, D):
        """
        elements: sorted list of chosen elements (0 is first)
        diffs: set of all positive differences
        missing_mask: bitmask of which diffs in 1..30 are still missing
        k: target size
        D: max allowed element
        Returns: list of elements or None
        """
        if time.time() > deadline - 5:
            return None
            
        if len(elements) == k:
            if missing_mask == 0:
                return elements
            return None
            
        # Pruning: Can we cover all missing diffs with remaining slots?
        # Max possible new diffs
        remaining_slots = k - len(elements)
        # Total diffs at end = C(k, 2)
        # Current diffs = len(diffs)
        # New diffs = C(k, 2) - len(diffs)
        # We need to cover popcount(missing_mask) diffs.
        # Note: This is a weak bound because new diffs might be > 30.
        # But if even the total count is insufficient, we prune.
        if (k * (k - 1) // 2) - len(diffs) < bin(missing_mask).count('1'):
            return None

        current_max = elements[-1]
        
        # Generate candidates
        # We iterate x from current_max + 1 to D
        # To improve speed, we can filter and sort
        candidates = []
        for x in range(current_max + 1, D + 1):
            # Check Sidon property
            valid = True
            new_diffs = set(diffs)
            for a in elements:
                d = x - a
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Count covered missing diffs
            covered_count = 0
            for d in new_diffs - diffs:
                if 1 <= d <= 30 and (missing_mask >> (d-1)) & 1:
                    covered_count += 1
                    
            # Heuristic: If we don't cover any missing diff, it's less preferred
            # unless we are forced to.
            candidates.append((x, covered_count, new_diffs))
            
        # Sort by covered_count desc, then x asc
        candidates.sort(key=lambda item: (-item[1], item[0]))
        
        for x, covered_count, new_diffs in candidates:
            # Optional Prune: If covered_count == 0 and we have few slots left,
            # it might be bad. But let's let the search handle it.
            
            new_missing_mask = missing_mask
            for d in new_diffs - diffs:
                if 1 <= d <= 30:
                    new_missing_mask &= ~(1 << (d-1))
                    
            result = backtrack(elements + [x], new_diffs, new_missing_mask, k, D)
            if result is not None:
                return result
                
        return None

    best_result = None
    best_max_elem = float('inf')
    
    # Iterate over possible set sizes k
    # Minimum k is 9
    for k in range(9, 13):
        # Lower bound for D
        lower_bound = k * (k - 1) // 2
        
        # We want to find the minimal D for this k
        # If we find a solution for k with D, and D < lower_bound(k+1),
        # then no larger k can beat it.
        
        found_for_k = False
        for D in range(lower_bound, 100):
            if time.time() > deadline - 10:
                break
                
            # Initial state: elements=[0], diffs=set(), missing_mask=(1<<30)-1
            result = backtrack([0], set(), (1 << 30) - 1, k, D)
            if result is not None:
                if result[-1] < best_max_elem:
                    best_max_elem = result[-1]
                    best_result = result
                found_for_k = True
                # Since D is increasing, this is the min D for this k.
                # Check if we can stop entirely
                if D < (k + 1) * k // 2:
                    return result
                break # Stop increasing D for this k, try next k? 
                # No, if we found a solution for k, we should check if a smaller D exists?
                # We are iterating D in increasing order, so the first one found is minimal for this k.
                # But maybe k+1 has a smaller D?
                # As established, D_min(k+1) >= C(k+1, 2).
                # If D_min(k) < C(k+1, 2), then k is better.
                # If D_min(k) >= C(k+1, 2), it's possible k+1 is better.
                # So we should continue to next k if the condition is met.
            else:
                continue
        
        if best_result is not None:
            # Check if we can stop searching for larger k
            if best_max_elem < (k + 1) * k // 2:
                return best_result

    if best_result is not None:
        return best_result
        
    # Fallback
    return [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 32, 35, 38, 40, 44, 47, 50, 53, 55, 59, 62, 65, 68, 70, 74, 77, 80, 83, 85, 89, 92, 95]
