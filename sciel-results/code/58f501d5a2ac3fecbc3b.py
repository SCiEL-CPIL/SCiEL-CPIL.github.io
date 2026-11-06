import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None

        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)

        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def find_min_diameter_set(max_candidate=80):
        """
        Backtracking with strong pruning and variable ordering.
        Uses bitmask for covered differences for speed.
        """
        for D in range(30, max_candidate + 1):
            if time.time() > deadline - 5:
                break
            
            found = [None]
            node_count = [0]
            max_nodes = 2000000  # Limit nodes per D to prevent timeout

            def backtrack(current_elements, all_diffs, covered_mask, start_idx):
                if found[0] is not None:
                    return
                
                if time.time() > deadline - 5:
                    return
                
                node_count[0] += 1
                if node_count[0] > max_nodes:
                    return
                
                # Goal check: all 30 differences covered (bits 0..29 set)
                if covered_mask == (1 << 30) - 1:
                    found[0] = current_elements[:]
                    return
                
                m = len(current_elements)
                # Count missing differences
                missing_count = 30 - bin(covered_mask).count('1')
                
                # Pruning: not enough elements left
                if 32 - m <= 0:
                    if missing_count > 0:
                        return
                
                # Pruning: not enough candidate positions
                k_max = D - start_idx + 1
                if k_max <= 0:
                    if missing_count > 0:
                        return
                else:
                    # Max new diffs from k_max new elements: each new element 
                    # creates diffs with all existing m elements + diffs among new elements
                    max_new = k_max * m + k_max * (k_max - 1) // 2
                    if max_new < missing_count:
                        return
                
                # Stronger pruning: total diffs so far + max possible must cover 30
                current_diff_count = len(all_diffs)
                total_possible = current_diff_count + max_new
                if total_possible < 30:
                    return
                
                # Generate and score candidates
                candidates = []
                for c in range(start_idx, D + 1):
                    ok = True
                    new_diffs = []
                    for x in current_elements:
                        d = c - x
                        if d in all_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Compute new coverage bitmask
                    new_mask = 0
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_mask |= (1 << (d - 1))
                    
                    # Count new covered differences
                    new_cover_count = bin(new_mask & ~covered_mask).count('1')
                    
                    candidates.append((c, new_cover_count, new_diffs, new_mask))
                
                # Sort by new_cover_count descending, then by element value ascending
                candidates.sort(key=lambda x: (-x[1], x[0]))
                
                for c, new_cover_count, new_diffs, new_mask in candidates:
                    if found[0] is not None:
                        return
                    
                    new_covered_mask = covered_mask | new_mask
                    new_all_diffs = all_diffs.union(new_diffs)
                    new_elements = current_elements + [c]
                    
                    backtrack(new_elements, new_all_diffs, new_covered_mask, c + 1)
                    
                    if found[0] is not None:
                        return
            
            backtrack([0], set(), 0, 1)
            
            if found[0] is not None:
                return found[0]
                
        return None

    def greedy_fallback(time_limit):
        """
        Simple greedy heuristic if backtracking fails or is too slow.
        """
        a = [0]
        diffs = set()
        x = 1
        while time.time() < time_limit:
            # Find smallest x not in a such that adding x keeps Sidon property
            # and helps cover missing diffs
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            
            if valid:
                # Check if this x covers any new required diff
                new_cover = False
                for y in a:
                    if x > y:
                        d = x - y
                        if 1 <= d <= 30 and d not in diffs:
                            new_cover = True
                            break
                            
                # Accept if it covers new diff or if we have fewer than 9 elements (to ensure enough room)
                if new_cover or len(a) < 9:
                    for y in a:
                        if x > y:
                            diffs.add(x - y)
                    a.append(x)
                    
                    if all(d in diffs for d in range(1, 31)):
                        return a
            
            x += 1
            if x > 1000: # Safety break
                break
                
        return a

    # Main Execution
    # Try smaller max_candidate first for speed, expand if needed
    best_set = find_min_diameter_set(max_candidate=60)
    
    if best_set is None and time.time() < deadline - 10:
        best_set = find_min_diameter_set(max_candidate=80)
    
    if best_set is None:
        # 2. Fall back to greedy if exact search fails (shouldn't happen for D<80 usually)
        print("Exact search failed or timed out, using greedy fallback", file=sys.stderr)
        best_set = greedy_fallback(deadline - 5)
        
    # Final Validation
    if best_set is None:
        # Emergency: just return a known valid set if even greedy fails?
        # A known Sidon set covering 1..30: 
        # {0, 1, 4, 10, 12, 28, 32, 45, 63} has diameter 63.
        # Let's verify quickly.
        emergency = [0, 1, 4, 10, 12, 28, 32, 45, 63]
        v, _, _ = is_valid(emergency)
        if v:
            return emergency
        else:
            # This should never happen, but just in case
            return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    return best_set

# EVOLVE-BLOCK-END
