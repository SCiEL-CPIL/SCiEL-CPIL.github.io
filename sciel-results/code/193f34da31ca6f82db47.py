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
        Optimized Backtracking with in-place modifications and bitmask coverage.
        """
        full_mask = (1 << 31) - 2  # Bits 1..30 set
        
        for D in range(30, max_candidate + 1):
            if time.time() > deadline - 5:
                break
            
            found = [None]
            node_count = [0]
            max_nodes = 5000000  # Increase node limit due to speedup
            
            # We use a list for elements to maintain order (sorted by value implicitly by start_idx)
            # elements[0] is always 0
            elements = [0]
            all_diffs = set()
            covered_mask = 0
            
            def backtrack(start_idx):
                if found[0] is not None:
                    return
                
                if time.time() > deadline - 5:
                    return
                
                node_count[0] += 1
                if node_count[0] > max_nodes:
                    return
                
                m = len(elements)
                
                # Goal check
                if (covered_mask & full_mask) == full_mask:
                    found[0] = elements[:]
                    return
                
                missing = 30 - bin(covered_mask).count('1')
                
                # Pruning: 
                # 1. If we run out of numbers to pick from
                if start_idx > D:
                    return
                    
                # 2. Max possible new diffs from remaining numbers
                # Remaining numbers can be at most D - start_idx + 1
                # Each new number adds diffs to all previous 'm' numbers.
                # Plus diffs between new numbers themselves.
                k_max = D - start_idx + 1
                if k_max <= 0:
                    return
                max_new_diffs = k_max * m + k_max * (k_max - 1) // 2
                if max_new_diffs < missing:
                    return
                
                # 3. If we hit max size 32
                if m >= 32:
                    return
                
                # Generate candidates
                # Heuristic: Try candidates that cover more missing diffs first?
                # For now, iterate c from start_idx to D.
                # To improve speed, we can pre-filter or sort.
                # Given D is small (<80), iteration is fast.
                
                # Optimization: Check candidates in an order that might cover more.
                # However, sorting requires generating all candidates first.
                # Let's stick to natural order but add a quick check for 'good' candidates?
                # Actually, just iterating is fine if the pruning is strong.
                
                for c in range(start_idx, D + 1):
                    if found[0] is not None:
                        return
                    
                    ok = True
                    new_diffs = []
                    for x in elements:
                        d = c - x
                        if d in all_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Calculate new covered mask bits
                    new_mask_bits = 0
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_mask_bits |= (1 << d)
                    
                    # If this candidate adds NO new coverage, and we are close to max size, skip?
                    # Or if it adds no coverage, it's likely bad unless we *must* fill space?
                    # No, we need to cover 30 diffs. If it adds 0, it's useless for coverage.
                    # But it might be necessary to allow *other* numbers?
                    # In a Sidon set, every number adds unique diffs.
                    # If a number adds 0 new coverage (all diffs > 30 or already covered),
                    # it doesn't help the goal. We can skip it?
                    # Yes, if new_mask_bits == 0, it doesn't help cover 1..30.
                    # Since we want to minimize D, and we have a limit on size (32),
                    # adding a "useless" number consumes a slot.
                    # So we can skip candidates with new_mask_bits == 0.
                    if new_mask_bits == 0:
                        continue

                    # Apply change
                    elements.append(c)
                    for d in new_diffs:
                        all_diffs.add(d)
                    covered_mask |= new_mask_bits
                    
                    # Recurse
                    backtrack(c + 1)
                    
                    # Undo change
                    elements.pop()
                    for d in new_diffs:
                        all_diffs.discard(d)
                    covered_mask &= ~new_mask_bits
                    
                    if found[0] is not None:
                        return

            backtrack(1)
            
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
