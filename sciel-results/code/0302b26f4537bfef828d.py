import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 192
    max_size = 48
    
    def is_valid(S):
        """
        Check if S is valid.
        For each a in S, check if any multiple of a in the sum range
        can be formed by 3 distinct other elements.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return True
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        S_set = set(S_list)
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                
                # Check if target can be formed by 3 distinct elements from S \ {a}
                # Use set for O(1) lookup
                others_set = S_set - {a}
                
                # Iterate over pairs (b, c) with b < c, then d = target - b - c
                # Need d > c, d in others_set, d != a
                for i, b in enumerate(S_list):
                    if b == a:
                        continue
                    # b must be small enough: b < (target - 2)/3 roughly
                    # But let's just check all and rely on the d > c check
                    for j, c in enumerate(S_list):
                        if j <= i:
                            continue
                        if c == a:
                            continue
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            return False
        
        return True

    def get_violations(S):
        """
        Return list of all violations (a, b, c, d, m).
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return []
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        S_set = set(S_list)
        violations = []
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others_set = S_set - {a}
                
                found_any = False
                for i, b in enumerate(S_list):
                    if b == a:
                        continue
                    for j, c in enumerate(S_list):
                        if j <= i:
                            continue
                        if c == a:
                            continue
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            violations.append((a, b, c, d, m))
                            found_any = True
                            break
                    if found_any:
                        break
        
        return violations

    def count_violations_per_element(S):
        """
        Count how many violations each element participates in.
        """
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        if len(S_list) < 4:
            return counts
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        S_set = set(S_list)
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others_set = S_set - {a}
                
                found_any = False
                for i, b in enumerate(S_list):
                    if b == a:
                        continue
                    for j, c in enumerate(S_list):
                        if j <= i:
                            continue
                        if c == a:
                            continue
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            counts[a] += 1
                            counts[b] += 1
                            counts[c] += 1
                            counts[d] += 1
                            found_any = True
                            break
                    if found_any:
                        break
        
        return counts

    def get_violation_info(S):
        """
        Returns (a, b, c, d, m) for the first violation found, or None.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return None
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        S_set = set(S_list)
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others_set = S_set - {a}
                
                for i, b in enumerate(S_list):
                    if b == a:
                        continue
                    for j, c in enumerate(S_list):
                        if j <= i:
                            continue
                        if c == a:
                            continue
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            return (a, b, c, d, m)
        return None

    # Strategy:
    # 1. Try several initial sets (ranges)
    # 2. Repair each by removing worst violators
    # 3. Expand by adding compatible numbers
    # 4. Keep the best valid set found
    
    best_S = set()
    
    # Candidate starting ranges
    starting_ranges = [
        list(range(145, 193)),
        list(range(140, 188)),
        list(range(135, 183)),
        list(range(130, 178)),
        list(range(125, 173)),
    ]
    
    for start_range in starting_ranges:
        if time.time() > deadline - 5:
            break
            
        S = set(start_range)
        
        # Repair Loop
        while time.time() < deadline - 5:
            if len(S) >= max_size and is_valid(S):
                if len(S) > len(best_S):
                    best_S = set(S)
                break
            
            if not is_valid(S):
                # Count violations per element
                counts = count_violations_per_element(S)
                if counts:
                    # Remove the element with the highest violation count
                    worst = max(counts.keys(), key=lambda x: counts[x])
                    S.discard(worst)
                else:
                    break
            else:
                # Valid, try to add more
                if len(S) >= max_size:
                    if len(S) > len(best_S):
                        best_S = set(S)
                    break
                
                # Add a number
                added = False
                # Try numbers from 192 down to 1
                for x in range(n, 0, -1):
                    if time.time() > deadline - 5:
                        break
                    if x in S:
                        continue
                    S_trial = S | {x}
                    if is_valid(S_trial):
                        S = S_trial
                        added = True
                        break
                
                if not added:
                    # Cannot add any single number. Try swaps.
                    improved = False
                    S_list = list(S)
                    # Try removing each element and adding a different one
                    for x_to_remove in S_list:
                        if time.time() > deadline - 5:
                            break
                        S_minus = S - {x_to_remove}
                        
                        # Try to add a candidate
                        for y in range(n, 0, -1):
                            if y in S_minus:
                                continue
                            S_trial = S_minus | {y}
                            if is_valid(S_trial):
                                S = S_trial
                                improved = True
                                break
                        if improved:
                            break
                    
                    if not improved:
                        # Stuck
                        if len(S) > len(best_S):
                            best_S = set(S)
                        break

    # Final check and return
    if len(best_S) == 0:
        # Fallback: return a small valid set
        best_S = set(range(183, 193))

    return list(best_S)
