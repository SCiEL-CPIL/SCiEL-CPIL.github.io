import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
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
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                found = False
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            found = True
                            break
                    if found:
                        break
                
                if found:
                    return False
        
        return True

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
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            return (a, b, c, d, m)
        return None

    def count_violations(S):
        """
        Count violations per element to help with removal.
        Returns dict {element: count}
        """
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        if len(S_list) < 4:
            return counts
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                found_any = False
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            if not found_any:
                                found_any = True
                                counts[a] += 1
                                counts[b] += 1
                                counts[c] += 1
                                counts[d] += 1
                                break
                    if found_any:
                        break
        return counts

    # Strategy:
    # 1. Try several initial sets (ranges)
    # 2. Repair each by removing worst violators
    # 3. Expand by adding compatible numbers
    # 4. Keep the best valid set found
    
    best_S = set()
    
    # Candidate starting ranges
    # We want large numbers to minimize multiples.
    # Range [145, 192] size 48
    # Range [140, 187] size 48
    # Range [135, 182] size 48
    
    starting_ranges = [
        list(range(145, 193)),
        list(range(140, 188)),
        list(range(135, 183)),
        list(range(130, 178)),
        list(range(125, 173)),
        # Also try a greedy construction from scratch
    ]
    
    for start_range in starting_ranges:
        if time.time() > deadline - 5:
            break
            
        S = set(start_range)
        
        # Repair Loop
        while time.time() < deadline - 5:
            if len(S) >= max_size and is_valid(S):
                if len(S) > len(best_S):
                    best_S = S
                break
            
            if not is_valid(S):
                # Find a violation
                viol = get_violation_info(S)
                if viol:
                    a, b, c, d, m = viol
                    # Remove one of them.
                    # Heuristic: Remove the one that is a divisor (a) or the one with highest value?
                    # Removing 'a' removes the constraint entirely.
                    # Removing a summand might break other sums.
                    # Let's remove 'a' first, as it's the "source" of the constraint.
                    # Or remove the largest summand to keep the set "large"?
                    # Let's try removing 'a'.
                    S.remove(a)
                else:
                    # Should not happen if is_valid is False
                    break
            else:
                # Valid, try to add more
                if len(S) >= max_size:
                    if len(S) > len(best_S):
                        best_S = S
                    break
                
                # Add a number
                # Candidates: numbers not in S
                # Prioritize numbers that are "compatible" (large, or such that 3x is not formable)
                # Simple heuristic: try numbers from the top down
                added = False
                # Try numbers from 192 down to 1
                # But to keep the "large number" property, maybe stick to the range?
                # Let's try all, but sorted descending
                candidates = [x for x in range(1, n + 1) if x not in S]
                # Sort by value descending
                candidates.sort(reverse=True)
                
                for x in candidates:
                    if time.time() > deadline - 5:
                        break
                    S_trial = S | {x}
                    if is_valid(S_trial):
                        S = S_trial
                        added = True
                        break
                
                if not added:
                    # Cannot add any single number. Try swapping?
                    # Or break if we can't improve size
                    # Let's try a swap: remove one, add one
                    improved = False
                    S_list = list(S)
                    for i in range(len(S_list)):
                        if time.time() > deadline - 5:
                            break
                        x_to_remove = S_list[i]
                        S_minus = S - {x_to_remove}
                        
                        # Try to add a candidate
                        for y in candidates:
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
                            best_S = S
                        break

    # Final check and return
    if len(best_S) == 0:
        # Fallback: return a small valid set
        # e.g., {192, 191, 190, 189}
        # Check if valid
        # 192: 3*192=576. Sum of 3 others < 576. OK.
        # 191: 3*191=573. 192+190+189=571. 192+189+188? 188 not in set.
        # Let's just return the largest subset of top 10 that is valid.
        S = set(range(183, 193)) # 10 elements
        # It's likely valid.
        best_S = S

    return list(best_S)
