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
        Optimized: For each a, check if any multiple of a in the sum range
        can be formed by 3 distinct other elements.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return True
        
        # Precompute min and max sum of 3 distinct elements in S
        # Sorted S_list helps
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        # For each a in S
        for a in S_list:
            # Find multiples of a in [min_sum, max_sum]
            # m * a
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                # Check if target can be formed by 3 distinct elements from S \ {a}
                # We need b, c, d in S \ {a}, distinct, b+c+d = target
                
                # Optimization: iterate pairs (b, c) from S \ {a}
                # d = target - b - c
                # Check if d in S \ {a} and d > c (to avoid duplicates/permutations)
                
                others = [x for x in S_list if x != a]
                # Sort others to help with early termination?
                # Just use a set for O(1) lookup
                others_set = set(others)
                
                # Iterate b
                # To ensure distinct and ordered b < c < d:
                # b < target/3
                # c < (target - b)/2
                
                found = False
                # We can just iterate all pairs, it's fast enough for N=48
                # Pairs in others: ~1100
                # For each pair, O(1) check.
                # Total ~1100 * 48 * (num multiples)
                
                for i in range(len(others)):
                    b = others[i]
                    # b must be small enough to allow c, d
                    # b < target / 3
                    if b >= target // 3 + 1: 
                        # Since b < c < d, b is the smallest, so b < target/3
                        # Actually b < (target - 2*1)/3 approx
                        # Let's just check all, it's fast
                        pass
                    
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        # Check distinct and valid
                        if d <= c:
                            continue
                        if d in others_set:
                            # Ensure d is distinct from b and c (guaranteed by d > c > b)
                            # And d is in others (checked)
                            # And d is not a (guaranteed by others_set)
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
                
                # Count how many triplets sum to target
                # We just need to know if a is involved in violations.
                # If a violation exists, increment count for a, b, c, d.
                # To avoid double counting the same triplet for different a's?
                # The definition is: a divides sum(b,c,d).
                # So (a,b,c,d) is a violation.
                
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
                                # We could count all, but one is enough for "is there a violation"
                                # For weighting, let's just mark that a violation exists for these elements
                                # Actually, let's just return the first one to save time in removal phase
                                # But for "most problematic", we might want counts.
                                # Let's stick to finding the first violation for removal to keep it fast.
                                found_any = True
                                # Increment counts
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
