import time
import random
import numpy as np
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
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]

        # For each a in S
        for a in S_list:
            # Find multiples of a in [min_sum, max_sum]
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a

            for m in range(start_m, end_m + 1):
                target = m * a
                # Check if target can be formed by 3 distinct elements from S \ {a}
                others = [x for x in S_list if x != a]
                others_set = set(others)

                # Iterate pairs (b, c) from S \ {a}
                # d = target - b - c
                # Check if d in S \ {a} and d > c
                found = False
                for i in range(len(others)):
                    b = others[i]
                    # Pruning: if b is too large, no valid c, d
                    # b < c < d => 3b < b+c+d = target => b < target/3
                    # Also b < target/2 roughly
                    if b >= target // 3 + 1:
                        # Since sorted? No, others is not necessarily sorted.
                        # But if b is large, c and d must be larger.
                        # Let's just check.
                        pass
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
        Returns a dict {element: count}
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

                # We only need to know if a violation exists for this a,
                # but to weight removal, we might want to count how many
                # distinct violations involve each element.
                # For simplicity and speed, let's just count if a is a divisor in a violation.
                # And maybe increment the summands too?
                
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
                                # Mark all 4 as involved in at least one violation
                                # We can be more precise by counting all, but that's expensive.
                                # Let's just increment by 1 for now.
                                counts[a] += 1
                                counts[b] += 1
                                counts[c] += 1
                                counts[d] += 1
                                found_any = True
                                # Continue to find more violations for this a?
                                # If we break, we miss other violations involving a.
                                # But for "most problematic", finding the first is a heuristic.
                                # Let's not break to get a better count?
                                # No, let's break to save time, assuming one violation is enough signal.
                                # Actually, let's count all for accuracy if time permits?
                                # For now, break.
                                break
                    if found_any:
                        # Break out of j loop
                        pass
                    # Don't break i loop immediately?
                    # If we found a violation for this a, we can stop checking this a?
                    # The count for a is incremented.
                    # But b, c, d might be in other violations with different a's.
                    # So we should continue checking other a's.
                    # But for this specific a, if we found one, do we need more?
                    # Probably not for a simple heuristic.
                    break

        return counts

    def can_add(S, x):
        """
        Check if x can be added to S without creating a violation.
        S is a set.
        """
        if x in S:
            return False

        S_list = list(S)
        S_set = set(S_list)

        # New set S' = S U {x}
        # Violations can involve:
        # 1. x as divisor: x | (a+b+c) for a,b,c in S.
        # 2. a in S as divisor: a | (x+b+c) for b,c in S\{a}.

        # Check 1: x as divisor
        # a+b+c = k*x
        # a,b,c in S.
        # Since a,b,c <= 192, sum <= 576.
        # k*x <= 576 => k <= 576/x.
        
        max_k1 = 576 // x
        for k in range(1, max_k1 + 1):
            target = k * x
            # Find a, b, c in S such that a+b+c = target.
            # Iterate a in S
            for a in S_list:
                rem1 = target - a
                # Iterate b in S, b > a
                for b in S_list:
                    if b <= a: continue
                    rem2 = rem1 - b
                    if rem2 <= b: continue
                    if rem2 in S_set:
                        # a < b < rem2. Distinct. In S.
                        return False

        # Check 2: a in S as divisor
        # a | (x+b+c) => x+b+c = k*a
        # b,c in S\{a}.
        
        for a in S_list:
            # x+b+c <= 192+192+191 = 575
            # k*a <= 575 => k <= 575/a
            max_k2 = 575 // a
            for k in range(1, max_k2 + 1):
                target = k * a
                # b+c = target - x
                rem1 = target - x
                if rem1 <= 0: continue
                # Find b, c in S\{a} such that b+c = rem1.
                # Iterate b in S\{a}
                for b in S_list:
                    if b == a: continue
                    c = rem1 - b
                    if c <= b: continue
                    if c == a: continue
                    if c in S_set:
                        # b < c, c != a, b != a.
                        # b, c in S.
                        return False

        return True

    # Strategy:
    # 1. Try several initial sets (ranges)
    # 2. Repair each by removing worst violators
    # 3. Expand by adding compatible numbers
    # 4. Keep the best valid set found

    best_S = set()

    # Candidate starting ranges
    # We want large numbers to minimize multiples.
    starting_ranges = [
        list(range(145, 193)),  # Size 48
        list(range(140, 188)),  # Size 48
        list(range(135, 183)),  # Size 48
        list(range(130, 178)),  # Size 48
        list(range(125, 173)),  # Size 48
        list(range(120, 168)),  # Size 48
        list(range(115, 163)),  # Size 48
        list(range(110, 158)),  # Size 48
    ]

    for start_range in starting_ranges:
        if time.time() > deadline - 5:
            break

        S = set(start_range)

        # Repair Phase
        # While invalid, remove the element with the highest violation count (or the divisor in the first violation)
        repair_steps = 0
        while time.time() < deadline - 10:
            if is_valid(S):
                break

            # Get violation info
            viol = get_violation_info(S)
            if not viol:
                break

            a, b, c, d, m = viol
            # Heuristic: Remove the divisor 'a' or the largest summand?
            # Removing 'a' eliminates the constraint source.
            # Removing a large summand might help with other constraints.
            # Let's try removing 'a' first.
            S.remove(a)
            repair_steps += 1

            # If we removed too many, break
            if len(S) < max_size - 10:
                break

        # Expansion Phase
        # Try to add numbers from 1..192 not in S.
        # Sort candidates descending (prefer large numbers).
        candidates = [x for x in range(n, 0, -1) if x not in S]

        # We can use a local search / hill climbing approach here.
        # Try to add one, then maybe swap.
        # For simplicity, greedy add.
        added = 0
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if len(S) >= max_size:
                break
            if can_add(S, x):
                S.add(x)
                added += 1

        # If we reached max_size and it's valid, it's a great candidate.
        if len(S) >= max_size and is_valid(S):
            if len(S) > len(best_S):
                best_S = S.copy()
            # Can we do better? No, max is 48.
            # So we can break?
            # Maybe a different start range yields a set that allows easier expansion?
            # But if we have 48, we are done maximizing cardinality.
            # So we can return immediately?
            # The task says "Maximize cardinality". 48 is the max allowed.
            # So if we find a valid set of 48, we can return it.
            return list(S)

        # If we didn't reach 48, keep the best so far.
        if len(S) > len(best_S):
            best_S = S.copy()

    # If we didn't find a valid set of 48 in the main loop,
    # we might have a valid set of size < 48 in best_S.
    # Let's try one more time with a different strategy if time permits?
    # Or just return best_S.

    # Fallback: if best_S is empty or small, try a greedy construction from scratch.
    if len(best_S) < 10:
        S = set()
        # Greedy: add numbers from 192 down to 1
        for x in range(n, 0, -1):
            if time.time() > deadline - 2:
                break
            if len(S) >= max_size:
                break
            if can_add(S, x):
                S.add(x)
        if len(S) > len(best_S):
            best_S = S

    # Final check
    if is_valid(best_S):
        return list(best_S)
    else:
        # Should not happen if is_valid is correct, but just in case.
        # Return a small valid set.
        # Numbers > 96 are safe?
        # If a > 96, 2a > 192. Sum of 3 others <= 3*192 = 576.
        # a | (b+c+d).
        # If a is large, few multiples.
        # Let's just return the largest valid set we can find quickly.
        S = set(range(145, 193))
        while not is_valid(S) and S:
            viol = get_violation_info(S)
            if viol:
                a, b, c, d, m = viol
                S.remove(a)
            else:
                break
        return list(S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
