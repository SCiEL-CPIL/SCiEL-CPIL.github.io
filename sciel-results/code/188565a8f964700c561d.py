import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82.0

    def valid(S):
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(i+1, len(L)):
                b = L[j]
                for k in range(j, len(L)):
                    c = L[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        L = sorted(S)
        # Case 1: x is smallest
        for i, b in enumerate(L):
            if b <= x:
                continue
            for c in L[i:]:
                if (b + c) % x == 0:
                    return False
        # Case 2: x is middle
        for a in L:
            if a >= x:
                continue
            for c in L:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        # Case 3: x is largest
        for i, a in enumerate(L):
            for b in L[i+1:]:
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        return True

    def greedy_from_tail(tail_start, order):
        S = set(range(tail_start, 257))
        for candidate in order:
            if candidate in S:
                continue
            if can_add(S, candidate):
                S.add(candidate)
        return S

    def greedy_removal_from_top():
        """
        Start with [129, 256].
        Repeatedly find conflicts and remove the most 'conflicted' element.
        """
        S = set(range(129, 257))
        
        while True:
            if time.time() > deadline - 10:
                break
            
            # Find all conflicts
            L = sorted(S)
            conflict_counts = {}
            has_conflict = False
            
            for i in range(len(L)):
                a = L[i]
                for j in range(i+1, len(L)):
                    b = L[j]
                    for k in range(j, len(L)):
                        c = L[k]
                        if (b + c) % a == 0:
                            has_conflict = True
                            conflict_counts[a] = conflict_counts.get(a, 0) + 1
                            conflict_counts[b] = conflict_counts.get(b, 0) + 1
                            conflict_counts[c] = conflict_counts.get(c, 0) + 1
                            
            if not has_conflict:
                break
            
            # Remove element with max count
            # Tie-breaker: remove smallest? Largest?
            # Smallest 'a' causes more constraints generally.
            # Let's remove the one with max count, tie-break by value (smaller first).
            max_count = max(conflict_counts.values())
            candidates_to_remove = [x for x, cnt in conflict_counts.items() if cnt == max_count]
            remove_target = min(candidates_to_remove)
            
            S.remove(remove_target)
            
        return S

    def local_search(S, max_time):
        S = set(S)
        best = S
        end = start_time + max_time
        while time.time() < end:
            improved = False
            L = sorted(S)
            for i in range(len(L)):
                if time.time() > end:
                    break
                a = L[i]
                S2 = S - {a}
                for candidate in range(1, 257):
                    if candidate in S2:
                        continue
                    if can_add(S2, candidate):
                        S2.add(candidate)
                        for candidate2 in range(1, 257):
                            if candidate2 in S2:
                                continue
                            if can_add(S2, candidate2):
                                S2.add(candidate2)
                        if len(S2) > len(best):
                            best = S2
                            S = set(S2)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return best

    # Run strategies
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy Removal
    if time.time() < deadline - 30:
        S_rem = greedy_removal_from_top()
        if len(S_rem) > best_size:
            best = S_rem
            best_size = len(S_rem)

    # Strategy 2: Greedy Addition (existing)
    tail_starts = [171, 175, 180, 185, 190, 195, 200, 210, 220]
    orders = [
        list(range(1, 171)),
        list(range(170, 0, -1)),
    ]
    rng = random.Random(42)
    for _ in range(6):
        o = list(range(1, 171))
        rng.shuffle(o)
        orders.append(o)

    for ts in tail_starts:
        if time.time() > deadline - 20:
            break
        for order in orders:
            if time.time() > deadline - 20:
                break
            S_add = greedy_from_tail(ts, order)
            if len(S_add) > best_size:
                best = S_add
                best_size = len(S_add)

    # Local Search on Best
    if time.time() < deadline - 5:
        best = local_search(best, max_time=deadline - 5)
        best_size = len(best)

    # Verify
    if not valid(best):
        # Fallback
        best = set(range(171, 257))

    return sorted(best)
