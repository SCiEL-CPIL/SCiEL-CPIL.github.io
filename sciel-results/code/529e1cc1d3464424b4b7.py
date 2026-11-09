import time
import heapq

def construct():
    deadline = time.time() + 80

    N = 512
    MAX_CARD = 48

    # Precompute k=2 conflicts: for each x, which pairs (a,b) have a+b = 2x
    # conflict[x] = set of y such that there exists z != x,y with x+z = 2y OR y+z = 2x
    # Actually: x is mean of {a,b} iff a+b = 2x, so x conflicts with a and b.
    # Also a is mean of {x,b} iff x+b = 2a, so a conflicts with x and b.
    # So for each triple {a,b,x} where a+b=2x (or x+b=2a, or a+x=2b), all three are in conflict.
    
    # Let's compute: for each x in 1..512, the set of y such that x and y are in a 3-AP together
    # i.e., there exists z such that {x,y,z} forms an arithmetic progression
    # In a 3-AP {a, m, b} with a+b=2m, all three elements conflict with each other.
    
    # conflict_set[x] = set of y != x such that x and y appear together in some 3-AP within [1,512]
    conflict_set = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != a and m != b:
                    # a, m, b form 3-AP, m is mean of {a,b}
                    conflict_set[a].add(m)
                    conflict_set[b].add(m)
                    conflict_set[m].add(a)
                    conflict_set[m].add(b)
    # Also handle case where a is mean: a = (b+c)/2 => 2a = b+c
    # This is already covered by the above when we iterate all pairs.
    # And c is mean: c = (a+b)/2, also covered.
    
    # Greedy with conflict-aware ordering
    selected = set()
    remaining = list(range(1, N + 1))
    
    # Priority: number of conflicts with already-selected elements
    # Use a heap: (conflict_count, value)
    
    while len(selected) < MAX_CARD and remaining and time.time() < deadline - 30:
        # Find candidate with minimum conflicts with selected
        best = None
        best_conf = float('inf')
        for x in remaining:
            if x in selected:
                continue
            conf = len(conflict_set[x] & selected)
            if conf < best_conf:
                best_conf = conf
                best = x
                if conf == 0:
                    break
        if best is None:
            break
        selected.add(best)
        remaining.remove(best)
    
    # Local search: try to add excluded elements, remove if needed
    excluded = [x for x in range(1, N + 1) if x not in selected]
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for x in excluded[:]:
            if len(selected) >= MAX_CARD:
                break
            # Check if x can be added without conflict
            if len(conflict_set[x] & selected) == 0:
                selected.add(x)
                excluded.remove(x)
                improved = True
            else:
                # Try to remove one conflicting element and add x
                conflicts = conflict_set[x] & selected
                for y in conflicts:
                    # Check if removing y and adding x is beneficial
                    # Remove y, add x, check if any new conflicts introduced
                    new_selected = selected - {y} | {x}
                    # Check if x conflicts with any element in new_selected (excluding y)
                    if len(conflict_set[x] & new_selected) == 0:
                        # Also check that no other element in new_selected now has new conflicts
                        # This is complex; for now just do the swap
                        selected = new_selected
                        excluded.remove(x)
                        excluded.append(y)
                        improved = True
                        break
                if improved:
                    break
    
    # Final verification with full subset-sum DP
    def is_valid_full(s):
        s = sorted(s)
        n = len(s)
        for i, a in enumerate(s):
            others = [s[j] for j in range(n) if j != i]
            diffs = [y - a for y in others]
            max_sum = sum(abs(d) for d in diffs)
            if max_sum == 0:
                continue
            # dp[k] = set of achievable sums with exactly k elements from diffs
            dp = [set() for _ in range(n)]
            dp[0].add(0)
            for d in diffs:
                for k in range(len(diffs) - 1, -1, -1):
                    if not dp[k]:
                        continue
                    for sm in dp[k]:
                        dp[k + 1].add(sm + d)
                for k in range(2, n):
                    if 0 in dp[k]:
                        return False
        return True
    
    result = list(selected)
    if not is_valid_full(result):
        # Fallback: greedy without local search, verify each step
        selected2 = []
        for x in range(1, N + 1):
            if len(selected2) >= MAX_CARD:
                break
            candidate = selected2 + [x]
            if is_valid_full(candidate):
                selected2 = candidate
        result = selected2
    
    return result[:MAX_CARD]
