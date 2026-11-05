import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    def elapsed():
        return time.time() - start

    def is_valid_sidon_with_coverage(S):
        """Check S is Sidon, contains 0, covers 1..30, at most 32 elements."""
        if len(S) > 32 or 0 not in S:
            return False
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def forward_check(S, used_diffs, need_set):
        """
        After placing elements in S (sorted), check:
        1. All pairwise differences are distinct (Sidon property)
        2. We can still cover all of need_set with remaining placements
        Returns (ok, new_used_diffs, new_need_set)
        """
        n = len(S)
        # Check Sidon property for all pairs
        diff_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diff_set:
                    return False, set(), set()
                diff_set.add(d)
        # Check coverage is still achievable
        # Remaining elements can be at most (32 - n) more
        # Each new element x adds (n) new differences (x - s for s in S)
        # So max new differences = sum over remaining placements
        remaining_slots = 32 - n
        # Max new differences we can add: for k remaining elements,
        # first adds n diffs, second adds n+1, etc.
        max_new_diffs = sum(n + i for i in range(remaining_slots))
        if len(need_set) > max_new_diffs:
            return False, diff_set, need_set
        # Also: differences must be positive and <= bound (handled by caller)
        new_need = need_set - diff_set
        return True, diff_set, new_need

    def dfs_construct(bound, rng):
        """
        DFS with forward checking to find a Sidon set in [0, bound]
        covering 1..30. Returns the set if found, else None.
        """
        need = set(range(1, 31))
        
        def dfs(S, used_diffs, need_set, depth):
            if elapsed() > deadline - 5.0:
                return None
            if not need_set:
                return list(S)
            if len(S) >= 32:
                return None
            
            n = len(S)
            # Lower bound: we need len(need_set) more differences
            # With (32-n) remaining slots, max new diffs = sum(n+i for i in range(32-n))
            remaining = 32 - n
            max_possible_new = sum(n + i for i in range(remaining))
            if len(need_set) > max_possible_new:
                return None
            
            # Try candidate values for next element
            # Start from S[-1]+1 to bound
            # Heuristic: prefer values that cover more needed differences
            candidates = []
            for x in range(S[-1] + 1, bound + 1):
                # Check if x creates any duplicate difference
                ok = True
                new_diffs = []
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                # Score: how many needed differences does this cover?
                score = sum(1 for d in new_diffs if d in need_set)
                candidates.append((score, x))
            
            if not candidates:
                return None
            
            # Sort by score descending (cover more needed diffs first)
            candidates.sort(key=lambda t: -t[0])
            
            for score, x in candidates:
                if elapsed() > deadline - 5.0:
                    return None
                # Try this candidate
                new_used = used_diffs | set(x - s for s in S)
                new_need = need_set - new_used
                new_S = S + [x]
                
                # Forward check
                ok, new_used2, new_need2 = forward_check(new_S, new_used, new_need)
                if not ok:
                    continue
                
                result = dfs(new_S, new_used2, new_need2, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return dfs([0], set(), need, 0)

    def greedy_with_pruning(bound, rng):
        """
        Greedy that picks the element maximizing coverage while maintaining
        Sidon property, with backtracking if stuck.
        """
        S = [0]
        used = set()
        need = set(range(1, 31))
        
        while need and len(S) < 32:
            best_x = None
            best_score = -1
            for x in range(S[-1] + 1, bound + 1):
                ok = True
                score = 0
                for s in S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                    if d in need:
                        score += 1
                if ok and score > best_score:
                    best_score = score
                    best_x = x
                    if best_score == len(need):
                        break
            if best_x is None:
                return None
            S.append(best_x)
            for s in S[:-1]:
                d = S[-1] - s
                used.add(d)
            need -= used
        
        if need:
            return None
        return S

    def local_search_refine(S, bound, rng):
        """Try to reduce max element via local search."""
        S = sorted(S)
        improved = True
        while improved and elapsed() < deadline - 10.0:
            improved = False
            # Try replacing each element with a smaller one
            for i in range(len(S) - 1, 0, -1):  # try largest first
                if S[i] <= 1:
                    continue
                for new_x in range(1, S[i]):
                    if new_x in S:
                        continue
                    trial = S[:i] + [new_x] + S[i + 1:]
                    trial.sort()
                    if is_valid_sidon_with_coverage(trial):
                        S = trial
                        improved = True
                        break
                if improved:
                    break
            # Try removing elements
            for i in range(len(S) - 1, 0, -1):
                trial = S[:i] + S[i + 1:]
                if is_valid_sidon_with_coverage(trial):
                    S = trial
                    improved = True
                    break
        return S

    # Strategy: iterative deepening on the bound
    # Lower bound on max element: for n elements, C(n,2) <= M
    # We need at least 9 elements (C(9,2)=36 >= 30)
    # For n=9: M >= 36, for n=10: M >= 45, etc.
    # Try bounds from low to high
    
    best_S = None
    best_max = float('inf')
    rng = random.Random(42)
    
    # Start from a reasonable lower bound
    # Minimum possible: 9 elements need M >= 36, but Sidon constraints push it higher
    # Known: smallest Sidon set with 9 elements covering 1..30 likely has M around 50-70
    
    for bound in range(40, 120):
        if elapsed() > deadline - 20.0:
            break
        if best_S is not None and bound >= best_max:
            break
        
        # Try DFS construction
        S = dfs_construct(bound, rng)
        if S is not None and is_valid_sidon_with_coverage(S):
            S = local_search_refine(S, bound, rng)
            mx = max(S)
            if mx < best_max:
                best_max = mx
                best_S = list(S)
        
        # Also try greedy as a different heuristic
        if elapsed() < deadline - 15.0:
            S2 = greedy_with_pruning(bound, rng)
            if S2 is not None and is_valid_sidon_with_coverage(S2):
                S2 = local_search_refine(S2, bound, rng)
                mx2 = max(S2)
                if mx2 < best_max:
                    best_max = mx2
                    best_S = list(S2)
    
    # If nothing found, use a generous bound
    if best_S is None:
        for bound in range(120, 300, 10):
            if elapsed() > deadline - 10.0:
                break
            S = dfs_construct(bound, rng)
            if S is not None and is_valid_sidon_with_coverage(S):
                S = local_search_refine(S, bound, rng)
                best_S = list(S)
                best_max = max(S)
                break
            S2 = greedy_with_pruning(bound, rng)
            if S2 is not None and is_valid_sidon_with_coverage(S2):
                S2 = local_search_refine(S2, bound, rng)
                best_S = list(S2)
                best_max = max(S2)
                break
    
    # Final fallback: pure greedy with large bound
    if best_S is None or not is_valid_sidon_with_coverage(best_S):
        S = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            if not need:
                break
            found = False
            for x in range(S[-1] + 1, S[-1] + 2000):
                ok = True
                for s in S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    for s in S[:-1]:
                        used.add(x - s)
                    need -= used
                    found = True
                    break
            if not found:
                break
        if is_valid_sidon_with_coverage(S):
            best_S = S
        else:
            # Absolute last resort
            best_S = [0, 1, 4, 10, 18, 23, 28, 34, 42, 51]
            # Verify and fix if needed
            if not is_valid_sidon_with_coverage(best_S):
                # Build from scratch with very generous bound
                S = [0]
                used = set()
                need = set(range(1, 31))
                for _ in range(32):
                    if not need:
                        break
                    for x in range(S[-1] + 1, 100000):
                        ok = True
                        for s in S:
                            d = x - s
                            if d in used:
                                ok = False
                                break
                        if ok:
                            S.append(x)
                            for s in S[:-1]:
                                used.add(x - s)
                            need -= used
                            break
                best_S = S
    
    # Final validation
    if not is_valid_sidon_with_coverage(best_S):
        # Emergency: rebuild
        S = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            if not need:
                break
            for x in range(S[-1] + 1, 1000000):
                ok = True
                for s in S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    for s in S[:-1]:
                        used.add(x - s)
                    need -= used
                    break
        best_S = S
    
    return best_S
