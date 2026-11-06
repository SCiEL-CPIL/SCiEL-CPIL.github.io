import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin
    
    # We need a Sidon set S containing 0, with all positive differences distinct,
    # and every d in 1..30 occurring as a difference.
    # Minimize the largest element.
    
    # A known approach: use a greedy/backtracking search with bitmasks for differences.
    # Since we need diffs 1..30, we track which of these are covered.
    
    # Let's use a more efficient search with numpy for speed where possible.
    # Key insight: the set must be a Sidon set, so all differences are unique.
    
    # Start with S = [0], and build up.
    # For each new element x, it creates differences x - s for all s in S.
    # These must all be new (not in used_diffs) and we want to cover 1..30.
    
    # Let's try a DFS with good pruning.
    
    target_mask = (1 << 30) - 1  # bits 0..29 represent diffs 1..30
    
    def check_valid(S, used_diffs):
        """Verify S is a valid Sidon set covering all target diffs."""
        # Check Sidon: all positive differences must be distinct
        diffs = set()
        for i in range(len(S)):
            for j in range(i):
                d = S[i] - S[j]
                if d in diffs:
                    return False
                diffs.add(d)
        # Check all target diffs covered
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    # Search for minimum D
    # Lower bound: need at least 30 distinct differences, and Sidon constraint
    # For a Sidon set of size k in [0, D], max differences is k(k-1)/2
    # Need k(k-1)/2 >= 30, so k >= 9 (since 9*8/2=36)
    # With k=9, max diff = 8*9/2=36, so D must be at least 30
    # But Sidon constraint makes it larger.
    
    # Let's try a smarter approach: use the fact that we can fix the structure.
    # A perfect difference set of order n has n elements in Z_n^2 - 1... 
    # Actually, let's just do a well-optimized DFS.
    
    def search(D, time_budget=5.0):
        """Search for a valid set with max element <= D."""
        S = [0]
        used_diffs = set()
        covered_mask = 0
        
        # Candidate generation: for each position, try values that cover missing diffs
        def dfs(S, used_diffs, covered_mask, depth):
            nonlocal deadline
            
            if time.time() > deadline:
                return None
            
            # Check if all targets covered
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            cur_max = S[-1] if S else 0
            
            # Pruning: not enough room
            if cur_max >= D:
                return None
            
            # Pruning: even if we add all possible remaining elements,
            # can we cover all missing diffs?
            # Max elements we can add: D - cur_max (roughly)
            # But more precisely, each new element x covers at most len(S) new diffs
            # (x - s for s in S), plus diffs between new elements.
            # Simple bound: remaining room = D - cur_max
            # Each new element adds at most len(S) + (number of new elements before it) diffs
            # This is getting complex; use a simpler bound.
            
            # If we have k elements so far, and need m more missing diffs,
            # each new element can cover at most k + j - 1 new diffs (where j is its index among new)
            # Total potential = sum_{j=0}^{r-1} (k + j) = r*k + r*(r-1)/2 for r new elements
            # We need this >= num_missing
            # r*k + r*(r-1)/2 >= num_missing
            # Try small r
            k = len(S)
            r = 1
            while r <= D - cur_max:
                potential = r * k + r * (r - 1) // 2
                if potential >= num_missing:
                    break
                r += 1
            if r > D - cur_max:
                return None
            
            # Generate candidates: values x > cur_max such that x - s not in used_diffs for all s in S
            # Sort by how many missing diffs they cover
            cands = []
            for x in range(cur_max + 1, D + 1):
                new_covered = 0
                ok = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_covered |= (1 << (d - 1))
                if not ok:
                    continue
                new_missing_covered = new_covered & missing
                cands.append((bin(new_missing_covered).count('1'), x))
            
            # Sort by coverage (desc), then x (asc)
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates
            max_cands = 30
            if len(cands) > max_cands:
                cands = cands[:max_cands]
            
            for _, x in cands:
                # Build new state
                new_used = set(used_diffs)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                S.append(x)
                res = dfs(S, new_used, new_mask, depth + 1)
                if res is not None:
                    return res
                S.pop()
            
            return None
        
        return dfs(S, used_diffs, covered_mask, 0)
    
    # Try increasing D
    for D in range(31, 100):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
    
    # Fallback: greedy construction
    S = [0]
    used = set()
    target = set(range(1, 31))
    
    while target and time.time() < deadline - 2:
        best_x = None
        best_cover = -1
        cur_max = max(S)
        for x in range(cur_max + 1, cur_max + 200):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            # Try further out
            for x in range(cur_max + 200, cur_max + 1000):
                ok = True
                for s in S:
                    if x - s in used:
                        ok = False
                        break
                if ok:
                    best_x = x
                    break
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
    
    return S
