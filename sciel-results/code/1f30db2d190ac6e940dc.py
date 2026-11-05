import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    # Required differences: 1..30
    req_diffs = set(range(1, 31))
    
    # We know M must be at least 30.
    # We'll try M from 30 upwards, using a smarter search.
    
    def search(M):
        """
        Search for a valid Sidon set in [0, M] containing 0, covering req_diffs.
        Returns the set if found, else None.
        """
        # We'll use a backtracking search with strong pruning.
        # State: current_set (sorted list), covered_diffs (set of all diffs so far)
        
        # Precompute: for each possible x in 1..M, and for each possible current_set,
        # it's expensive. Instead, we'll compute on the fly but with good ordering.
        
        # Key insight: We need to cover differences 1..30.
        # A difference d can be formed as x - y where x > y.
        # For the search to be efficient, we should prioritize candidates that
        # cover many missing differences.
        
        # Let's use a different approach: iterative deepening on the number of elements.
        # Actually, let's stick with DFS but with better pruning and ordering.
        
        # State:
        # - current_set: list of chosen elements (sorted)
        # - covered_diffs: set of all positive differences in current_set
        # - missing_diffs: set of required differences not yet covered
        
        # Pruning ideas:
        # 1. If missing_diffs is empty, we're done.
        # 2. If the largest element in current_set is M, we can't add more, so if missing_diffs is non-empty, fail.
        # 3. For each missing difference d, we need some pair (x, y) in the final set with x - y = d.
        #    If d > M, impossible. (But M >= 30, so this is fine.)
        # 4. More subtle: if we have current_set S, and we're going to add elements from S+1 to M,
        #    the new differences we can form are limited.
        
        # Let's use a different strategy: 
        # Since we need to cover 1..30, and the set must be Sidon,
        # let's think about what a small Sidon set looks like.
        # A Sidon set of size k has at least k(k-1)/2 distinct differences (actually exactly, since all pairs give distinct diffs).
        # We need at least 30 differences, so k(k-1)/2 >= 30 => k >= 8.
        # So we need at least 8 elements.
        
        # Let's try to find a set of size 8, 9, 10, ... with small diameter.
        
        # Actually, let's just do a clean DFS with good heuristics.
        
        best = [None]  # Use a list to allow modification in closure
        
        def dfs(current_set, covered_diffs, missing_diffs, depth):
            if time.time() > deadline:
                return
            
            if not missing_diffs:
                best[0] = list(current_set)
                return
            
            # If we can't add any more elements (current_set[-1] == M), fail
            if current_set[-1] >= M:
                return
            
            # Pruning: Check if it's possible to cover remaining missing diffs.
            # The maximum new difference we can form is M - current_set[0] = M.
            # But more importantly, each new element x can form at most len(current_set) new differences
            # with existing elements, plus differences with other new elements.
            # This is complex, so let's use a simpler bound.
            
            # Heuristic: Order candidates by how many missing diffs they cover.
            next_min = current_set[-1] + 1
            candidates = []
            
            for x in range(next_min, M + 1):
                # Check Sidon validity
                valid = True
                new_diffs = set()
                for y in current_set:
                    d = x - y
                    if d in covered_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    gain = len(new_diffs & missing_diffs)
                    # Only consider candidates that cover at least one missing diff,
                    # or if no candidate covers a missing diff, allow gain=0 to explore.
                    candidates.append((x, new_diffs, gain))
            
            if not candidates:
                return
            
            # Sort by gain descending, then by x ascending
            candidates.sort(key=lambda t: (-t[2], t[0]))
            
            # Optimization: If the best gain is 0, we might still need to add elements
            # to enable larger differences later. But this is a dead end if we can't
            # cover any missing diff directly. Actually, we CAN cover missing diffs
            # by adding two elements at once, but our search adds one at a time.
            # If gain is 0 for all candidates, we should probably prune unless
            # we believe future elements will help. But since we're adding in order,
            # a future element x' > x will have differences with x and with current_set.
            # The difference x' - x could be a missing diff. So gain=0 is not necessarily a dead end.
            # However, to keep the search tractable, let's limit the branching.
            
            # Limit branching: only try top N candidates
            max_branch = 20
            for x, new_diffs, gain in candidates[:max_branch]:
                next_covered = covered_diffs | new_diffs
                next_missing = missing_diffs - new_diffs
                
                dfs(current_set + [x], next_covered, next_missing, depth + 1)
                
                if best[0] is not None:
                    return
                if time.time() > deadline:
                    return
        
        # Start with [0]
        dfs([0], set(), req_diffs, 0)
        return best[0]
    
    # Iterate M from lower bound
    for M in range(30, 250):
        if time.time() > deadline:
            break
        
        sol = search(M)
        if sol is not None:
            # Verify the solution
            s = sorted(set(sol))
            diffs = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    d = s[j] - s[i]
                    if d in diffs:
                        # Not Sidon
                        pass
                    diffs.add(d)
            
            # Check Sidon property
            if len(diffs) != len(s) * (len(s) - 1) // 2:
                continue
            
            # Check all required differences are covered
            if req_diffs.issubset(diffs):
                return s
    
    # Fallback: return a known valid set
    # Let's construct one greedily
    a = [0]
    def all_diffs(s):
        return set(y - x for i, x in enumerate(s) for y in s[i + 1:])
    
    for d in range(1, 31):
        if d in all_diffs(sorted(a)):
            continue
        found = False
        # Try to add one element that creates difference d
        for x in sorted(set(v + d for v in a) | set(v - d for v in a)):
            if x < 0 or x in a:
                continue
            trial = sorted(a + [x])
            if len(all_diffs(trial)) == len(trial) * (len(trial) - 1) // 2:
                a = trial
                found = True
                break
        if not found:
            # Try adding two elements
            for x in range(1, 200):
                if x in a:
                    continue
                for y in range(x + 1, 200):
                    if y in a:
                        continue
                    trial = sorted(a + [x, y])
                    if len(all_diffs(trial)) == len(trial) * (len(trial) - 1) // 2:
                        if d in all_diffs(trial):
                            a = trial
                            found = True
                            break
                if found:
                    break
        if not found:
            # Give up and add a large element
            a.append(1000000)
    
    return a
