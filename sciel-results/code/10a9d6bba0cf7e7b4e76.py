import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145
    
    def get_diffs_set(S):
        """Get set of all positive differences."""
        S = sorted(S)
        return set(S[j] - S[i] for i in range(len(S)) for j in range(i+1, len(S)))
    
    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        n = len(S)
        diffs = get_diffs_set(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_1_30(S):
        """Check if all differences 1..30 are covered."""
        diffs = get_diffs_set(S)
        return all(d in diffs for d in range(1, 31))
    
    def valid_solution(S):
        return is_sidon(S) and covers_1_30(S) and len(S) <= 32 and 0 in S and max(S) <= 10**6
    
    # Pre-computed candidate: search for small Sidon sets covering 1..30
    # Use a systematic backtracking with strong pruning
    
    def search_min_max(max_size=12, max_val_limit=200):
        """Backtracking search for Sidon set covering 1..30 with min max element."""
        best = [None]
        best_max = [max_val_limit + 1]
        
        def backtrack(S, covered, used_diffs):
            if time.time() - start_time > deadline - 5:
                return
            
            if len(S) > max_size:
                return
            
            # Prune: if current max already >= best, skip
            if S and max(S) >= best_max[0]:
                return
            
            # Check if we've covered all 1..30
            if covered.issuperset(range(1, 31)):
                if max(S) < best_max[0]:
                    best_max[0] = max(S)
                    best[0] = S[:]
                return
            
            if len(S) == max_size:
                return
            
            missing = set(range(1, 31)) - covered
            
            # Generate candidates: elements that would cover at least one missing diff
            candidates = set()
            for d in missing:
                for x in S:
                    c1 = x + d
                    if c1 <= max_val_limit and c1 > max(S) if S else c1 >= 0:
                        candidates.add(c1)
                    c2 = x - d
                    if c2 >= 0 and c2 not in S:
                        candidates.add(c2)
            
            # Also try small values not yet tried
            current_max = max(S) if S else 0
            for v in range(1, min(current_max + 15, max_val_limit + 1)):
                if v not in S:
                    candidates.add(v)
            
            # Sort candidates by value to find small solutions first
            for c in sorted(candidates):
                if c in S:
                    continue
                
                # Check Sidon constraint
                valid = True
                for x in S:
                    diff = abs(c - x)
                    if diff == 0:
                        continue
                    if diff in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Add c
                new_S = S + [c]
                new_S.sort()
                
                # Compute new diffs
                new_diffs = set()
                for i in range(len(new_S)):
                    for j in range(i+1, len(new_S)):
                        new_diffs.add(new_S[j] - new_S[i])
                
                new_covered = new_diffs & set(range(1, 31))
                
                # Prune: must make progress
                if len(new_covered) <= len(covered):
                    continue
                
                # Additional prune: check if remaining capacity can cover missing
                remaining_needed = len(set(range(1, 31)) - new_covered)
                remaining_slots = max_size - len(new_S)
                # Each new element can cover at most (len(new_S)) new diffs
                # But we need at least remaining_needed new diffs
                if remaining_slots * len(new_S) < remaining_needed:
                    continue
                
                backtrack(new_S, new_covered, new_diffs)
        
        backtrack([0], set(), set())
        return best[0]
    
    # Run the search
    result = search_min_max(max_size=12, max_val_limit=150)
    
    if result and valid_solution(result):
        return result
    
    # Fallback: try larger max_val_limit
    if time.time() - start_time < deadline - 10:
        result2 = search_min_max(max_size=14, max_val_limit=300)
        if result2 and valid_solution(result2):
            return result2
    
    # Hard fallback: a known valid construction
    # {0, 1, 3, 7, 12, 20, 30, 42, 56} - check if this works
    # Let's just return a safe large solution
    # Use the trivial construction: {0, 1, 3, 6, 10, 15, 21, 28, 36}
    # Diffs from 0: 1,3,6,10,15,21,28,36
    # From 1: 2,5,9,14,20,27,35
    # From 3: 3,7,12,18,25,33
    # From 6: 4,9,15,22,30
    # From 10: 5,11,18,26
    # From 15: 6,13,21
    # From 21: 7,15
    # From 28: 8
    # All diffs: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,25,26,27,30,33,35,36
    # Missing: 16,17,19,23,24,28,29 - not all covered
    
    # Let me just do a quick greedy to get something valid
    S = [0]
    covered = set()
    for _ in range(50):
        if covered.issuperset(range(1, 31)):
            break
        missing = set(range(1, 31)) - covered
        best_c = None
        for d in missing:
            for x in S:
                for c in [x + d, x - d]:
                    if c < 0 or c in S:
                        continue
                    trial = sorted(S + [c])
                    if is_sidon(trial):
                        new_diffs = get_diffs_set(trial)
                        if len(new_diffs & set(range(1,31))) > len(covered):
                            if best_c is None or c < best_c:
                                best_c = c
        if best_c is not None:
            S = sorted(S + [best_c])
            covered = get_diffs_set(S)
        else:
            # Add a large element
            c = max(S) + 1
            while c in S or not is_sidon(sorted(S + [c])):
                c += 1
            S = sorted(S + [c])
            covered = get_diffs_set(S)
    
    return S
