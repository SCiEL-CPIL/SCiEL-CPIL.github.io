import time
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 110  # Safety margin
        
        # Lower bound estimate: k=9, D >= 72
        # Try D from 70 upwards
        for D in range(70, 150):
            if time.time() > deadline:
                break
            # Try to find a Sidon set of size >= 9 in [0, D] covering 1..30
            # We fix 0 in the set.
            # We need to select 8 more numbers from 1..D.
            
            # Heuristic: The set should be "spread out".
            # We can use a simple backtracking.
            
            def solve(D):
                # Returns list of elements or None
                # State: current set, used diffs, missing diffs
                # To be fast, we'll do a DFS.
                
                # We can prune if the maximum possible new differences from remaining slots < missing diffs count
                # Max new diffs if we add m elements to a set of size k is m*(k) + m*(m-1)/2
                # This is a weak bound.
                
                # Better: Just try.
                
                target_diffs = set(range(1, 31))
                used_diffs = set()
                current_set = [0]
                
                # We need to add elements.
                # Order: Try to add elements that cover many missing diffs.
                
                # Simple DFS
                def dfs(current_set, used_diffs, missing_diffs, last_elem, max_elem):
                    if not missing_diffs:
                        return current_set[:]
                    
                    if time.time() > deadline:
                        return None
                        
                    # Pruning:
                    # 1. If we can't fit more elements?
                    # We need at least enough elements to cover missing diffs.
                    # Current size k. Max diffs from adding 1 element is k.
                    # If len(missing) > k + (max_possible_elements - k) * ...
                    
                    # Simple bound:
                    # Max elements we can add is (max_elem - last_elem)
                    # But Sidon constraint limits this.
                    
                    # Let's just iterate candidates for next element.
                    # Candidates: last_elem + 1 to max_elem
                    # To improve, we can prioritize candidates that form many new diffs with existing elements.
                    
                    candidates = range(last_elem + 1, max_elem + 1)
                    
                    # Sort candidates by how many missing diffs they cover?
                    # This is expensive.
                    # Just iterate.
                    
                    for x in candidates:
                        new_diffs = []
                        valid = True
                        for y in current_set:
                            d = abs(x - y)
                            if d in used_diffs:
                                valid = False
                                break
                            if d <= 30:
                                new_diffs.append(d)
                            else:
                                # d > 30 is fine, it's just an extra diff, doesn't hurt Sidon-ness
                                # but it doesn't help cover 1..30
                                pass
                        
                        if not valid:
                            continue
                        
                        # Check if any new_diff is already in missing?
                        # If d in missing, it gets covered.
                        # If d not in missing, it's just a new diff.
                        
                        new_missing = missing_diffs - set(new_diffs)
                        
                        # Recurse
                        res = dfs(current_set + [x], used_diffs | set(new_diffs) | {d for d in new_diffs}, new_missing, x, max_elem)
                        if res is not None:
                            return res
                            
                    return None

                return dfs(current_set, used_diffs, target_diffs, 0, D)

            result = solve(D)
            if result is not None:
                return result
                
        # Fallback: return a known valid set if found in previous runs or generate a large one
        # Greedy fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in range(1, 1000):
                if x in a: continue
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
