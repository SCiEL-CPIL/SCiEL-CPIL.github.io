import time

_START = None

def _seed_greedy():
    # Keep the original fallback, maybe improve it slightly or leave as is
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _search(D, node_budget):
    full = tuple(range(1, 31))
    full_mask = 0
    for d in full:
        full_mask |= (1 << d)
        
    visited = set()
    nodes = [0]
    best = [None]
    start = time.time()

    def dfs(marks, used_mask):
        if best[0] is not None:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        if time.time() - start > 95:
            return
            
        # Check if all required differences are covered
        if (used_mask & full_mask) == full_mask:
            best[0] = list(marks)
            return

        # Heuristic: Try to add the smallest possible valid element
        # We iterate x from 1 to D
        # Optimization: If we haven't covered small differences, we might need small x's.
        # But simply trying x=1, 2, ... is a standard DFS.
        
        # Pruning: 
        # If there is a missing difference d, it must be formable by some pair.
        # A simple prune: if max(marks) is large and we are stuck, maybe backtrack.
        
        # To ensure we find a solution if it exists, we must try all valid x.
        # However, trying x in ascending order helps minimize the max element naturally 
        # because we stop as soon as we find a valid completion.
        
        # We can add a prune: if the number of remaining elements we can add is insufficient 
        # to cover the remaining differences.
        # Max remaining elements = D - max(marks) (roughly).
        # Each new element adds at most len(marks) new differences.
        # This is weak.
        
        # Let's stick to the bitmask check which is fast.
        
        current_max = max(marks) if marks else 0
        for x in range(1, D + 1):
            if x in marks:
                continue
                
            # Check Sidon property
            valid = True
            new_diffs_mask = 0
            for y in marks:
                d = abs(x - y)
                if d > D: # Should not happen if x,y <= D
                    valid = False
                    break
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
            
            if not valid:
                continue
                
            # Additional check: Do the new differences conflict with each other?
            # No, because they are |x-y_i| for distinct y_i, so they are distinct.
            
            nm = list(marks)
            nm.append(x)
            nm.sort()
            
            key = (tuple(nm), used_mask | new_diffs_mask)
            # Note: keying on mask might be too large for set, but tuple(nm) is unique enough 
            # if we assume deterministic path? No, different paths can lead to same set.
            # Just use tuple(nm).
            key = tuple(nm)
            if key in visited:
                continue
            visited.add(key)
            
            dfs(nm, used_mask | new_diffs_mask)
            if best[0] is not None:
                return

    dfs((0,), 0)
    return best[0]

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # We know a solution exists for small D.
    # Based on literature, a Sidon set covering 1..30 might have diameter around 30-50?
    # Actually, for B_2 sequences, the minimal diameter for N elements is ~N^2.
    # We need to cover 30 diffs.
    # Let's search from D=30 upwards.
    
    for D in range(30, 160):
        if time.time() - _START > 100:
            break
        res = _search(D, 500000) # Increased budget due to bitmask speed
        if res is not None:
            return res
            
    return _seed_greedy()
