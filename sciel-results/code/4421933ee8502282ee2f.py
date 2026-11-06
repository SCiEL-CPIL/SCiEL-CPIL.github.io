import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    S = set(range(1, n + 1))
    
    def find_witness(x, available_values):
        """
        Find a subset of available_values that sums to V[x].
        Returns the list of values in the subset, or None.
        available_values is a list of (value, index) pairs.
        """
        target = V[x]
        # Prune: if sum < target, impossible
        if sum(v for v, _ in available_values) < target:
            return None
        
        # Sort by value descending to find large sums quickly?
        # Or ascending?
        # Let's use a set of reachable sums and backtrack.
        # Given small N, simple DFS is fine.
        
        vals = [v for v, _ in available_values]
        idxs = [i for _, i in available_values]
        
        # Sort by value descending to hit target faster?
        # Actually, sorting by value might help pruning.
        pairs = sorted(zip(vals, idxs), key=lambda p: -p[0])
        
        def dfs(start, current_sum, path):
            if current_sum == target:
                return path[:]
            if current_sum > target:
                return None
            if start >= len(pairs):
                return None
            
            # Pruning: if current_sum + sum(remaining) < target, return None
            # We can precompute suffix sums
            # But for now, simple check
            remaining_sum = sum(p[0] for p in pairs[start:])
            if current_sum + remaining_sum < target:
                return None
            
            # Option 1: include pairs[start]
            res = dfs(start + 1, current_sum + pairs[start][0], path + [pairs[start][1]])
            if res is not None:
                return res
            
            # Option 2: exclude pairs[start]
            res = dfs(start + 1, current_sum, path)
            if res is not None:
                return res
            
            return None
        
        return dfs(0, 0, [])

    while time.time() < deadline - 5.0:
        conflicts = []
        # Check conflicts for all x in S
        for x in S:
            # Candidates y > x
            avail = [(V[y], y) for y in S if y > x]
            if not avail:
                continue
            
            witness = find_witness(x, avail)
            if witness is not None:
                # witness is a list of indices y
                conflicts.append((x, witness))
        
        if not conflicts:
            break
        
        # Count occurrences
        # member_counts[y] = number of conflicts where y is in witness
        member_counts = {y: 0 for y in S}
        target_counts = {x: 0 for x in S}
        
        for x, witness in conflicts:
            target_counts[x] += 1
            for y in witness:
                member_counts[y] += 1
        
        # Choose element to remove
        # Score = count
        # Prefer removing members (y) over targets (x) if counts are similar?
        # Removing a member y fixes all conflicts involving y.
        # Removing a target x fixes only conflicts for x.
        # So members are generally better to remove.
        
        best_elem = None
        best_score = -1
        
        # Check members
        for y in S:
            score = member_counts[y]
            # Tie-breaker: larger y is better to remove (weaker)
            if score > best_score or (score == best_score and best_elem is not None and y > best_elem):
                if score > 0: # Only consider if involved in conflict
                    best_elem = y
                    best_score = score
        
        # Check targets
        for x in S:
            score = target_counts[x]
            if score > best_score or (score == best_score and best_elem is not None and x > best_elem):
                if score > 0:
                    best_elem = x
                    best_score = score
        
        if best_elem is None:
            break
            
        S.remove(best_elem)

    return sorted(S)
