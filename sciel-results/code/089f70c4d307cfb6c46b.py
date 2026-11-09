import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s buffer for safety
    
    # 1. Precomputation
    # Find all 4-APs
    ap4s = []
    ap4_map = {i: [] for i in range(1, n+1)} # element -> list of ap4 indices
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                idx = len(ap4s)
                ap4s.append((a, b, c, e))
                for x in (a, b, c, e):
                    ap4_map[x].append(idx)
    
    # Find all 3-APs
    ap3s = []
    ap3_map = {i: [] for i in range(1, n+1)} # element -> list of ap3 indices
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n:
                idx = len(ap3s)
                ap3s.append((a, b, c))
                for x in (a, b, c):
                    ap3_map[x].append(idx)

    # 2. Helper functions
    def count_3aps(in_set):
        count = 0
        for a, b, c in ap3s:
            if in_set[a] and in_set[b] and in_set[c]:
                count += 1
        return count

    def try_greedy_add(in_set, ap4_count, score):
        """Try to add any element that increases score and is valid. Returns (new_in_set, new_ap4_count, new_score, changed)"""
        added_any = False
        # Iterate in random order to break ties and explore different local optima
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if in_set[x]:
                continue
            
            # Check 4-AP validity
            valid = True
            for idx in ap4_map[x]:
                if ap4_count[idx] == 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate 3-AP gain
            gain = 0
            for idx in ap3_map[x]:
                a, b, c = ap3s[idx]
                # x is one of a, b, c. Check the other two.
                # Since we are adding x, we just need the other two to be in set.
                # We can optimize this by precomputing pairs, but for n=256 it's fine.
                # Let's just check the tuple.
                # To avoid repeated checks, we can store for each x, the list of (other1, other2)
                # But let's stick to simple check first.
                others = [y for y in (a, b, c) if y != x]
                if in_set[others[0]] and in_set[others[1]]:
                    gain += 1
            
            if gain > 0:
                # Perform add
                in_set[x] = True
                for idx in ap4_map[x]:
                    ap4_count[idx] += 1
                score += gain
                added_any = True
        
        return in_set, ap4_count, score, added_any

    def try_remove_and_readd(in_set, ap4_count, score):
        """Try removing one element to potentially allow adding more."""
        # Pick a random element to remove
        # We should probably remove an element that is 'costly' in terms of blocking others
        # or just random.
        candidates = [x for x in range(1, n+1) if in_set[x]]
        if not candidates:
            return in_set, ap4_count, score, False
        
        random.shuffle(candidates)
        
        best_gain = 0
        best_x = None
        best_new_state = None
        
        # We can't try all removals as it's expensive. Try a few.
        for x in candidates[:10]: # Try first 10 random candidates
            # Simulate removal
            old_score = score
            # Remove x
            in_set[x] = False
            for idx in ap4_map[x]:
                ap4_count[idx] -= 1
            
            # Calculate loss from removing x
            loss = 0
            for idx in ap3_map[x]:
                a, b, c = ap3s[idx]
                others = [y for y in (a, b, c) if y != x]
                if in_set[others[0]] and in_set[others[1]]:
                    loss += 1
            
            current_score = old_score - loss
            
            # Try greedy add
            new_in_set, new_ap4_count, new_score, _ = try_greedy_add(in_set, ap4_count, current_score)
            
            net_gain = new_score - old_score
            
            if net_gain > best_gain:
                best_gain = net_gain
                best_x = x
                best_new_state = (new_in_set, new_ap4_count, new_score)
            
            # Restore state for next candidate
            in_set[x] = True
            for idx in ap4_map[x]:
                ap4_count[idx] += 1
            # Note: try_greedy_add modified in_set and ap4_count. 
            # We need to revert those changes too.
            # This is getting complex. Let's simplify: 
            # Instead of simulating, we just do one move at a time.
            
        # Simplified approach: 
        # 1. Try adding. If stuck, remove a random element, then try adding again.
        # This is a valid hill-climbing step with restarts.
        
        if best_gain > 0 and best_new_state:
             return best_new_state[0], best_new_state[1], best_new_state[2], True
             
        return in_set, ap4_count, score, False

    # 3. Initialization
    # Start with greedy construction
    in_set = [False] * (n + 1)
    ap4_count = [0] * len(ap4s)
    score = 0
    
    for x in range(1, n + 1):
        # Check if adding x creates a 4-AP
        valid = True
        for idx in ap4_map[x]:
            if ap4_count[idx] == 3:
                valid = False
                break
        if valid:
            in_set[x] = True
            for idx in ap4_map[x]:
                ap4_count[idx] += 1
            
            # Update score
            gain = 0
            for idx in ap3_map[x]:
                a, b, c = ap3s[idx]
                others = [y for y in (a, b, c) if y != x]
                if in_set[others[0]] and in_set[others[1]]:
                    gain += 1
            score += gain

    # 4. Optimization Loop
    best_score = score
    best_in_set = in_set[:]
    
    while time.time() < deadline:
        # Phase 1: Greedy Add
        in_set, ap4_count, score, changed = try_greedy_add(in_set, ap4_count, score)
        if changed:
            if score > best_score:
                best_score = score
                best_in_set = in_set[:]
        
        # Phase 2: If no change, try a removal step to escape local optimum
        if not changed:
            # Remove a random element
            candidates = [x for x in range(1, n+1) if in_set[x]]
            if candidates:
                x_to_remove = random.choice(candidates)
                # Remove it
                in_set[x_to_remove] = False
                for idx in ap4_map[x_to_remove]:
                    ap4_count[idx] -= 1
                
                # Recalculate score (expensive but n is small)
                # Or subtract loss
                loss = 0
                for idx in ap3_map[x_to_remove]:
                    a, b, c = ap3s[idx]
                    others = [y for y in (a, b, c) if y != x_to_remove]
                    if in_set[others[0]] and in_set[others[1]]:
                        loss += 1
                score -= loss
                
                # Now try to add back
                in_set, ap4_count, score, _ = try_greedy_add(in_set, ap4_count, score)
                
                if score > best_score:
                    best_score = score
                    best_in_set = in_set[:]
            else:
                # No elements to remove, break
                break

    # Return the best set found
    result = [i for i in range(1, n+1) if best_in_set[i]]
    return sorted(result)

construct()
