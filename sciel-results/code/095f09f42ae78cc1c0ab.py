import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def get_mask(sel):
        """Compute bitmask of all sums a+b for a,b in sel, a<=b."""
        m = 0
        for i in range(len(sel)):
            a = sel[i]
            m |= 1 << (2 * a)
            for j in range(i + 1, len(sel)):
                b = sel[j]
                m |= 1 << (a + b)
        return m

    def can_add(x, sel, sum_mask, sel_set):
        """Check if x can be added to sel (Sidon property maintained)."""
        if x in sel_set:
            return False
        if (sum_mask >> (2 * x)) & 1:
            return False
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                return False
        return True

    def add_element(x, sel, sel_set, sum_mask):
        """Add x to sel, update sel_set and sum_mask."""
        sel.append(x)
        sel_set.add(x)
        for a in sel[:-1]:
            sum_mask |= 1 << (x + a)
        sum_mask |= 1 << (2 * x)
        return sel, sel_set, sum_mask

    def is_maximal(sel, sum_mask, sel_set):
        """Check if sel is inclusion-maximal."""
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if can_add(y, sel, sum_mask, sel_set):
                return False
        return True

    def saturate(sel, sel_set, sum_mask):
        """Add all possible elements to make the set maximal."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if can_add(x, sel, sum_mask, sel_set):
                    sel, sel_set, sum_mask = add_element(x, sel, sel_set, sum_mask)
                    changed = True
        return sel, sel_set, sum_mask

    def greedy_build(seed):
        """Greedy construction with blocking heuristic."""
        rng = random.Random(seed)
        sel = []
        sel_set = set()
        sum_mask = 0
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
        while remaining and time.time() < deadline:
            best_x = -1
            best_score = -1
            
            # Sample candidates to speed up
            candidates = remaining if len(remaining) < 50 else rng.sample(remaining, 50)
            
            for x in candidates:
                if x in sel_set:
                    continue
                if not can_add(x, sel, sum_mask, sel_set):
                    continue
                
                # Score: how many remaining elements does x block?
                # An element y is blocked if it cannot be added after x.
                # This is expensive to compute exactly, so we approximate.
                # If we add x, we block y if (x+y) is in sum_mask OR 2y is in sum_mask (after adding x's sums).
                # Actually, simpler: just count how many elements in remaining are "close" to x in terms of sum conflicts.
                
                # Heuristic: Prefer x that creates many new sum conflicts with existing elements.
                # Each existing element a creates a new sum a+x.
                # This sum s = a+x conflicts with any y such that y+z = s for some z in sel.
                # This is complex. Let's use a simpler proxy:
                # Count how many elements in remaining would be blocked by the new sums.
                
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    # If y is addable now, check if it becomes non-addable after adding x.
                    # This requires simulating. Too slow.
                    
                    # Alternative heuristic: 
                    # Elements that are "isolated" are less likely to be in the final small set?
                    # No, we want to pick elements that "cover" the space.
                    
                    # Let's use the number of conflicts created.
                    # For each a in sel, the sum a+x is new.
                    # This sum forbids y if y = (a+x)/2 or y = a+x - a' for some a' in sel.
                    # Let's just count how many y in remaining have (y+x) in sum_mask OR (2y) in sum_mask.
                    # Wait, if (y+x) is already in sum_mask, y is already blocked.
                    # We want to know how many *newly* blocked elements.
                    
                    # This is getting complicated. Let's stick to a simpler randomized greedy with restarts.
                    pass
                
                # Simplified score: Randomly pick from valid candidates?
                # Or pick the one with the smallest value?
                # Or the one with the largest value?
                
                # Let's try: Pick the element that maximizes the number of "immediate" conflicts.
                # Conflict if (x + a) is in sum_mask for any a in sel? No, that would mean x is not addable.
                # x is addable, so (x+a) is NOT in sum_mask.
                
                # Let's just pick randomly from valid candidates in the sample.
                if best_x == -1:
                    best_x = x
                    best_score = 0 # Arbitrary
                else:
                    # Prefer smaller values? Or larger?
                    # Smaller values tend to create smaller sums, which might conflict with other small sums.
                    # Let's try to prefer elements that are "central".
                    if x < best_x:
                        best_x = x
                        best_score = 0

            if best_x == -1:
                # Try full remaining if sample failed
                for x in remaining:
                    if x in sel_set:
                        continue
                    if can_add(x, sel, sum_mask, sel_set):
                        best_x = x
                        break
            
            if best_x == -1:
                break
            
            sel, sel_set, sum_mask = add_element(best_x, sel, sel_set, sum_mask)
            remaining.remove(best_x)
        
        # Saturate
        sel, sel_set, sum_mask = saturate(sel, sel_set, sum_mask)
        return sel

    def local_search_reduce(sel, time_budget=5.0):
        """Try to reduce size via remove-and-saturate moves."""
        start = time.time()
        current = list(sel)
        current_size = len(current)
        
        while time.time() < start + time_budget:
            improved = False
            current_set = set(current)
            current_mask = get_mask(current)
            
            # Try removing each element
            for i in range(len(current)):
                if time.time() > start + time_budget:
                    break
                
                x1 = current[i]
                rem_sel = [e for e in current if e != x1]
                rem_set = set(rem_sel)
                rem_mask = get_mask(rem_sel)
                
                # Saturate the reduced set
                new_sel, new_set, new_mask = saturate(rem_sel, rem_set, rem_mask)
                
                if len(new_sel) < current_size:
                    current = new_sel
                    current_size = len(current)
                    improved = True
                    break
            
            if not improved:
                # Perturbation: Remove 2, add 1
                if len(current) < 2:
                    break
                
                x_idx1 = random.randint(0, len(current) - 1)
                x_idx2 = random.randint(0, len(current) - 1)
                if x_idx1 == x_idx2:
                    continue
                    
                x1 = current[x_idx1]
                x2 = current[x_idx2]
                
                rem_sel = [e for e in current if e != x1 and e != x2]
                rem_set = set(rem_sel)
                rem_mask = get_mask(rem_sel)
                
                found = False
                # Try adding random elements to the reduced set
                for _ in range(10):
                    y = random.randint(1, N)
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_sel, rem_mask, rem_set):
                        continue
                    
                    new_sel = rem_sel + [y]
                    new_set = rem_set | {y}
                    new_mask = rem_mask
                    for a in rem_sel:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    
                    # Saturate
                    new_sel, new_set, new_mask = saturate(new_sel, new_set, new_mask)
                    
                    if len(new_sel) < current_size:
                        current = new_sel
                        current_size = len(current)
                        found = True
                        break
                
                if not found:
                    break
        
        return current

    best_set = None
    best_size = float('inf')
    
    # Phase 1: Generate candidates with greedy
    num_trials = 50
    for i in range(num_trials):
        if time.time() > deadline - 15:
            break
        sel = greedy_build(i)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    if best_set is None:
        best_set = greedy_build(0)
        best_size = len(best_set)

    # Phase 2: Local search reduction
    current = list(best_set)
    current_size = len(current)
    
    while time.time() < deadline - 3:
        improved = False
        current_set = set(current)
        current_mask = get_mask(current)
        
        # Try removing each element
        for i in range(len(current)):
            if time.time() > deadline - 3:
                break
            
            x1 = current[i]
            rem_sel = [e for e in current if e != x1]
            rem_set = set(rem_sel)
            rem_mask = get_mask(rem_sel)
            
            # Saturate the reduced set
            new_sel, new_set, new_mask = saturate(rem_sel, rem_set, rem_mask)
            
            if len(new_sel) < current_size:
                current = new_sel
                current_size = len(current)
                improved = True
                break
        
        if not improved:
            # Perturbation: Remove 2, add 1
            if len(current) < 2:
                break
            
            x_idx1 = random.randint(0, len(current) - 1)
            x_idx2 = random.randint(0, len(current) - 1)
            if x_idx1 == x_idx2:
                continue
                
            x1 = current[x_idx1]
            x2 = current[x_idx2]
            
            rem_sel = [e for e in current if e != x1 and e != x2]
            rem_set = set(rem_sel)
            rem_mask = get_mask(rem_sel)
            
            found = False
            # Try adding random elements to the reduced set
            for _ in range(15):
                y = random.randint(1, N)
                if y in rem_set:
                    continue
                if not can_add(y, rem_sel, rem_mask, rem_set):
                    continue
                
                new_sel = rem_sel + [y]
                new_set = rem_set | {y}
                new_mask = rem_mask
                for a in rem_sel:
                    new_mask |= 1 << (y + a)
                new_mask |= 1 << (2 * y)
                
                # Saturate
                new_sel, new_set, new_mask = saturate(new_sel, new_set, new_mask)
                
                if len(new_sel) < current_size:
                    current = new_sel
                    current_size = len(current)
                    found = True
                    break
            
            if not found:
                break

    return current
