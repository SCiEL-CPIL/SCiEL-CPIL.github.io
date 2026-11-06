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

    def is_sidon(sel):
        """Check if sel is Sidon."""
        sums = set()
        for i in range(len(sel)):
            a = sel[i]
            if 2 * a in sums:
                return False
            sums.add(2 * a)
            for j in range(i + 1, len(sel)):
                b = sel[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(sel, sel_mask, sel_set):
        """Check if sel is maximal. sel_mask is bitmask of sums, sel_set is set of elements."""
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            # Check if y can be added
            # New sums: 2y, y+a for a in sel
            # Conflict if any new sum is in sel_mask
            if (sel_mask >> (2 * y)) & 1:
                continue
            conflict = False
            for a in sel:
                if (sel_mask >> (y + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                return False
        return True

    def greedy_maximal(seed):
        """Construct a maximal Sidon set using greedy max-block."""
        rng = random.Random(seed)
        sel = []
        sel_set = set()
        sum_mask = 0
        
        # Precompute potential blockers for each y against current sel
        # To speed up, we just iterate
        
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
        while remaining:
            best_x = -1
            best_score = -1
            
            # Sample candidates if list is large
            candidates = remaining if len(remaining) < 50 else rng.sample(remaining, 50)
            
            for x in candidates:
                if x in sel_set:
                    continue
                # Check if x can be added
                if (sum_mask >> (2 * x)) & 1:
                    continue
                can_add = True
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Score: How many remaining elements does x block?
                # y is blocked if (2y in new_mask) or (y+a in new_mask) for a in sel_new
                # new_mask = sum_mask | (1 << 2x) | (sum(1 << (x+a) for a in sel))
                # We approximate by counting how many y have (y+x) in sum_mask OR 2y in sum_mask (if x is large enough? No)
                # Simpler: Count y such that (y+x) in sum_mask OR 2y in sum_mask
                # Actually, if we add x, y is blocked if:
                # 1. 2y == 2x (y=x, already removed)
                # 2. 2y == x+a => y = (x+a)/2
                # 3. y+x == a+b (existing sum)
                
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    # y blocked if (y+x) in sum_mask OR 2y in sum_mask (and 2y != 2x)
                    # Note: 2y in sum_mask means y is already blocked by existing sel?
                    # If 2y is in sum_mask, y was already blocked before adding x.
                    # We only care about *new* blocks.
                    # But for greedy, maximizing total blocks is fine.
                    if (sum_mask >> (y + x)) & 1:
                        block_count += 1
                    elif (sum_mask >> (2 * y)) & 1 and 2 * y != 2 * x:
                        # This block existed before, but it helps keep y out.
                        # However, we want to know if x *adds* to blocking.
                        pass
                    
                # Better score: Number of y that become blocked *only* because of x?
                # Too complex. Use: Number of y such that (y+x) is in sum_mask.
                # This means y and x form a sum that conflicts with existing pair.
                
                if block_count > best_score:
                    best_score = block_count
                    best_x = x
            
            if best_x == -1:
                # No valid x found in sample, try full remaining
                for x in remaining:
                    if x in sel_set:
                        continue
                    if (sum_mask >> (2 * x)) & 1:
                        continue
                    can_add = True
                    for a in sel:
                        if (sum_mask >> (x + a)) & 1:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    best_x = x
                    break
            
            if best_x == -1:
                break
            
            # Add best_x
            sel.append(best_x)
            sel_set.add(best_x)
            remaining.remove(best_x)
            
            # Update sum_mask
            for a in sel[:-1]:
                sum_mask |= 1 << (best_x + a)
            sum_mask |= 1 << (2 * best_x)
            
        # Saturate: Add any remaining elements that can be added, in ascending order
        # This ensures maximality
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if (sum_mask >> (2 * x)) & 1:
                    continue
                can_add = True
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        can_add = False
                        break
                if not can_add:
                    continue
                # Add x
                sel.append(x)
                sel_set.add(x)
                for a in sel[:-1]:
                    sum_mask |= 1 << (x + a)
                sum_mask |= 1 << (2 * x)
                changed = True
        
        return sel

    best_set = None
    best_size = float('inf')
    
    # Phase 1: Generate candidates
    num_trials = 50
    for i in range(num_trials):
        if time.time() > deadline - 10:
            break
        sel = greedy_maximal(i)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    if best_set is None:
        # Fallback
        best_set = greedy_maximal(0)
        best_size = len(best_set)

    # Phase 2: Local Search with 2-Swap Reduction
    # Try to reduce size by removing 2 and adding 1
    current = list(best_set)
    current_size = len(current)
    
    while time.time() < deadline - 2:
        improved = False
        current_set = set(current)
        current_mask = get_mask(current)
        
        # Get list of elements for iteration
        elems = list(current)
        
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                x1 = elems[i]
                x2 = elems[j]
                
                # Remove x1, x2
                rem_sel = [e for e in current if e != x1 and e != x2]
                rem_set = set(rem_sel)
                
                # Check if rem_sel is Sidon (it is, subset of Sidon)
                # Compute mask of rem_sel
                rem_mask = 0
                for k in range(len(rem_sel)):
                    a = rem_sel[k]
                    rem_mask |= 1 << (2 * a)
                    for l in range(k + 1, len(rem_sel)):
                        b = rem_sel[l]
                        rem_mask |= 1 << (a + b)
                
                # Try to add one element y
                # We want y such that rem_sel + y is Sidon and Maximal
                # And preferably y is "good"
                
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    
                    # Check if y can be added to rem_sel
                    if (rem_mask >> (2 * y)) & 1:
                        continue
                    conflict = False
                    for a in rem_sel:
                        if (rem_mask >> (y + a)) & 1:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    # New set
                    new_sel = rem_sel + [y]
                    new_set = rem_set | {y}
                    
                    # Compute new mask
                    new_mask = rem_mask
                    for a in rem_sel:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    
                    # Check Maximal
                    if is_maximal(new_sel, new_mask, new_set):
                        # Accept
                        current = new_sel
                        current_set = new_set
                        current_mask = new_mask
                        current_size = len(current)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random restarts on current best or break
            # If no improvement found in a full pass, try a few random perturbations
            if time.time() > deadline - 5:
                break
            
            # Perturbation: Swap one element
            # Remove random x, add random y that keeps Sidon, then saturate
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            rem_sel = current[:x_idx] + current[x_idx + 1:]
            rem_set = set(rem_sel)
            rem_mask = get_mask(rem_sel)
            
            found = False
            for _ in range(20):
                y = random.randint(1, N)
                if y in rem_set:
                    continue
                if (rem_mask >> (2 * y)) & 1:
                    continue
                conflict = False
                for a in rem_sel:
                    if (rem_mask >> (y + a)) & 1:
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_sel = rem_sel + [y]
                new_set = rem_set | {y}
                new_mask = rem_mask
                for a in rem_sel:
                    new_mask |= 1 << (y + a)
                new_mask |= 1 << (2 * y)
                
                # Saturate
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for z in range(1, N + 1):
                        if z in new_set:
                            continue
                        if (new_mask >> (2 * z)) & 1:
                            continue
                        can_add = True
                        for a in new_sel:
                            if (new_mask >> (z + a)) & 1:
                                can_add = False
                                break
                        if not can_add:
                            continue
                        new_sel.append(z)
                        new_set.add(z)
                        for a in new_sel[:-1]:
                            new_mask |= 1 << (z + a)
                        new_mask |= 1 << (2 * z)
                        changed = True
                
                if len(new_sel) < current_size:
                    current = new_sel
                    current_size = len(current)
                    found = True
                    break
            
            if found:
                continue
            else:
                break

    return current
