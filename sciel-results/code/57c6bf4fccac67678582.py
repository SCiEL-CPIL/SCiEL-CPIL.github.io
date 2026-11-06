import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin
    
    def get_all_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        n = len(seq)
        if n == 0: return True
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_lookahead(seq, sums_set, max_depth=2):
        """
        Greedy with lookahead.
        Tries to add the next element that allows the most future extensions.
        """
        current = list(seq)
        current_sums = set(sums_set)
        
        while True:
            best_candidate = None
            best_score = -1
            best_new_sums = None
            
            # Determine range for next element
            if not current:
                lo, hi = 1, 256
            else:
                lo, hi = current[-1] + 1, 256
            
            candidates = []
            for x in range(lo, hi + 1):
                # Compute new sums
                new_sums = set(current_sums)
                total = x
                new_sums.add(total)
                for a in reversed(current):
                    total += a
                    new_sums.add(total)
                
                if new_sums.intersection(current_sums):
                    continue
                
                # Score this candidate
                # Heuristic: How many numbers in [x+1, 256] are "blocked"?
                # A number y is blocked if adding it creates a collision.
                # Approximation: Count how many y in [x+1, 256] are in new_sums? No.
                # Better: Check if we can add at least one more element after x.
                
                # Lookahead 1 step
                lookahead_score = 0
                for y in range(x + 1, min(x + 20, 257)): # Check next 20
                    total_y = y
                    # Sums ending in y
                    # y, y+x, y+x+a_last, ...
                    # We need to check if any of these are in new_sums
                    # And also check if y itself is in new_sums (it shouldn't be if x < y and sums are positive)
                    # Actually, y is a new element, so y is not in new_sums (which are sums of current + x).
                    # Wait, new_sums contains sums of blocks in current+x.
                    # The sum 'y' is not in new_sums unless y is a sum of a block in current+x.
                    # Since current+x is increasing and positive, the smallest sum is current[0].
                    # If y > current[-1], y is likely not a sum.
                    
                    # Check collision for y
                    valid_y = True
                    total_y = y
                    if total_y in new_sums:
                        valid_y = False
                    else:
                        for a in reversed([x] + current):
                            total_y += a
                            if total_y in new_sums:
                                valid_y = False
                                break
                    if valid_y:
                        lookahead_score += 1
                        
                # Prefer candidates with higher lookahead score
                # Tie-break: prefer smaller x (to leave more room) or larger x (to maximize value)?
                # Smaller x leaves more room for more elements.
                if lookahead_score > best_score or (lookahead_score == best_score and (best_candidate is None or x < best_candidate)):
                    best_score = lookahead_score
                    best_candidate = x
                    best_new_sums = new_sums
                    
            if best_candidate is None:
                break
                
            current.append(best_candidate)
            current_sums = best_new_sums
            
            # Time check
            if time.time() - start_time > time_limit:
                break
                
        return current, current_sums

    def try_extend(seq, sums_set):
        """Try to insert additional elements into the sequence"""
        current = list(seq)
        current_sums = set(sums_set)
        improved = True
        while improved:
            improved = False
            for pos in range(len(current) + 1):
                if pos == 0:
                    lo, hi = 1, current[0] - 1 if current else 256
                elif pos == len(current):
                    lo, hi = current[-1] + 1, 256
                else:
                    lo, hi = current[pos-1] + 1, current[pos] - 1
                
                for x in range(hi, lo - 1, -1):
                    # Check if inserting x at pos keeps all sums distinct
                    new_sums = set(current_sums)
                    total = x
                    new_sums.add(total)
                    # Sums ending at x
                    for a in reversed(current[:pos]):
                        total += a
                        new_sums.add(total)
                        
                    # Sums starting after x? No, those are unchanged.
                    # Sums crossing x?
                    # A sum crossing x is sum(current[i:pos]) + x + sum(current[pos:j])
                    # These are NOT in current_sums.
                    # We need to check if they collide with current_sums or each other.
                    
                    # This simple check is insufficient.
                    # We must recompute ALL sums for the new sequence.
                    
                    new_seq = current[:pos] + [x] + current[pos:]
                    all_new_sums = get_all_sums(new_seq)
                    
                    if len(all_new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current = new_seq
                        current_sums = all_new_sums
                        improved = True
                        break
                if improved:
                    break
        return current, current_sums

    def local_search(seq, sums_set, max_iters=500):
        """Local search with random restarts and moves"""
        current = list(seq)
        current_sums = set(sums_set)
        current_len = len(current)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Try to extend
            extended, ext_sums = try_extend(current, current_sums)
            if len(extended) > current_len:
                current = extended
                current_sums = ext_sums
                current_len = len(current)
                continue
                
            # Try a random move: Remove one element, try to re-insert it or a new one
            if len(current) < 2:
                break
                
            i = random.randint(0, len(current) - 1)
            val = current.pop(i)
            current_sums = get_all_sums(current)
            
            # Try to re-insert val at a different position
            inserted = False
            for pos in range(len(current) + 1):
                if pos == i:
                    continue
                if pos == 0:
                    if current and current[0] <= val:
                        continue
                elif pos == len(current):
                    if current and current[-1] >= val:
                        continue
                else:
                    if current[pos-1] >= val or current[pos] <= val:
                        continue
                
                new_seq = current[:pos] + [val] + current[pos:]
                if is_valid(new_seq):
                    current = new_seq
                    current_sums = get_all_sums(current)
                    inserted = True
                    break
            
            if not inserted:
                # Try to insert a new value instead of val
                # Find a valid position for val
                inserted = False
                for pos in range(len(current) + 1):
                    if pos == 0:
                        if current and current[0] <= val:
                            continue
                    elif pos == len(current):
                        if current and current[-1] >= val:
                            continue
                    else:
                        if current[pos-1] >= val or current[pos] <= val:
                            continue
                    new_seq = current[:pos] + [val] + current[pos:]
                    if is_valid(new_seq):
                        current = new_seq
                        current_sums = get_all_sums(current)
                        inserted = True
                        break
                
                if not inserted:
                    # Restore val
                    current.insert(i, val)
                    current_sums = get_all_sums(current)
                    # Try to find a replacement for val
                    # This is complex, skip for now
                    continue

    # 1. Generate initial candidates
    candidates = []
    
    # Greedy Ascending
    vals = []
    sums = set()
    for x in range(1, 257):
        new_sums = set(sums)
        total = x
        new_sums.add(total)
        for a in reversed(vals):
            total += a
            new_sums.add(total)
        if not new_sums.intersection(sums):
            vals.append(x)
            sums = new_sums
    candidates.append((vals, sums))
    
    # Greedy Descending
    vals = []
    sums = set()
    for x in range(256, 0, -1):
        new_sums = set(sums)
        total = x
        new_sums.add(total)
        for a in reversed(vals):
            total += a
            new_sums.add(total)
        if not new_sums.intersection(sums):
            vals.append(x)
            sums = new_sums
    vals.sort()
    candidates.append((vals, sums))
    
    # Greedy Lookahead
    vals, sums = greedy_lookahead([], set())
    candidates.append((vals, sums))
    
    # Randomized Greedy
    for _ in range(5):
        vals = []
        sums = set()
        available = list(range(1, 257))
        random.shuffle(available)
        for x in available:
            new_sums = set(sums)
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not new_sums.intersection(sums):
                vals.append(x)
                sums = new_sums
        vals.sort()
        candidates.append((vals, sums))
        
    best_seq, best_sums = max(candidates, key=lambda x: len(x[0]))
    best_len = len(best_seq)
    
    # 2. Local Search
    improved_seq, improved_sums = local_search(best_seq, best_sums)
    if len(improved_seq) > best_len:
        best_seq = improved_seq
        best_sums = improved_sums
        best_len = len(best_seq)
        
    # 3. Try to extend with DFS if time remains
    def dfs(seq, sums_set, last_val):
        nonlocal best_seq, best_sums, best_len, start_time, time_limit
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)
            best_sums = set(sums_set)
            
        if time.time() - start_time > time_limit:
            return
            
        # Pruning
        if current_len + (256 - last_val) <= best_len:
            return
            
        for x in range(256, last_val, -1):
            if (256 - x) % 10 == 0 and time.time() - start_time > time_limit:
                return
                
            new_sums = set(sums_set)
            total = x
            new_sums.add(total)
            for a in reversed(seq):
                total += a
                new_sums.add(total)
                
            if sums_set.intersection(new_sums):
                continue
                
            dfs(seq + [x], new_sums, x)
            
            if time.time() - start_time > time_limit:
                return

    if best_seq:
        dfs(best_seq, best_sums, best_seq[-1])
        
    return best_seq

