import time
import random
import sys

N = 256
DEADLINE = time.time() + 55

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Verify all unordered sums a+b (a<=b) are distinct."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _get_unblocked(selected, sums):
    """
    Returns the list of elements x in [1, N] \ selected that can be added 
    without violating the Sidon property.
    """
    sset = set(selected)
    unblocked = []
    for x in range(1, N + 1):
        if x in sset:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _repair_maximality(selected, sums):
    """
    Greedily adds elements to 'selected' to make it maximal.
    Returns the new selected list, new sums, and the number of elements added.
    """
    added_count = 0
    cur_sel = list(selected)
    cur_sums = set(sums)
    
    # Loop until maximal
    while True:
        unblocked = _get_unblocked(cur_sel, cur_sums)
        if not unblocked:
            break
        
        # Find the element in unblocked that "blocks" the most other unblocked elements
        # Adding y blocks x if y+x is in sums or 2y is in sums (relative to the new state)
        # Actually, adding y to S creates new sums: 2y and y+a for a in S.
        # These new sums might block other candidates z in unblocked.
        
        best_y = None
        max_blocks = -1
        
        # Optimization: If len(unblocked) is small, just pick the first one? 
        # No, greedy choice matters for minimizing added count.
        # However, checking all pairs is expensive. 
        # Let's approximate: pick y that creates the most new sums that are in the "potential" sums of others?
        # Simpler heuristic: Pick y that is "central" or just random if time is tight.
        # Let's try a few candidates and pick the best.
        
        candidates = unblocked
        # If list is huge, sample it? N=256, max unblocked is ~256. 
        # Checking 256 * 256 interactions is 65k ops. Acceptable.
        
        for y in candidates:
            # Simulate adding y
            # New sums: 2y, and y+a for a in cur_sel
            # These new sums block z if:
            # 1. 2z == 2y => z==y (already handled)
            # 2. 2z == y+a => z = (y+a)/2
            # 3. z+a' == 2y => z = 2y - a'
            # 4. z+a' == y+a => z = y+a-a'
            
            # This is getting complex. Let's just simulate the addition of y and count how many 
            # elements from 'unblocked' become blocked.
            
            # To speed up, we can just pick the first one if we are in a tight loop, 
            # but "minimize added" is key.
            
            # Let's use a simpler heuristic: 
            # The element y that is involved in the most "potential conflicts" is good.
            # Or just pick the smallest?
            
            # Let's try: Pick y that minimizes the number of *remaining* unblocked elements after adding y.
            # This requires checking each z in unblocked against y.
            
            new_sums = set(cur_sums)
            new_sums.add(2 * y)
            for a in cur_sel:
                new_sums.add(y + a)
            
            # Check how many z in unblocked (z != y) are now blocked by new_sums
            # z is blocked if 2z in new_sums OR (z+a in new_sums for some a in cur_sel U {y})
            # Note: cur_sel U {y} is the new set.
            
            blocked_count = 0
            for z in unblocked:
                if z == y:
                    continue
                if 2 * z in new_sums:
                    blocked_count += 1
                    continue
                # Check z+a for a in cur_sel U {y}
                # We can check against cur_sums first? No, new sums are added.
                # If z+a was in cur_sums, it was already blocked (so z wouldn't be in unblocked).
                # So we only need to check if z+a is in the *newly added* sums.
                # Newly added sums: 2y and y+a for a in cur_sel.
                
                if z + y in cur_sums: # z + y == y + a => z == a (impossible as z not in cur_sel)
                    # Wait, z+y is a sum. If z+y is in cur_sums, then z+y = a+b.
                    # Adding y creates sum y+z? No, y is added, z is not.
                    # The sum y+z is NOT created until z is added.
                    # The sums created by adding y are 2y and y+a.
                    # These sums block z if:
                    # 2z == 2y or 2z == y+a
                    # OR z+a == 2y or z+a == y+b (for some b in cur_sel)
                    
                    # Let's just check the new sums directly.
                    pass
                
                # Efficient check:
                # z is blocked by new sums if:
                # 1. 2z == 2y (z=y, skip)
                # 2. 2z == y+a  => z = (y+a)/2
                # 3. z+a == 2y  => z = 2y - a
                # 4. z+a == y+b => z = y+b-a
                
                # Instead of deriving, let's just check if z is blocked by the new set.
                # Since N is small, we can just check the condition directly for z against new_sums and new_sel.
                # But we want to avoid O(N^2) per y.
                
                # Let's just use the direct check for z against the *new* sums set.
                # It's O(K) per z. Total O(U * K).
                
                is_blocked = False
                if 2 * z in new_sums:
                    is_blocked = True
                else:
                    # Check z+a for a in new_sel
                    # We know z+a was NOT in cur_sums (since z was unblocked).
                    # So z+a must be in the *newly added* sums.
                    # Newly added sums are 2y and y+a for a in cur_sel.
                    # So check if z+a == 2y or z+a == y+b for some b in cur_sel.
                    # z+a == 2y => a = 2y - z
                    if (2 * y - z) in cur_sums: # No, a must be in cur_sel.
                         pass
                    # This is getting messy. Let's just do the full check for z against new_sums.
                    # new_sums is a set.
                    # For a in new_sel: if z+a in new_sums: blocked.
                    # This is O(K).
                    for a in cur_sel:
                        if z + a in new_sums:
                            is_blocked = True
                            break
                    if not is_blocked and z + y in new_sums:
                        is_blocked = True
                        # Wait, z+y is a sum involving y.
                        # If z+y is in new_sums, it means z+y = 2y or z+y = y+a.
                        # z+y=2y => z=y (skip)
                        # z+y=y+a => z=a (impossible)
                        # So z+y is NOT in new_sums unless z is in cur_sel (which it isn't).
                        # So we don't need to check z+y.
                        
                if is_blocked:
                    blocked_count += 1
            
            if blocked_count > max_blocks:
                max_blocks = blocked_count
                best_y = y
        
        # If best_y is None, something went wrong, but unblocked was not empty.
        if best_y is None:
            best_y = unblocked[0]
            
        # Add best_y to cur_sel
        y = best_y
        cur_sel.append(y)
        cur_sums.add(2 * y)
        for a in cur_sel[:-1]:
            cur_sums.add(y + a)
        added_count += 1
        
        # Safety check to prevent infinite loops (should not happen)
        if added_count > len(selected) + 5:
            break

    return cur_sel, cur_sums, added_count

def _greedy_randomize(perm):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in perm:
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if conflict:
            continue
            
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
            
    return selected, sums

def construct():
    rng = random.Random(42)
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Phase 1: Find a good initial maximal Sidon set
    # We use random greedy to get a maximal set (greedy always produces a maximal set if it processes all N elements? 
    # Yes, if we process all elements and add if possible, the result is maximal because 
    # any element not added was blocked at the time of its turn. 
    # However, adding later elements might have unblocked it? No, adding elements only adds sums, 
    # so it can only block MORE elements, not unblock. 
    # So a simple greedy over all N elements yields a maximal Sidon set.)
    
    # Run several iterations to find the smallest initial maximal set
    for _ in range(100):
        if time.time() > DEADLINE - 5:
            break
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        selected, sums = _greedy_randomize(perm)
        
        # It is automatically maximal
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)
            
    if best_result is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums: continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict: continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected
        best_sums = set(sums)
        best_size = len(selected)

    # Phase 2: Simulated Annealing / Local Search with "Remove and Repair"
    # State: (selected, sums) which is a Maximal Sidon Set.
    # Move: Remove a random element, then repair maximality.
    # Cost: Size of the set.
    
    current_sel = list(best_result)
    current_sums = set(best_sums)
    current_size = len(current_sel)
    
    # We track the best found
    global_best_size = best_size
    global_best_sel = list(best_result)
    global_best_sums = set(best_sums)
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < DEADLINE - 2:
        if T < T_min:
            # Reheat
            T = T_min * 10
            # Random restart? Or just continue?
            # Let's try a random restart if we are stuck
            if rng.random() < 0.1:
                 perm = list(range(1, N + 1))
                 rng.shuffle(perm)
                 current_sel, current_sums = _greedy_randomize(perm)
                 current_size = len(current_sel)
        
        # Pick a random element to remove
        if len(current_sel) > 1:
            idx = rng.randint(0, len(current_sel) - 1)
            removed_el = current_sel[idx]
            
            # Remove it
            new_sel = current_sel[:idx] + current_sel[idx+1:]
            
            # Rebuild sums for new_sel
            # Optimization: We can remove sums involving removed_el
            removed_sums = {2 * removed_el}
            for a in new_sel:
                removed_sums.add(removed_el + a)
            new_sums = current_sums - removed_sums
            
            # Now repair maximality
            repaired_sel, repaired_sums, added_count = _repair_maximality(new_sel, new_sums)
            
            new_size = len(repaired_sel)
            
            # Calculate delta
            delta = new_size - current_size
            
            # Acceptance probability
            if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                # Accept
                current_sel = repaired_sel
                current_sums = repaired_sums
                current_size = new_size
                
                if current_size < global_best_size:
                    global_best_size = current_size
                    global_best_sel = list(current_sel)
                    global_best_sums = set(current_sums)
                    # Restart cooling if we found a new best?
                    # T = max(T, 5.0)
            
        T *= cooling
        
    return global_best_sel

# EVOLVE-BLOCK-END
