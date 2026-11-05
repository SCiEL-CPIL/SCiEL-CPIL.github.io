"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    def is_valid_sidon(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_blocked(selected):
        """Return set of elements that cannot be added due to Sidon violation."""
        if not selected:
            return set()
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        
        blocked = set()
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check if adding x would violate Sidon
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict and 2 * x in sums:
                conflict = True
            if not conflict:
                # Need to check all pairs with x
                for i, a in enumerate(selected):
                    for j in range(i, len(selected)):
                        b = selected[j]
                        if a + b == x + a or a + b == x + b or a + b == 2 * x:
                            conflict = True
                            break
                    if conflict:
                        break
            if conflict:
                blocked.add(x)
        return blocked
    
    def is_maximal(selected):
        """Check if no element can be added."""
        if not selected:
            return False
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check if x can be added
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                # More careful check
                for a in selected:
                    for b in selected:
                        if a + b == x + a or a + b == x + b:
                            can_add = False
                            break
                    if not can_add:
                        break
            if can_add:
                return False
        return True
    
    def greedy_construct(random_order=None):
        """Greedy: pick elements that maximize blocked count."""
        if random_order is None:
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
        
        selected = []
        sums = set()
        
        for x in random_order:
            # Check if x can be added
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            
            # Try adding x
            new_selected = selected + [x]
            new_sums = sums.copy()
            new_sums.update(added_sums)
            
            # Count how many elements are now blocked
            blocked_count = 0
            for y in range(1, N + 1):
                if y in new_selected:
                    continue
                # Check if y is blocked
                y_blocked = False
                for a in new_selected:
                    if x + y in new_sums or y + a in new_sums:
                        pass
                # Simpler: check if adding y would violate
                conflict = False
                for a in new_selected:
                    if y + a in new_sums:
                        conflict = True
                        break
                if not conflict and 2 * y in new_sums:
                    conflict = True
                if conflict:
                    blocked_count += 1
            
            # Only add if it helps maximality progress
            selected.append(x)
            sums = new_sums
        
        return selected
    
    def smart_greedy():
        """Greedy that prioritizes elements blocking many others."""
        selected = []
        sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_score = -1
            
            for x in list(remaining):
                # Check if x can be added
                added_sums = [x + a for a in selected] + [2 * x]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if sums.intersection(added_sums):
                    continue
                
                # Score: how many elements would be blocked
                new_sums = sums.copy()
                new_sums.update(added_sums)
                
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y is blocked by new set
                    conflict = False
                    for a in selected:
                        if y + a in new_sums:
                            conflict = True
                            break
                    if not conflict and 2 * y in new_sums:
                        conflict = True
                    if conflict:
                        blocked += 1
                blocked += 1  # x itself is "used"
                
                if blocked > best_score:
                    best_score = blocked
                    best_x = x
            
            if best_x is None:
                break
            
            added_sums = [best_x + a for a in selected] + [2 * best_x]
            selected.append(best_x)
            sums.update(added_sums)
            remaining.discard(best_x)
            
            if time.time() > deadline:
                break
        
        return selected
    
    def check_maximal_and_valid(selected):
        """Verify Sidon property and maximality."""
        # Check Sidon
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        
        # Check maximality
        for x in range(1, N + 1):
            if x in selected:
                continue
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                for i, a in enumerate(selected):
                    for j in range(i, len(selected)):
                        b = selected[j]
                        if a + b == x + a or a + b == x + b:
                            can_add = False
                            break
                    if not can_add:
                        break
            if can_add:
                return False
        
        return True
    
    # Try multiple strategies
    best_result = None
    best_size = float('inf')
    
    # First, try the original ascending greedy as baseline
    selected_orig, sums_orig = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected_orig] + [2 * x]
        if len(set(added)) == len(added) and not sums_orig.intersection(added):
            selected_orig.append(x)
            sums_orig.update(added)
    
    if check_maximal_and_valid(selected_orig):
        best_result = selected_orig
        best_size = len(selected_orig)
    
    # Try smart greedy with different random seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        result = smart_greedy()
        
        if not result:
            continue
        
        # Try to extend if not maximal
        if not check_maximal_and_valid(result):
            # Try to fix by adding more elements
            extended = result[:]
            sums_ext = set()
            for i, a in enumerate(extended):
                for b in extended[i:]:
                    sums_ext.add(a + b)
            
            for x in range(1, N + 1):
                if x in extended:
                    continue
                added = [x + a for a in extended] + [2 * x]
                if len(set(added)) == len(added) and not sums_ext.intersection(added):
                    extended.append(x)
                    sums_ext.update(added)
            
            result = extended
        
        if check_maximal_and_valid(result) and len(result) < best_size:
            best_result = result
            best_size = len(result)
    
    # If we found a good result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return original
    return selected_orig
# EVOLVE-BLOCK-END
