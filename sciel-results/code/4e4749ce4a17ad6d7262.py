import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 70.0  # Safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def check_add(x, selected, triple_sums):
        """Check if x can be added to selected without creating duplicate triple sums.
        Returns True if valid, False otherwise."""
        n = len(selected)
        # New triple sums involving x:
        # (x, x, x)
        # (x, x, a) for each a in selected
        # (x, a, b) for a <= b in selected
        # (a, b, c) already in triple_sums
        
        # Check (x,x,x)
        s3 = 3 * x
        if s3 in triple_sums:
            return False
        
        # Check (x,x,a) for each a
        for a in selected:
            s2a = 2 * x + a
            if s2a in triple_sums:
                return False
        
        # Check (x,a,b) for a <= b in selected
        for i in range(n):
            a = selected[i]
            for j in range(i, n):
                b = selected[j]
                s_ab = x + a + b
                if s_ab in triple_sums:
                    return False
        
        return True

    def add_to_set(x, selected, triple_sums):
        """Add x to selected and update triple_sums."""
        n = len(selected)
        triple_sums.add(3 * x)
        for a in selected:
            triple_sums.add(2 * x + a)
        for i in range(n):
            a = selected[i]
            for j in range(i, n):
                b = selected[j]
                triple_sums.add(x + a + b)
        selected.append(x)

    def remove_from_set(x, selected, triple_sums):
        """Remove x from selected and update triple_sums."""
        idx = selected.index(x)
        n = len(selected)
        triple_sums.discard(3 * x)
        for a in selected:
            if a != x:
                triple_sums.discard(2 * x + a)
        for i in range(n):
            a = selected[i]
            for j in range(i, n):
                b = selected[j]
                if a != x and b != x:
                    triple_sums.discard(x + a + b)
        selected.pop(idx)

    def build_greedy_ordered(start_from=1, max_size=MAX_SIZE):
        """Greedy: add numbers in natural order."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 25:
                break
            if check_add(x, selected, triple_sums):
                add_to_set(x, selected, triple_sums)
        
        return selected, triple_sums

    def build_greedy_spaced(max_size=MAX_SIZE):
        """Greedy with spaced candidates to reduce collisions early."""
        selected = []
        triple_sums = set()
        
        # Try different spacing patterns
        for base in [3, 5, 7, 11]:
            if time.time() > DEADLINE - 25:
                break
            selected = []
            triple_sums = set()
            x = base
            while x <= N and len(selected) < max_size:
                if check_add(x, selected, triple_sums):
                    add_to_set(x, selected, triple_sums)
                x += base
            if len(selected) > len(selected):
                pass  # track best
        
        # Then fill in gaps with natural order
        if len(selected) < max_size:
            for x in range(1, N + 1):
                if len(selected) >= max_size:
                    break
                if time.time() > DEADLINE - 25:
                    break
                if x not in selected and check_add(x, selected, triple_sums):
                    add_to_set(x, selected, triple_sums)
        
        return selected, triple_sums

    def local_improve(selected, triple_sums, max_size=MAX_SIZE):
        """Try to improve by swapping or adding elements."""
        if time.time() > DEADLINE - 15:
            return selected, triple_sums
        
        improved = True
        while improved and len(selected) < max_size:
            improved = False
            if time.time() > DEADLINE - 15:
                break
            
            # Try adding any missing element
            selected_set = set(selected)
            for x in range(1, N + 1):
                if time.time() > DEADLINE - 15:
                    break
                if x in selected_set:
                    continue
                if check_add(x, selected, triple_sums):
                    add_to_set(x, selected, triple_sums)
                    improved = True
                    break
        
        # Try swaps: remove one, add two
        if len(selected) < max_size and time.time() < DEADLINE - 10:
            for i in range(len(selected)):
                if time.time() > DEADLINE - 10:
                    break
                x_remove = selected[i]
                # Save state
                sel_copy = selected.copy()
                sums_copy = triple_sums.copy()
                
                remove_from_set(x_remove, selected, triple_sums)
                
                # Try to add two new elements
                added = 0
                selected_set = set(selected)
                for x_add in range(1, N + 1):
                    if time.time() > DEADLINE - 10:
                        break
                    if x_add in selected_set or x_add == x_remove:
                        continue
                    if check_add(x_add, selected, triple_sums):
                        add_to_set(x_add, selected, triple_sums)
                        added += 1
                        if added == 2:
                            break
                
                if added == 2:
                    improved = True
                else:
                    # Restore
                    selected[:] = sel_copy
                    triple_sums.clear()
                    triple_sums.update(sums_copy)
        
        return selected, triple_sums

    # Strategy 1: Natural order greedy
    selected1, sums1 = build_greedy_ordered()
    
    # Strategy 2: Spaced greedy
    if time.time() < DEADLINE - 30:
        selected2, sums2 = build_greedy_spaced()
        if len(selected2) > len(selected1):
            selected1, sums1 = selected2, sums2
    
    # Local improvement
    if time.time() < DEADLINE - 15:
        selected1, sums1 = local_improve(selected1, sums1)
    
    # Final: try to add more if possible
    if time.time() < DEADLINE - 5:
        selected_set = set(selected1)
        for x in range(1, N + 1):
            if len(selected1) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 3:
                break
            if x in selected_set:
                continue
            if check_add(x, selected1, sums1):
                add_to_set(x, selected1, sums1)
    
    return sorted(selected1)
