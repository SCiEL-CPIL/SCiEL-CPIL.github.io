"""
Strategy: Dual Hitting-Set (removal) + Add-Back repair.
Start with full set, remove elements that break the most 4-APs,
then add back removed elements where possible, with swap local search.
"""
import time
import random
import numpy as np

N = 512

def _precompute():
    """Precompute all 4-APs and per-element conflict lists."""
    aps_by_element = {x: [] for x in range(1, N + 1)}
    all_aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ap = (a, b, c, e)
                all_aps.append(ap)
                for x in ap:
                    aps_by_element[x].append(ap)
    return aps_by_element, all_aps

_AP_BY_ELEMENT, _ALL_APS = _precompute()

def _can_add(x, current_set):
    """Check if x can be added without creating a 4-AP."""
    for ap in _AP_BY_ELEMENT[x]:
        others = [y for y in ap if y != x]
        if all(y in current_set for y in others):
            return False
    return True

def _hitting_set_removal(deadline):
    """
    Start with full set. Repeatedly find the element in the most remaining 4-APs,
    remove it, and update. Continue until no 4-APs remain.
    """
    current = set(range(1, N + 1))
    removed = []
    
    # Count how many 4-APs each element is in (among those still "alive")
    # An AP is alive if all 4 elements are in current
    # We'll track alive APs
    
    # For efficiency, track per-element alive AP count
    alive_ap_count = {}
    for x in range(1, N + 1):
        alive_ap_count[x] = len(_AP_BY_ELEMENT[x])
    
    # Use a priority approach: repeatedly pick element with max alive AP count
    # But we need to update counts when we remove an element
    
    # Simpler: iterative removal
    # For each round, find the element that's in the most alive 4-APs
    # An AP is alive iff all 4 of its elements are in current
    
    # To speed up, maintain for each AP whether it's alive
    # But that's O(#APs) per removal... #APs is about 512*511/3 ≈ 87k, too slow for many removals
    
    # Alternative: just do it with the can_add check in reverse
    # Actually, let's think differently:
    # Greedy hitting set: while there exists a 4-AP in current, pick the element
    # in the most 4-APs (within current) and remove it.
    
    # For speed, limit iterations and use a simpler heuristic:
    # Remove elements in order of total AP count (precomputed), checking if it helps.
    
    # Actually, let's use a different approach:
    # Sort elements by their total number of 4-APs (descending).
    # Go through in that order, and for each element, check if removing it
    # would eliminate at least one 4-AP. If so, remove it.
    
    # Even simpler and faster: just use the standard greedy hitting set
    # with periodic rescans.
    
    # Let's use a priority queue approach with numpy for speed
    # Actually, let's just do iterative: at each step, scan all elements,
    # count alive APs for each, pick max, remove.
    # #APs total is ~87k. Each removal requires O(#APs) to check.
    # Max removals is bounded by... well, could be up to 512.
    # 512 * 87000 = 44M operations. In Python that might be slow.
    
    # Better: precompute for each AP its 4 elements. Maintain a set of alive APs.
    # When we remove x, all APs containing x become dead.
    
    alive_aps = set(range(len(_ALL_APS)))
    # Map: element -> set of AP indices containing it
    elem_to_aps = {x: set() for x in range(1, N + 1)}
    for i, ap in enumerate(_ALL_APS):
        for x in ap:
            elem_to_aps[x].add(i)
    
    # Count alive APs per element
    alive_count = {x: len(elem_to_aps[x]) for x in range(1, N + 1)}
    
    while alive_aps and time.time() < deadline:
        # Find element with max alive AP count
        best_x = max(range(1, N + 1), key=lambda x: alive_count[x])
        if alive_count[best_x] == 0:
            break
        
        # Remove best_x
        current.discard(best_x)
        removed.append(best_x)
        
        # Update: all APs containing best_x are now dead
        for ap_idx in elem_to_aps[best_x]:
            if ap_idx in alive_aps:
                alive_aps.discard(ap_idx)
                # Update alive_count for the other 3 elements
                ap = _ALL_APS[ap_idx]
                for x in ap:
                    if x != best_x:
                        alive_count[x] -= 1
    
    return current, removed

def _add_back_repair(current, removed, deadline):
    """
    Try to add back removed elements. Use local search with swaps.
    """
    current = set(current)
    removed = list(removed)
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Shuffle removed for variety
        random.shuffle(removed)
        improved = False
        
        for x in removed:
            if x in current:
                continue
            if _can_add(x, current):
                current.add(x)
                removed.remove(x)
                improved = True
                no_improve = 0
                break
        else:
            no_improve += 1
        
        if not improved and no_improve >= max_no_improve:
            break
        
        # Swap local search: try removing one element to add another
        if not improved and time.time() < deadline:
            # Try a few swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                # Pick a random element to remove from current
                y = random.choice(list(current))
                trial = current - {y}
                # Try to add back a few removed elements
                for x in removed[:10]:
                    if x == y:
                        continue
                    if _can_add(x, trial):
                        current = trial
                        current.add(x)
                        if y in removed:
                            removed.remove(y)
                        if x in removed:
                            removed.remove(x)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = N
    all_elements = list(range(1, n + 1))
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Hitting set removal + add-back
    current, removed = _hitting_set_removal(deadline)
    if time.time() < deadline:
        repaired = _add_back_repair(current, removed, deadline)
        best_set = repaired
        best_size = len(repaired)
    
    # Strategy 2: Multiple random greedy passes (as backup/comparison)
    def greedy_pass(order):
        selected = set()
        for x in order:
            if _can_add(x, selected):
                selected.add(x)
        return selected
    
    # Natural order
    if time.time() < deadline:
        s = greedy_pass(all_elements)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Reverse order
    if time.time() < deadline:
        s = greedy_pass(list(reversed(all_elements)))
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Random orders
    num_random = 15
    for i in range(num_random):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        s = greedy_pass(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Strategy 3: Local search from best result
    if time.time() < deadline:
        def local_search(cs, max_iters=3000):
            cs = set(cs)
            no_imp = 0
            max_no_imp = 300
            missing = [x for x in all_elements if x not in cs]
            
            while time.time() < deadline and no_imp < max_no_imp:
                if not missing:
                    break
                x = random.choice(missing)
                if _can_add(x, cs):
                    cs.add(x)
                    missing.remove(x)
                    no_imp = 0
                else:
                    no_imp += 1
                    # Swap
                    if no_imp % 5 == 0 and len(cs) > 5:
                        y = random.choice(list(cs))
                        trial = cs - {y}
                        if _can_add(x, trial):
                            cs = trial
                            cs.add(x)
                            if x in missing:
                                missing.remove(x)
                            no_imp = 0
            return cs
        
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Strategy 4: More hitting set restarts with random tie-breaking
    for restart in range(3):
        if time.time() > deadline:
            break
        # Randomized hitting set: when multiple elements have same max count, pick randomly
        current2 = set(range(1, n + 1))
        removed2 = []
        
        alive_aps2 = set(range(len(_ALL_APS)))
        elem_to_aps2 = {x: set() for x in range(1, n + 1)}
        for i, ap in enumerate(_ALL_APS):
            for x in ap:
                elem_to_aps2[x].add(i)
        alive_count2 = {x: len(elem_to_aps2[x]) for x in range(1, n + 1)}
        
        while alive_aps2 and time.time() < deadline:
            # Find max alive count
            max_count = max(alive_count2.values())
            if max_count == 0:
                break
            # Find all elements with this count
            candidates = [x for x in range(1, n + 1) if alive_count2[x] == max_count]
            best_x = random.choice(candidates)
            
            current2.discard(best_x)
            removed2.append(best_x)
            
            for ap_idx in elem_to_aps2[best_x]:
                if ap_idx in alive_aps2:
                    alive_aps2.discard(ap_idx)
                    ap = _ALL_APS[ap_idx]
                    for x in ap:
                        if x != best_x:
                            alive_count2[x] -= 1
        
        if time.time() < deadline:
            repaired2 = _add_back_repair(current2, removed2, deadline)
            if len(repaired2) > best_size:
                best_set = repaired2
                best_size = len(repaired2)
    
    # Final verification
    def has_4ap(s):
        s_set = set(s)
        s_list = sorted(s)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                d = b - a
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(best_set):
        # Fallback
        return list(range(1, 10))
    
    return sorted(best_set)
