import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_selected = []
    best_size = 0
    
    def greedy_order(order):
        """Run greedy with given order, return selected list."""
        selected = []
        # Use a set for reachable sums, but limit to 0..200
        reachable = {0}
        
        for x in order:
            if time.time() > deadline:
                break
            if 200 - x in reachable:
                continue
            selected.append(x)
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable.update(new_sums)
        
        return selected
    
    # Try descending order (original)
    order_desc = list(range(100, 0, -1))
    sel = greedy_order(order_desc)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = sel
    
    # Try ascending order
    if time.time() < deadline:
        order_asc = list(range(1, 101))
        sel = greedy_order(order_asc)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
    
    # Try several random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5.0:
        order = list(range(1, 101))
        random.shuffle(order)
        sel = greedy_order(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
        attempts += 1
        # Limit random attempts to avoid infinite loop
        if attempts >= 50:
            break
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 2.0 and best_selected:
        selected_set = set(best_selected)
        all_nums = set(range(1, 101))
        skipped = all_nums - selected_set
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try removing one element and adding two skipped ones
            for rem in list(best_selected):
                if time.time() > deadline - 1.0:
                    break
                new_set = best_selected.copy()
                new_set.remove(rem)
                new_set_set = set(new_set)
                
                # Compute reachable for new_set
                reachable = {0}
                for x in new_set:
                    new_sums = {s + x for s in reachable if s + x <= 200}
                    reachable.update(new_sums)
                
                # Try to add two skipped elements
                added = []
                for s1 in skipped:
                    if time.time() > deadline - 1.0:
                        break
                    if 200 - s1 in reachable:
                        continue
                    # Temporarily add s1
                    new_reachable1 = reachable | {s + s1 for s in reachable if s + s1 <= 200}
                    for s2 in skipped:
                        if s2 == s1:
                            continue
                        if 200 - s2 in new_reachable1:
                            continue
                        # Check if adding both is valid
                        candidate = new_set + [s1, s2]
                        if len(candidate) > len(best_selected):
                            # Verify no subset sums to 200
                            valid = True
                            reach = {0}
                            for x in candidate:
                                new_s = {s + x for s in reach if s + x <= 200}
                                reach.update(new_s)
                                if 200 in reach:
                                    valid = False
                                    break
                            if valid:
                                best_selected = candidate
                                best_size = len(candidate)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        # Also try adding single skipped elements
        if time.time() < deadline - 1.0:
            selected_set = set(best_selected)
            reachable = {0}
            for x in best_selected:
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable.update(new_sums)
            
            for s in list(skipped):
                if time.time() > deadline - 1.0:
                    break
                if 200 - s in reachable:
                    continue
                # Try adding s
                candidate = best_selected + [s]
                valid = True
                reach = {0}
                for x in candidate:
                    new_s = {t + x for t in reach if t + x <= 200}
                    reach.update(new_s)
                    if 200 in reach:
                        valid = False
                        break
                if valid:
                    best_selected = candidate
                    best_size = len(best_selected)
                    selected_set.add(s)
                    new_sums = {t + s for t in reachable if t + s <= 200}
                    reachable.update(new_sums)
    
    if not best_selected:
        best_selected = [100]
    
    return best_selected
