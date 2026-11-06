import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def can_add(x, selected, existing_sums):
        """Check if x can be added to the set without creating duplicate triple sums."""
        if 3 * x in existing_sums:
            return False
        for y in selected:
            if 2 * x + y in existing_sums:
                return False
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                if x + y + z in existing_sums:
                    return False
        return True
    
    def add_element(x, selected, existing_sums):
        """Add x to selected and update existing_sums."""
        selected.append(x)
        existing_sums.add(3 * x)
        for y in selected[:-1]:
            existing_sums.add(2 * x + y)
        k_old = len(selected) - 1
        for i in range(k_old):
            y = selected[i]
            for j in range(i, k_old):
                z = selected[j]
                existing_sums.add(x + y + z)
    
    def rebuild_sums(selected):
        """Rebuild existing_sums from scratch."""
        existing_sums = set()
        n = len(selected)
        for i in range(n):
            x = selected[i]
            existing_sums.add(3 * x)
            for j in range(i):
                y = selected[j]
                existing_sums.add(2 * x + y)
                for k in range(j):
                    z = selected[k]
                    existing_sums.add(x + y + z)
        return existing_sums
    
    def greedy_from_order(order):
        """Greedy: iterate through order, add if valid."""
        selected = []
        existing_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected, existing_sums):
                add_element(x, selected, existing_sums)
        return selected
    
    def local_search(selected, max_iters=30):
        """Try to improve the set by adding new elements or swapping."""
        selected = sorted(list(selected))
        existing_sums = rebuild_sums(selected)
        selected_set = set(selected)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3.0:
            improved = False
            iters += 1
            
            # Try to add a new element (scan from 1 upward)
            for x in range(1, 4097):
                if x in selected_set:
                    continue
                if can_add(x, selected, existing_sums):
                    add_element(x, selected, existing_sums)
                    selected_set.add(x)
                    improved = True
                    break
            
            # Try to swap: replace one element with a different one
            if not improved:
                for i in range(len(selected)):
                    if time.time() > deadline - 3.0:
                        break
                    old_x = selected[i]
                    # Try a few candidate replacements
                    candidates = []
                    # Try values near old_x
                    for delta in range(1, 50):
                        for cand in (old_x + delta, old_x - delta):
                            if 1 <= cand <= 4096 and cand not in selected_set:
                                candidates.append(cand)
                    random.shuffle(candidates)
                    
                    for new_x in candidates[:20]:
                        # Remove old_x temporarily
                        remaining = [v for v in selected if v != old_x]
                        remaining_sums = rebuild_sums(remaining)
                        
                        if can_add(new_x, remaining, remaining_sums):
                            # Do the swap
                            selected[i] = new_x
                            selected_set.remove(old_x)
                            selected_set.add(new_x)
                            existing_sums = rebuild_sums(selected)
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return sorted(selected)
    
    # 1. Standard ascending greedy
    best_set = greedy_from_order(range(1, 4097))
    best_len = len(best_set)
    
    # 2. Descending greedy
    if time.time() < deadline - 5.0:
        candidate = greedy_from_order(range(4096, 0, -1))
        if len(candidate) > best_len:
            best_set = candidate
            best_len = len(candidate)
    
    # 3. Multiple randomized greedy passes
    if time.time() < deadline - 5.0:
        for trial in range(15):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 4097))
            random.shuffle(order)
            candidate = greedy_from_order(order)
            if len(candidate) > best_len:
                best_set = candidate
                best_len = len(candidate)
    
    # 4. Randomized greedy with bias toward smaller numbers (partial shuffle)
    if time.time() < deadline - 5.0:
        for trial in range(10):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 4097))
            # Shuffle only the first portion, keep tail in order
            shuffle_len = random.randint(50, 200)
            random.shuffle(order[:shuffle_len])
            candidate = greedy_from_order(order)
            if len(candidate) > best_len:
                best_set = candidate
                best_len = len(candidate)
    
    # 5. Apply local search to the best set
    if time.time() < deadline - 5.0:
        improved_set = local_search(best_set, max_iters=30)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(improved_set)
    
    # 6. If we have more time, try local search on a few other good candidates
    if time.time() < deadline - 8.0:
        # Re-run a couple of random greedy and try local search on them
        for trial in range(3):
            if time.time() > deadline - 8.0:
                break
            order = list(range(1, 4097))
            random.shuffle(order)
            candidate = greedy_from_order(order)
            if len(candidate) >= best_len - 1:  # only polish near-best
                improved = local_search(candidate, max_iters=20)
                if len(improved) > best_len:
                    best_set = improved
                    best_len = len(improved)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
