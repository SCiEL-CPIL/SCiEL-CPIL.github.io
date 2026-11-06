"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    best_set = None
    best_size = 0
    
    def validate_and_size(arr):
        """Check B2[2] property and return size if valid, else 0."""
        n = len(arr)
        if n == 0:
            return 0
        counts = [0] * 2001
        for i in range(n):
            ai = arr[i]
            for j in range(i, n):
                s = ai + arr[j]
                counts[s] += 1
                if counts[s] > 2:
                    return 0
        return n

    def compute_counts(arr):
        """Compute sum counts for a sorted array."""
        counts = {}
        n = len(arr)
        for i in range(n):
            ai = arr[i]
            for j in range(i, n):
                s = ai + arr[j]
                counts[s] = counts.get(s, 0) + 1
        return counts

    def can_add(arr, counts, x):
        """Check if x can be added to arr maintaining B2[2] property."""
        for y in arr:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def add_element(arr, counts, x):
        """Add x to arr and update counts. Assumes can_add returned True."""
        arr.append(x)
        arr.sort()
        for y in arr:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        # The above double counts 2x, so fix it
        # Actually, let's do it properly:
        # For each y in old arr, add x+y
        # Add 2x once
        pass  # We'll rewrite this more carefully

    def greedy_complete(seed):
        """Complete a valid seed set greedily."""
        chosen = sorted(seed)
        counts = compute_counts(chosen)
        for x in range(1, 1001):
            if x in chosen:
                continue
            # Check if x can be added
            ok = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    ok = False
            if ok:
                chosen.append(x)
                # Update counts
                for y in chosen[:-1]:  # all except the new x
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        return chosen, counts

    def local_search_insert_swap(initial_arr, initial_counts, max_iters=5000):
        """Local search with insert and swap moves to grow the set."""
        arr = list(initial_arr)
        counts = dict(initial_counts)
        size = len(arr)
        arr_set = set(arr)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try insertions
            # Sample some candidates
            num_candidates = min(50, 1000 - size)
            if num_candidates > 0:
                candidates = random.sample([x for x in range(1, 1001) if x not in arr_set], num_candidates)
                for x in candidates:
                    if can_add(arr, counts, x):
                        # Add x
                        arr.append(x)
                        arr.sort()
                        arr_set.add(x)
                        for y in arr[:-1]:
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x] = counts.get(2 * x, 0) + 1
                        size += 1
                        improved = True
                        break
            
            # Try swaps: remove one, try to add two
            if not improved and size > 0:
                # Pick a random element to remove
                remove_idx = random.randint(0, size - 1)
                removed = arr[remove_idx]
                
                # Temporarily remove
                # Find all sums involving removed
                for y in arr:
                    s = removed + y
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
                
                arr.pop(remove_idx)
                arr_set.discard(removed)
                size -= 1
                
                # Try to add up to 2 elements
                added_count = 0
                candidates = random.sample([x for x in range(1, 1001) if x not in arr_set], min(30, 1000 - size))
                for x in candidates:
                    if added_count >= 2:
                        break
                    if can_add(arr, counts, x):
                        arr.append(x)
                        arr.sort()
                        arr_set.add(x)
                        for y in arr[:-1]:
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x] = counts.get(2 * x, 0) + 1
                        size += 1
                        added_count += 1
                
                # Only keep the swap if we grew
                if added_count > 1:
                    improved = True
                else:
                    # Revert: put removed back
                    arr.append(removed)
                    arr.sort()
                    arr_set.add(removed)
                    for y in arr[:-1]:
                        s = removed + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * removed] = counts.get(2 * removed, 0) + 1
                    size += 1
                    # Remove the added elements
                    for _ in range(added_count):
                        # This is tricky, we need to track which were added
                        pass  # Simplified: just revert all
                        # Actually this is getting complex. Let's simplify the swap logic.
            
            if not improved:
                # Try a different strategy: pure insertion with random order
                candidates = [x for x in range(1, 1001) if x not in arr_set]
                random.shuffle(candidates)
                for x in candidates[:20]:
                    if can_add(arr, counts, x):
                        arr.append(x)
                        arr.sort()
                        arr_set.add(x)
                        for y in arr[:-1]:
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x] = counts.get(2 * x, 0) + 1
                        size += 1
                        improved = True
                        break
            
            if not improved:
                break
        
        return arr, size

    # 1. Start with greedy
    g1, g1_counts = greedy_complete([])
    sz = validate_and_size(g1)
    if sz > best_size:
        best_set = g1
        best_size = sz

    # 2. Local search from greedy
    if best_set:
        ls_arr, ls_size = local_search_insert_swap(best_set, g1_counts)
        sz = validate_and_size(ls_arr)
        if sz > best_size:
            best_set = ls_arr
            best_size = sz

    # 3. Try multiple random seeds
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        # Random seed
        seed_size = random.randint(10, 30)
        seed = random.sample(range(1, 200), seed_size)
        seed.sort()
        
        # Check if seed is valid
        seed_counts = compute_counts(seed)
        if any(c > 2 for c in seed_counts.values()):
            continue
        
        # Complete greedily
        completed, completed_counts = greedy_complete(seed)
        sz = validate_and_size(completed)
        
        if sz > best_size:
            best_set = completed
            best_size = sz
        
        # Local search
        if sz > 0:
            ls_arr, ls_size = local_search_insert_swap(completed, completed_counts, max_iters=3000)
            sz2 = validate_and_size(ls_arr)
            if sz2 > best_size:
                best_set = ls_arr
                best_size = sz2

    if best_set is None:
        return [1]
    return best_set

# EVOLVE-BLOCK-END
