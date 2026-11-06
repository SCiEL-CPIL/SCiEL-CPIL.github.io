"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n_max = 4096

    def can_add(x, selected, sums):
        """Check if x can be added to selected without triple sum collisions."""
        # 3x
        if 3 * x in sums:
            return False
        # 2x + s for s in selected
        for s in selected:
            if 2 * x + s in sums:
                return False
        # x + s + t for s <= t in selected
        k = len(selected)
        for i in range(k):
            s = selected[i]
            for j in range(i, k):
                t = selected[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x(x, selected, sums):
        """Add x to selected and update sums."""
        selected.append(x)
        sums.add(3 * x)
        for s in selected[:-1]:
            sums.add(2 * x + s)
        k = len(selected)
        for i in range(k - 1):
            s = selected[i]
            for j in range(i, k - 1):
                t = selected[j]
                sums.add(x + s + t)

    def greedy_forward(start_list=None, step=1, skip_set=None):
        """Greedy forward from 1 to n_max."""
        if start_list is None:
            start_list = []
        selected = list(start_list)
        # Build initial sums
        sums = set()
        k = len(selected)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(selected[i] + selected[j] + selected[l])
        
        existing = set(selected)
        for x in range(1, n_max + 1, step):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if skip_set and x in skip_set:
                continue
            if can_add(x, selected, sums):
                add_x(x, selected, sums)
                existing.add(x)
        return selected

    def greedy_backward(start_list=None, step=1):
        """Greedy backward from n_max to 1."""
        if start_list is None:
            start_list = []
        selected = list(start_list)
        sums = set()
        k = len(selected)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(selected[i] + selected[j] + selected[l])
        
        existing = set(selected)
        for x in range(n_max, 0, -step):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, selected, sums):
                add_x(x, selected, sums)
                existing.add(x)
        return selected

    def local_search(selected, max_swaps=500):
        """Try to improve set by swapping elements."""
        if time.time() > deadline:
            return selected
        
        selected = list(selected)
        sums = set()
        k = len(selected)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(selected[i] + selected[j] + selected[l])
        
        existing = set(selected)
        improved = True
        swaps_made = 0
        
        while improved and swaps_made < max_swaps and time.time() < deadline:
            improved = False
            # Try removing each element and adding a better one
            for idx in range(len(selected)):
                if time.time() > deadline:
                    break
                orig = selected[idx]
                # Temporarily remove
                selected_no_idx = selected[:idx] + selected[idx+1:]
                sums_no_idx = set()
                k2 = len(selected_no_idx)
                for i in range(k2):
                    for j in range(i, k2):
                        for l in range(j, k2):
                            sums_no_idx.add(selected_no_idx[i] + selected_no_idx[j] + selected_no_idx[l])
                
                # Try to add a new element
                best_new = None
                best_new_pos = None
                for new_x in range(1, n_max + 1):
                    if new_x in existing and new_x != orig:
                        continue
                    if can_add(new_x, selected_no_idx, sums_no_idx):
                        # Prefer smaller elements to leave room
                        if best_new is None or new_x < best_new:
                            best_new = new_x
                            best_new_pos = idx
                        if new_x <= orig:
                            break  # Found a good replacement
                
                if best_new is not None and best_new != orig:
                    # Perform swap
                    selected[idx] = best_new
                    existing.discard(orig)
                    existing.add(best_new)
                    # Rebuild sums
                    sums = set()
                    k3 = len(selected)
                    for i in range(k3):
                        for j in range(i, k3):
                            for l in range(j, k3):
                                sums.add(selected[i] + selected[j] + selected[l])
                    improved = True
                    swaps_made += 1
                    break
        
        return selected

    # Strategy 1: Start with sparse skeleton (powers of 4) and greedy fill
    base = [4**i for i in range(10) if 4**i <= n_max]
    base_set = set(base)
    
    best_set = []
    best_len = 0

    # Try forward greedy from sparse base
    s1 = greedy_forward(start_list=base)
    if len(s1) > best_len:
        best_len = len(s1)
        best_set = s1[:]
    
    if time.time() > deadline:
        return best_set[:80]

    # Try backward greedy from sparse base
    s2 = greedy_backward(start_list=base)
    if len(s2) > best_len:
        best_len = len(s2)
        best_set = s2[:]

    if time.time() > deadline:
        return best_set[:80]

    # Try forward greedy without base
    s3 = greedy_forward()
    if len(s3) > best_len:
        best_len = len(s3)
        best_set = s3[:]

    if time.time() > deadline:
        return best_set[:80]

    # Try backward greedy without base
    s4 = greedy_backward()
    if len(s4) > best_len:
        best_len = len(s4)
        best_set = s4[:]

    if time.time() > deadline:
        return best_set[:80]

    # Local search on best solution
    if time.time() < deadline - 5:
        improved = local_search(best_set, max_swaps=300)
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved

    # Randomized restarts with different skeletons
    random.seed(42)
    for attempt in range(30):
        if time.time() > deadline - 2:
            break
        
        # Random sparse skeleton
        skeleton_size = random.randint(5, 12)
        skeleton = random.sample(range(1, n_max + 1), skeleton_size)
        skeleton.sort()
        
        # Verify skeleton is valid
        valid = True
        sk_sums = set()
        k = len(skeleton)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s_val = skeleton[i] + skeleton[j] + skeleton[l]
                    if s_val in sk_sums:
                        valid = False
                        break
                    sk_sums.add(s_val)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Greedy fill from this skeleton
        result = greedy_forward(start_list=skeleton)
        if len(result) > best_len:
            best_len = len(result)
            best_set = result[:]

    return best_set[:80]
# EVOLVE-BLOCK-END
