# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 620  # Safety margin

    # ---- Build forbidden difference set ----
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    forbidden_set = forbidden

    # ---- Conflict graph as bitmasks: conflict_mask[i] has bit j set iff i,j conflict ----
    # Note: (1 << 205) is too big for standard int in some contexts if not careful, but Python handles big ints.
    # However, bitwise ops on 205-bit ints are very fast.
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden_set:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def check_conflict(a, b):
        return ((conflict_mask[a] >> b) & 1) == 1

    def is_valid(s):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if check_conflict(s[i], s[j]):
                    return False
        return True

    # ---- Greedy build given an ordering ----
    def greedy_build(order):
        selected_mask = 0
        selected = []
        for cand in order:
            # Check if cand conflicts with any selected
            if selected_mask & conflict_mask[cand]:
                continue
            selected.append(cand)
            selected_mask |= (1 << cand)
        return selected

    # ---- Optimized Local Search ----
    # State: selected_mask, selected_list
    def local_search(current_list, rng, deadline):
        current_mask = 0
        for e in current_list:
            current_mask |= (1 << e)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_list_set = set(current_list)
            unselected = [i for i in range(m) if i not in current_list_set]
            
            # 1. Try adding any free element (1-add)
            for s in unselected:
                if (conflict_mask[s] & current_mask) == 0:
                    current_list.append(s)
                    current_mask |= (1 << s)
                    improved = True
            
            if time.time() >= deadline:
                break

            # 2. Try 1-for-1 swap
            # Rebuild unselected list if we added something
            if improved:
                current_list_set = set(current_list)
                unselected = [i for i in range(m) if i not in current_list_set]
            
            for s in unselected:
                if time.time() >= deadline:
                    break
                confs = [e for e in current_list if check_conflict(s, e)]
                if len(confs) == 1:
                    c = confs[0]
                    # Check if s conflicts with rest
                    rest_mask = current_mask & ~(1 << c)
                    if (conflict_mask[s] & rest_mask) == 0:
                        current_list.remove(c)
                        current_list.append(s)
                        current_mask = rest_mask | (1 << s)
                        improved = True
                        break
                elif len(confs) == 0:
                    # Should have been caught by 1-add, but just in case
                    current_list.append(s)
                    current_mask |= (1 << s)
                    improved = True
                    break

            if time.time() >= deadline:
                break

            # 3. Try 2-for-2 swap (remove 2, add 2)
            # This is more expensive, so limit candidates
            if improved:
                current_list_set = set(current_list)
                unselected = [i for i in range(m) if i not in current_list_set]
            
            # Sample a subset of unselected to keep it fast
            sample_size = min(50, len(unselected))
            candidates = rng.sample(unselected, sample_size) if len(unselected) > sample_size else unselected
            
            for i in range(len(candidates)):
                if time.time() >= deadline:
                    break
                s1 = candidates[i]
                confs1 = [e for e in current_list if check_conflict(s1, e)]
                if len(confs1) > 2:
                    continue
                
                for j in range(i + 1, len(candidates)):
                    s2 = candidates[j]
                    if check_conflict(s1, s2):
                        continue
                    confs2 = [e for e in current_list if check_conflict(s2, e)]
                    if len(confs2) > 2:
                        continue
                    
                    to_remove = set(confs1) | set(confs2)
                    if len(to_remove) > 2:
                        continue
                    
                    # Check if s1, s2 conflict with rest of current_list (excluding to_remove)
                    rest_mask = current_mask
                    for e in to_remove:
                        rest_mask &= ~(1 << e)
                    
                    if (conflict_mask[s1] & rest_mask) != 0:
                        continue
                    if (conflict_mask[s2] & rest_mask) != 0:
                        continue
                    
                    # Valid swap
                    for e in to_remove:
                        current_list.remove(e)
                    current_list.append(s1)
                    current_list.append(s2)
                    current_mask = rest_mask | (1 << s1) | (1 << s2)
                    improved = True
                    break
                if improved:
                    break

        return current_list

    # ---- Main Search Loop ----
    best = []
    best_size = 0
    rng = random.Random(42)

    # 1. Deterministic Greedy Starts
    base_orders = [
        list(range(m)),
        list(range(m - 1, -1, -1)),
        sorted(range(m), key=lambda x: x % 41),
        sorted(range(m), key=lambda x: x % 5),
    ]
    for order in base_orders:
        if time.time() >= deadline:
            break
        sel = greedy_build(order)
        sel = local_search(sel, rng, deadline)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # 2. Random Restarts
    # Budget remaining
    while time.time() < deadline - 5:
        order = list(range(m))
        rng.shuffle(order)
        sel = greedy_build(order)
        sel = local_search(sel, rng, deadline)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # 3. Simulated Annealing / Perturbation on Best
    if time.time() < deadline - 2:
        current = list(best)
        current_size = len(current)
        
        # Try to improve best with perturbations
        for _ in range(1000):
            if time.time() >= deadline - 2:
                break
            
            perturbed = list(current)
            # Remove 1-3 random elements
            k = rng.randint(1, 3)
            if len(perturbed) > k:
                to_remove = rng.sample(perturbed, k)
                for e in to_remove:
                    perturbed.remove(e)
            
            # Re-greedy fill
            perturbed_set = set(perturbed)
            order = list(range(m))
            rng.shuffle(order)
            for cand in order:
                if cand in perturbed_set:
                    continue
                mask = 0
                for e in perturbed:
                    mask |= (1 << e)
                if (conflict_mask[cand] & mask) == 0:
                    perturbed.append(cand)
                    perturbed_set.add(cand)
            
            # Local search
            perturbed = local_search(perturbed, rng, deadline)
            
            if len(perturbed) > current_size:
                current = perturbed
                current_size = len(current)
                if current_size > best_size:
                    best = current
                    best_size = current_size

    # Final Verification
    result = sorted(set(best))
    if not is_valid(result):
        # Fallback
        result = greedy_build(list(range(m)))
        if not is_valid(result):
            result = [0]

    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
