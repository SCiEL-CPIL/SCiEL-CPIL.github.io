# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-accelerated local search with multiple random restarts.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)

    # Build bitmask representation for each element
    # conflict_mask[i] has bit j set if (i - j) % m is in forbidden
    conflict_mask = np.zeros(m, dtype=np.uint32)
    for i in range(m):
        mask = 0
        for j in range(m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_valid(selected):
        """Check if selected set is valid (no forbidden differences)."""
        sel_mask = 0
        for s in selected:
            sel_mask |= (1 << s)
        for s in selected:
            if conflict_mask[s] & sel_mask:
                # Check if s conflicts with itself (shouldn't happen) or another
                # conflict_mask[s] includes s? No, we didn't add self.
                # But wait, the mask includes all j where difference is forbidden.
                # If s is in selected, and conflict_mask[s] has bit s set, that's a self-conflict.
                # We should ensure no self-conflicts in precomputation.
                # Let's re-check: (s-s)%m = 0, which is not in forbidden. So bit s is not set.
                # So if (conflict_mask[s] & sel_mask) != 0, there is a conflict with another element.
                return False
        return True

    def greedy_random(seed=None):
        """Greedy construction with random ordering."""
        if seed is not None:
            random.seed(seed)
        candidates = list(range(m))
        random.shuffle(candidates)
        selected = []
        sel_mask = 0
        for c in candidates:
            # Check if c conflicts with any selected element
            if conflict_mask[c] & sel_mask == 0:
                selected.append(c)
                sel_mask |= (1 << c)
        return selected

    def local_search(selected, max_iter=5000):
        """1-add and 1-swap local search using bitmasks."""
        sel_mask = 0
        for s in selected:
            sel_mask |= (1 << s)
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            # Try 1-add: add an element not in selected that doesn't conflict
            for c in range(m):
                if (sel_mask >> c) & 1:
                    continue
                if conflict_mask[c] & sel_mask == 0:
                    selected.append(c)
                    sel_mask |= (1 << c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add one (net gain 0 in size, but might enable further adds)
            # Actually, for maximizing size, 1-swap alone doesn't help unless it enables a subsequent add.
            # A better move is 2-exchange: remove two, add three? Or remove one, add one (no gain) then add?
            # Let's stick to 1-add first. If stuck, try removing one element and see if we can add two.
            # This is a 1-2 exchange.
            
            # Try 1-2 exchange: remove s, add c1, c2
            for s in selected:
                # Tentatively remove s
                new_sel_mask = sel_mask & ~(1 << s)
                # Find candidates to add
                addable = []
                for c in range(m):
                    if (new_sel_mask >> c) & 1:
                        continue
                    if conflict_mask[c] & new_sel_mask == 0:
                        addable.append(c)
                
                # Try adding pairs from addable
                for i in range(len(addable)):
                    c1 = addable[i]
                    for j in range(i + 1, len(addable)):
                        c2 = addable[j]
                        # Check if c1 and c2 conflict with each other
                        if conflict_mask[c1] & (1 << c2) or conflict_mask[c2] & (1 << c1):
                            continue
                        # Check if c1 and c2 conflict with the rest of new_sel_mask (already checked individually)
                        # Since c1 and c2 don't conflict with new_sel_mask individually, and don't conflict with each other,
                        # the set is valid.
                        selected.remove(s)
                        selected.append(c1)
                        selected.append(c2)
                        sel_mask = new_sel_mask | (1 << c1) | (1 << c2)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            
            # Try 1-1 swap (remove one, add one) to potentially enable future adds?
            # This is less effective for size maximization directly, but can escape local minima.
            # Let's try it if 1-add and 1-2 fail.
            if not improved:
                for s in selected:
                    new_sel_mask = sel_mask & ~(1 << s)
                    for c in range(m):
                        if (new_sel_mask >> c) & 1:
                            continue
                        if conflict_mask[c] & new_sel_mask == 0:
                            # Swap s with c
                            selected.remove(s)
                            selected.append(c)
                            sel_mask = new_sel_mask | (1 << c)
                            improved = True
                            break
                    if improved:
                        break

        return selected

    # Start with a few random restarts
    best_set = []
    best_size = 0

    # First, try a deterministic greedy (order 0,1,2,...)
    selected = greedy_random(seed=0)
    # Note: greedy_random shuffles, so seed=0 is not deterministic order 0..204.
    # Let's do a pure deterministic greedy first.
    selected_det = []
    sel_mask = 0
    for c in range(m):
        if conflict_mask[c] & sel_mask == 0:
            selected_det.append(c)
            sel_mask |= (1 << c)
    
    selected_det = local_search(selected_det)
    if len(selected_det) > best_size:
        best_size = len(selected_det)
        best_set = list(selected_det)

    # Now do random restarts
    num_restarts = 0
    while time.time() < deadline:
        seed = random.randint(0, 2**31 - 1)
        selected = greedy_random(seed=seed)
        selected = local_search(selected)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        num_restarts += 1
        # If we've done many restarts, increase local search effort or break?
        # For now, keep it simple.

    # Final validation
    if not is_valid(best_set):
        # Fallback to trivial set if something went wrong
        return np.array([0], dtype=np.int64)

    return np.array(best_set, dtype=np.int64)


# EVOLVE-BLOCK-END
