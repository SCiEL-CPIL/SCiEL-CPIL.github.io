"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None

    # Use bitmask for fast sum membership: sums range from 2 to 512, so we need 513 bits
    # Represent as a Python int (arbitrary precision)
    
    def can_add(x, selected, sums_mask):
        """Check if x can be added to the Sidon set."""
        # Check 2*x
        if (sums_mask >> (2 * x)) & 1:
            return False
        # Check x + a for all a in selected
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                return False
        return True

    def get_addable(selected, sums_mask):
        """Get list of all elements that can be added."""
        addable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if can_add(x, selected, sums_mask):
                addable.append(x)
        return addable

    def score_element(x, selected, sums_mask, addable):
        """Score element x by how many other addable elements it would block."""
        new_sums = 1 << (2 * x)
        for a in selected:
            new_sums |= 1 << (x + a)
        
        blocked = 0
        for y in addable:
            if y == x:
                continue
            # Check if y would still be addable after adding x
            # y needs: 2y not in sums_mask | new_sums, and for all a in selected, y+a not in sums_mask | new_sums
            if (new_sums >> (2 * y)) & 1:
                blocked += 1
                continue
            blocked_y = False
            for a in selected:
                if (new_sums >> (y + a)) & 1:
                    blocked_y = True
                    break
            if not blocked_y:
                # Also check y + x
                if (sums_mask >> (y + x)) & 1:
                    blocked_y = True
            if blocked_y:
                blocked += 1
        return blocked

    def backtrack(selected, sums_mask):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        # Check if maximal
        addable = get_addable(selected, sums_mask)

        if not addable:
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        if len(selected) >= best_size:
            return

        # Score by blocking potential
        scored = []
        for x in addable:
            blocked = score_element(x, selected, sums_mask, addable)
            scored.append((blocked, x))

        scored.sort(key=lambda t: (-t[0], t[1]))

        # Try top candidates
        for _, x in scored[:5]:
            new_sums = sums_mask | (1 << (2 * x))
            for a in selected:
                new_sums |= 1 << (x + a)
            
            selected.append(x)
            backtrack(selected, new_sums)
            selected.pop()
            
            if time.time() > deadline:
                return

    # Greedy baseline
    selected_greedy = []
    sums_mask_greedy = 0
    remaining = list(range(1, N + 1))
    while remaining:
        best_x = None
        best_block = -1
        for x in remaining:
            if not can_add(x, selected_greedy, sums_mask_greedy):
                continue
            new_sums = sums_mask_greedy | (1 << (2 * x))
            for a in selected_greedy:
                new_sums |= 1 << (x + a)
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                if (new_sums >> (2 * y)) & 1:
                    blocked += 1
                    continue
                blocked_y = False
                for a in selected_greedy:
                    if (new_sums >> (y + a)) & 1:
                        blocked_y = True
                        break
                if blocked_y:
                    blocked += 1
            if blocked > best_block:
                best_block = blocked
                best_x = x
        if best_x is None:
            break
        selected_greedy.append(best_x)
        sums_mask_greedy |= 1 << (2 * best_x)
        for a in selected_greedy:
            sums_mask_greedy |= 1 << (best_x + a)
        remaining.remove(best_x)

    if len(selected_greedy) < best_size:
        best_size = len(selected_greedy)
        best_set = list(selected_greedy)

    # Backtracking search with multiple restarts
    random.seed(42)
    for restart in range(15):
        if time.time() > deadline:
            break
        
        selected_bt = []
        backtrack(selected_bt, 0)

    if best_set is not None:
        return best_set
    else:
        return selected_greedy
# EVOLVE-BLOCK-END
