"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None

    def try_backtrack(selected, used_sums, forbidden_mask, depth):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        # Check if we've found a maximal set (all remaining elements are forbidden)
        # A set is maximal if no missing element can be added
        # An element x can be added if:
        # 1. x is not in selected
        # 2. 2*x is not in used_sums
        # 3. For all a in selected, x+a is not in used_sums
        # 4. x is not forbidden (i.e., not in selected and can form valid sums)

        # Quick check: if the number of addable elements is 0, we have a maximal set
        addable_count = 0
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check if x can be added
            can_add = True
            if 2 * x in used_sums:
                can_add = False
            else:
                for a in selected:
                    if (x + a) in used_sums:
                        can_add = False
                        break
            if can_add:
                addable_count += 1
                if addable_count > 0:
                    break

        if addable_count == 0:
            # We have a maximal Sidon set
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        # If we've already exceeded our best, prune
        if len(selected) >= best_size:
            return

        # Find all addable elements
        addable = []
        for x in range(1, N + 1):
            if x in selected:
                continue
            can_add = True
            if 2 * x in used_sums:
                can_add = False
            else:
                for a in selected:
                    if (x + a) in used_sums:
                        can_add = False
                        break
            if can_add:
                addable.append(x)

        if not addable:
            # All elements are forbidden, so we have a maximal set
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        # Score each addable element by how many other addable elements it would block
        # An element x blocks element y if adding x makes y no longer addable
        # This happens if: 2y is in {2x, x+y} OR for some a in selected, y+a is in {x+a, x+y, 2x}
        # Simplified: x blocks y if the new sums from adding x intersect with the required sums for y

        scored = []
        for x in addable:
            # Compute new sums if we add x
            new_sums = {2 * x}
            for a in selected:
                new_sums.add(x + a)

            # Count how many other addable elements this would block
            blocked = 0
            for y in addable:
                if y == x:
                    continue
                # Check if y would still be addable after adding x
                # y needs: 2y not in used_sums ∪ new_sums, and for all a in selected ∪ {x}, y+a not in used_sums ∪ new_sums
                if 2 * y in new_sums:
                    blocked += 1
                    continue
                blocked_y = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked_y = True
                        break
                if not blocked_y:
                    # Also check y + x
                    if (y + x) in used_sums:
                        blocked_y = True
                if blocked_y:
                    blocked += 1
            scored.append((blocked, x))

        # Sort by number of blocked elements (descending), then by value (ascending) for determinism
        scored.sort(key=lambda t: (-t[0], t[1]))

        # Try the top candidates first
        for blocked_count, x in scored[:5]:  # Try top 5 candidates
            # Add x to selected
            new_sums = set(used_sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)

            selected.append(x)
            try_backtrack(selected, new_sums, forbidden_mask, depth + 1)
            selected.pop()

            if time.time() > deadline:
                return

    # Run multiple randomized restarts with different orderings
    random.seed(42)

    # First, run a greedy baseline to get a good initial upper bound
    selected_greedy = []
    used_sums_greedy = set()
    remaining = list(range(1, N + 1))
    while remaining:
        best_x = None
        best_block = -1
        for x in remaining:
            can_add = True
            if 2 * x in used_sums_greedy:
                can_add = False
            else:
                for a in selected_greedy:
                    if (x + a) in used_sums_greedy:
                        can_add = False
                        break
            if not can_add:
                continue
            # Count how many this blocks
            new_sums = {2 * x}
            for a in selected_greedy:
                new_sums.add(x + a)
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    blocked += 1
                    continue
                blocked_y = False
                for a in selected_greedy:
                    if (y + a) in new_sums:
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
        used_sums_greedy.add(2 * best_x)
        for a in selected_greedy:
            used_sums_greedy.add(best_x + a)
        remaining.remove(best_x)

    if len(selected_greedy) < best_size:
        best_size = len(selected_greedy)
        best_set = list(selected_greedy)

    # Now run backtracking with the greedy solution as initial bound
    # Try a few different random orderings
    for restart in range(10):
        if time.time() > deadline:
            break

        # Shuffle the range for different search orders
        order = list(range(1, N + 1))
        random.shuffle(order)

        # Use backtracking starting from empty set
        # But use the order to break ties in the scoring
        selected_bt = []
        used_sums_bt = set()

        def backtrack_with_order(selected, used_sums):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return

            # Check if maximal
            addable = []
            for x in range(1, N + 1):
                if x in selected:
                    continue
                can_add = True
                if 2 * x in used_sums:
                    can_add = False
                else:
                    for a in selected:
                        if (x + a) in used_sums:
                            can_add = False
                            break
                if can_add:
                    addable.append(x)

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
                new_sums = {2 * x}
                for a in selected:
                    new_sums.add(x + a)
                blocked = 0
                for y in addable:
                    if y == x:
                        continue
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    blocked_y = False
                    for a in selected:
                        if (y + a) in new_sums:
                            blocked_y = True
                            break
                    if blocked_y:
                        blocked += 1
                scored.append((blocked, x))

            scored.sort(key=lambda t: (-t[0], t[1]))

            for _, x in scored[:4]:
                new_sums = set(used_sums)
                new_sums.add(2 * x)
                for a in selected:
                    new_sums.add(x + a)
                selected.append(x)
                backtrack_with_order(selected, new_sums)
                selected.pop()
                if time.time() > deadline:
                    return

        backtrack_with_order(selected_bt, used_sums_bt)

    if best_set is not None:
        return best_set
    else:
        # Fallback: return a valid Sidon set (the greedy one)
        return selected_greedy
# EVOLVE-BLOCK-END
