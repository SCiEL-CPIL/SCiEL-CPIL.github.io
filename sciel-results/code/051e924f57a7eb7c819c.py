N = 256

import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # 3s buffer before 30s limit

    # Precompute: for each pair (a,b) with a<=b, the sum s=a+b
    # Two elements conflict in a Sidon set if they share a sum with some third element
    # Actually, for Sidon: no two pairs (a,b) with a<=b have the same sum
    # So when adding x to set S, x+a for all a in S must be distinct from each other
    # and from all existing sums, and 2x must also be distinct.

    def greedy_randomized():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        selected = []
        sums = set()

        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            # Check internal consistency of new sums
            if len(set(added_sums)) != len(added_sums):
                continue
            # Check against existing sums
            if sums.intersection(added_sums):
                continue

            selected.append(x)
            sums.update(added_sums)

        return selected

    def build_sums(selected):
        """Build the set of all pair sums a+b with a<=b from selected."""
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def check_maximal(selected, sums):
        """Check if set is inclusion-maximal. Returns True if maximal."""
        sel_set = set(selected)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if not sums.intersection(added_sums):
                return False  # x can be added, not maximal
        return True

    def greedy_maximal_dominating():
        """
        Build a Sidon set that is inclusion-maximal by construction.
        Strategy: process elements in random order; maintain a 'dominated' set.
        An element x is dominated if adding it would conflict with some already-selected
        element's sums. We select elements that are NOT dominated, ensuring maximality.
        
        Key insight: if we process all elements and select those that don't conflict,
        the result is maximal because every unselected element conflicts with at least
        one selected element's sums.
        """
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        selected = []
        sums = set()
        dominated = set()  # elements that cannot be added

        for x in candidates:
            if x in dominated:
                continue
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            selected.append(x)
            sums.update(added_sums)
            # Mark all elements that conflict with this new selection as dominated
            # An element y conflicts if y+a is in sums for some a in selected, or 2y is in sums,
            # or y+a1 == y+a2 for a1!=a2 in selected (impossible since a's are distinct),
            # or y+a == 2y (i.e., y=a, already in set), etc.
            # Simpler: just check each remaining candidate
            # But that's O(N^2) per addition... let's be smarter.
            # For each a in selected, y+a is a sum. So y is dominated if y+a in sums for some a in selected.
            # Also 2y in sums means y is dominated.
            # And y+a1 = y+a2 impossible for distinct a's.
            # Also need: for any existing pair (a,b) with a+b=s, if y+a=s or y+b=s then conflict.
            # That's covered by: y+a in sums for some a in selected.
            # Also 2y in sums.
            # Also: y+a = y+b for a!=b -> impossible.
            # Also: y+a = 2y -> a=y, y already selected.
            # So: y is dominated iff (2*y in sums) or any((y+a) in sums for a in selected)
            # But we also need: the new sums from y must not conflict with each other.
            # y+a1 = y+a2 -> a1=a2, fine.
            # y+a = 2y -> a=y, y not in selected (we're checking unselected y).
            # So internal consistency of added_sums is automatically satisfied for unselected y
            # as long as y is not in selected.
            # Wait: y+a for a in selected: these are all distinct since a's are distinct.
            # 2y vs y+a: 2y = y+a -> a=y, but y not in selected. So fine.
            # So for unselected y, added_sums are always internally consistent.
            # Therefore y is dominated iff (2*y in sums) or (y+a in sums for some a in selected).
            
            # Optimize: precompute which y values are dominated
            # For each a in selected, y is dominated if y+a in sums, i.e., y in {s-a for s in sums}
            # And y is dominated if 2y in sums, i.e., y in {s//2 for s in sums if s%2==0}
            
            # This is getting complex. Let's just do it directly but efficiently.
            for y in candidates:
                if y in dominated or y in selected:
                    continue
                # Check if y can be added
                y_sums = [y + a for a in selected] + [2 * y]
                if sums.intersection(y_sums):
                    dominated.add(y)

        return selected

    def greedy_with_priority():
        """
        Greedy that prioritizes elements that dominate many others.
        Process in order of how many elements each would dominate if selected.
        """
        candidates = list(range(1, N + 1))
        
        # Precompute dominance: for each x, how many other elements would be dominated
        # if x were the first selected element.
        # If x is selected, sums = {2x}. An element y is dominated if 2y=2x (y=x) or y+x=2x (y=x).
        # So selecting x alone only dominates x itself. Not very useful as a metric.
        
        # Better: random order but with bias toward "spread out" elements
        # Elements far apart tend to create more distinct sums.
        random.shuffle(candidates)
        
        selected = []
        sums = set()

        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            selected.append(x)
            sums.update(added_sums)

        return selected

    best_set = None
    best_size = float('inf')

    # Phase 1: Many fast greedy restarts with different strategies
    while time.time() < deadline - 8.0:
        # Alternate between different greedy variants
        if random.random() < 0.5:
            current_set = greedy_randomized()
        else:
            current_set = greedy_with_priority()
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    if best_set is None:
        # Deterministic ascending greedy as fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected
        best_size = len(best_set)

    # Phase 2: Local search - try to remove elements while maintaining maximality
    def try_remove(best):
        """Try to remove one element and check if result is still maximal."""
        selected = list(best)
        sums = build_sums(selected)

        # Try removing each element
        for idx in range(len(selected) - 1, -1, -1):
            x = selected[idx]
            new_selected = selected[:idx] + selected[idx+1:]
            if not new_selected:
                continue

            new_sums = build_sums(new_selected)

            if check_maximal(new_selected, new_sums):
                return new_selected

        return None

    # Phase 3: Iterative removal with randomization
    while time.time() < deadline - 2.0:
        if not best_set:
            break
        result = try_remove(best_set)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result
        else:
            # Try randomized removals
            for _ in range(20):
                if time.time() >= deadline - 2.0:
                    break
                idx = random.randint(0, len(best_set) - 1)
                new_selected = best_set[:idx] + best_set[idx+1:]
                if not new_selected:
                    continue
                new_sums = build_sums(new_selected)
                if check_maximal(new_selected, new_sums):
                    best_set = new_selected
                    best_size = len(new_selected)
                    break
            else:
                # If no removal works, do a fresh greedy and try again
                if time.time() < deadline - 2.0:
                    current_set = greedy_randomized()
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = current_set

    # Phase 4: Try swap-based local search
    # Replace one element with another to potentially reduce size while maintaining maximality
    def try_swap(best):
        """Try swapping one element for another to maintain maximality."""
        selected = list(best)
        sums = build_sums(selected)
        sel_set = set(selected)
        
        for idx in range(len(selected)):
            x = selected[idx]
            new_selected = selected[:idx] + selected[idx+1:]
            if not new_selected:
                continue
            new_sums = build_sums(new_selected)
            
            # Try adding each missing element
            for y in range(1, N + 1):
                if y in sel_set:
                    continue
                added_sums = [y + a for a in new_selected] + [2 * y]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if new_sums.intersection(added_sums):
                    continue
                
                # Check if new set is maximal
                candidate = new_selected + [y]
                cand_sums = new_sums | set(added_sums)
                if check_maximal(candidate, cand_sums):
                    return candidate
        
        return None

    while time.time() < deadline - 1.0:
        if not best_set:
            break
        result = try_swap(best_set)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result
        else:
            # Try removal again
            result = try_remove(best_set)
            if result is not None and len(result) < best_size:
                best_size = len(result)
                best_set = result
            else:
                if time.time() < deadline - 1.0:
                    current_set = greedy_randomized()
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = current_set

    # Final verification pass
    def verify_final(selected):
        sel_set = set(selected)
        if len(sel_set) != len(selected):
            return False
        if any(x < 1 or x > N for x in selected):
            return False
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        # Check maximality
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) != len(added):
                continue
            if not sums.intersection(added):
                return False
        return True

    if not verify_final(best_set):
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    return best_set
