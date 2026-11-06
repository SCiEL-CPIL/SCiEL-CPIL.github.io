"""Rewritten program with stronger algorithm exploiting arithmetic structure."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline, 8s safety margin

    N = 256

    # Build set of perfect squares up to 2*N
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (diagonal constraint)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)

    # Map valid numbers to indices
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # Precompute degree
    degrees = [bin(adj[i]).count('1') for i in range(M)]

    # Greedy initialization with multiple strategies
    def greedy_min_degree():
        """Greedy: pick element with minimum conflicts with selected."""
        selected = 0
        remaining = list(range(M))
        while remaining:
            best_i = None
            best_conflicts = float('inf')
            for i in remaining:
                c = bin(adj[i] & selected).count('1')
                if c == 0:
                    best_i = i
                    best_conflicts = 0
                    break
                if c < best_conflicts:
                    best_conflicts = c
                    best_i = i
            if best_i is None or best_conflicts > 0:
                break
            selected |= (1 << best_i)
            remaining.remove(best_i)
        return selected

    def greedy_degree_order():
        """Greedy: order by degree ascending, add if no conflict."""
        order = sorted(range(M), key=lambda i: degrees[i])
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def greedy_random_order():
        """Greedy with random order."""
        order = list(range(M))
        random.shuffle(order)
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def greedy_weighted():
        """Greedy: order by (degree, random tiebreak)."""
        order = sorted(range(M), key=lambda i: (degrees[i], random.random()))
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def polish(mask, size):
        """Greedily add any vertex that doesn't conflict, making the set maximal."""
        added = 0
        for i in range(M):
            if not (mask >> i) & 1 and (adj[i] & mask) == 0:
                mask |= (1 << i)
                size += 1
                added += 1
        return mask, size

    # Get initial solutions from different strategies
    best_mask = greedy_degree_order()
    best_size = bin(best_mask).count('1')
    best_mask, best_size = polish(best_mask, best_size)

    mask2 = greedy_min_degree()
    sz2 = bin(mask2).count('1')
    mask2, sz2 = polish(mask2, sz2)
    if sz2 > best_size:
        best_mask = mask2
        best_size = sz2

    for _ in range(30):
        if time.time() > deadline - 10:
            break
        mask3 = greedy_random_order()
        sz3 = bin(mask3).count('1')
        mask3, sz3 = polish(mask3, sz3)
        if sz3 > best_size:
            best_mask = mask3
            best_size = sz3

    for _ in range(10):
        if time.time() > deadline - 10:
            break
        mask4 = greedy_weighted()
        sz4 = bin(mask4).count('1')
        mask4, sz4 = polish(mask4, sz4)
        if sz4 > best_size:
            best_mask = mask4
            best_size = sz4

    # Fast local search: maintain sel/unselected lists incrementally
    def local_search(start_mask, start_size, max_iters=50000):
        """Local search with 1-swap and 2-swap moves, using incremental list maintenance."""
        current_mask = start_mask
        current_size = start_size
        no_improve = 0
        max_no_improve = 150

        # Build initial lists
        sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
        unselected = [i for i in range(M) if not (current_mask >> i) & 1]

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
            if not sel_indices or not unselected:
                break

            improved = False

            # Try 1-swap: remove one, add up to two
            if random.random() < 0.6:
                ri = random.randrange(len(sel_indices))
                remove_idx = sel_indices[ri]
                trial_mask = current_mask & ~(1 << remove_idx)
                trial_size = current_size - 1

                # Find candidates that don't conflict with trial_mask
                candidates = [i for i in unselected if (adj[i] & trial_mask) == 0]
                if not candidates:
                    no_improve += 1
                    continue

                random.shuffle(candidates)
                added = 0
                new_sel_add = []
                for i in candidates:
                    if (adj[i] & trial_mask) == 0:
                        trial_mask |= (1 << i)
                        trial_size += 1
                        added += 1
                        new_sel_add.append(i)
                    if added >= 2:
                        break

                if trial_size > current_size:
                    # Accept: update lists
                    current_mask = trial_mask
                    current_size = trial_size
                    no_improve = 0
                    # Remove remove_idx from sel_indices
                    sel_indices[ri] = sel_indices[-1]
                    sel_indices.pop()
                    # Add new_sel_add to sel_indices
                    sel_indices.extend(new_sel_add)
                    # Remove new_sel_add from unselected
                    for i in new_sel_add:
                        if i in unselected:
                            unselected.remove(i)
                    continue
                elif trial_size == current_size and added >= 1:
                    if random.random() < 0.15:
                        current_mask = trial_mask
                        no_improve += 1
                        sel_indices[ri] = sel_indices[-1]
                        sel_indices.pop()
                        sel_indices.extend(new_sel_add)
                        for i in new_sel_add:
                            if i in unselected:
                                unselected.remove(i)
                        continue

            # Try 2-swap: remove two, add up to three
            else:
                if len(sel_indices) < 2:
                    no_improve += 1
                    continue

                # Pick two to remove
                r1 = random.randrange(len(sel_indices))
                r2 = random.randrange(len(sel_indices) - 1)
                if r2 >= r1:
                    r2 += 1
                p0, p1 = sel_indices[r1], sel_indices[r2]
                trial_mask = current_mask & ~(1 << p0) & ~(1 << p1)
                trial_size = current_size - 2

                candidates = [i for i in unselected if (adj[i] & trial_mask) == 0]
                if not candidates:
                    no_improve += 1
                    continue

                random.shuffle(candidates)
                added = 0
                new_sel_add = []
                for i in candidates:
                    if (adj[i] & trial_mask) == 0:
                        trial_mask |= (1 << i)
                        trial_size += 1
                        added += 1
                        new_sel_add.append(i)
                    if added >= 3:
                        break

                if trial_size > current_size:
                    current_mask = trial_mask
                    current_size = trial_size
                    no_improve = 0
                    # Remove p0 and p1 from sel_indices
                    # Rebuild for correctness (simpler than in-place removal of two)
                    sel_indices = [i for i in sel_indices if i != p0 and i != p1]
                    sel_indices.extend(new_sel_add)
                    for i in new_sel_add:
                        if i in unselected:
                            unselected.remove(i)
                    continue
                elif trial_size == current_size and added >= 2:
                    if random.random() < 0.1:
                        current_mask = trial_mask
                        no_improve += 1
                        sel_indices = [i for i in sel_indices if i != p0 and i != p1]
                        sel_indices.extend(new_sel_add)
                        for i in new_sel_add:
                            if i in unselected:
                                unselected.remove(i)
                        continue

            if not improved:
                no_improve += 1

        return current_mask, current_size

    # Run local search from best
    best_mask, best_size = local_search(best_mask, best_size, max_iters=100000)

    # Additional restarts with local search
    for restart in range(50):
        if time.time() > deadline - 5:
            break

        mask = greedy_random_order()
        sz = bin(mask).count('1')
        mask, sz = polish(mask, sz)

        mask, sz = local_search(mask, sz, max_iters=20000)

        if sz > best_size:
            best_mask = mask
            best_size = sz

    # Final polish from best
    best_mask, best_size = local_search(best_mask, best_size, max_iters=60000)
    best_mask, best_size = polish(best_mask, best_size)

    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    return sorted(result)
