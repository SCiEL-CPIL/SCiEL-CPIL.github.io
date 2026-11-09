<<<<<<< SEARCH
"""Rewritten program: conflict-graph greedy with random tie-breaking and polish."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55  # safety margin before 60s kill

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        for x in candidates:
            # Try to add x to A
            new_diffs_A = set()
            valid = True
            for a in A:
                d = abs(x - a)
                if d == 0:
                    continue
                new_diffs_A.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                new_sums = set()
                for a in A:
                    s = x + a
                    if s in sums_A:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_A:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    for a in A:
                        all_diffs.add(abs(x - a))
                    sums_A |= new_sums
                    diffs_A |= new_diffs_A
                    A.append(x)
                    continue
            # Try to add x to B
            new_diffs_B = set()
            valid = True
            for b in B:
                d = abs(x - b)
                if d == 0:
                    continue
                new_diffs_B.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                new_sums = set()
                for b in B:
                    s = x + b
                    if s in sums_B:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_B:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    for b in B:
                        all_diffs.add(abs(x - b))
                    sums_B |= new_sums
                    diffs_B |= new_diffs_B
                    B.append(x)

        return A, B

    def polish(A, B, time_limit=2.0):
        """Try to insert skipped elements into existing sets."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        for a in A:
            for a2 in A:
                sums_A.add(a + a2)
            for a2 in A:
                if a != a2:
                    diffs_A.add(abs(a - a2))

        for b in B:
            for b2 in B:
                sums_B.add(b + b2)
            for b2 in B:
                if b != b2:
                    diffs_B.add(abs(b - b2))

        all_diffs = diffs_A | diffs_B

        start = time.time()
        used = set(A) | set(B)
        remaining = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(remaining)

        for x in remaining:
            if time.time() - start > time_limit:
                break
            # Try A
            valid = True
            for a in A:
                d = abs(x - a)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                for a in A:
                    if x + a in sums_A or 2 * x in sums_A:
                        valid = False
                        break
            if valid and len(A) < max_size:
                for a in A:
                    all_diffs.add(abs(x - a))
                    sums_A.add(x + a)
                sums_A.add(2 * x)
                A.append(x)
                continue
            # Try B
            valid = True
            for b in B:
                d = abs(x - b)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                for b in B:
                    if x + b in sums_B or 2 * x in sums_B:
                        valid = False
                        break
            if valid and len(B) < max_size:
                for b in B:
                    all_diffs.add(abs(x - b))
                    sums_B.add(x + b)
                sums_B.add(2 * x)
                B.append(x)

        return A, B

    def greedy_with_coloring(seed=None):
        """Build conflict graph implicitly; use random ordering as coloring heuristic."""
        if seed is not None:
            random.seed(seed)
        
        # Random priority ordering for tie-breaking
        priorities = list(range(1, n + 1))
        random.shuffle(priorities)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        all_diffs = set()
        
        for x in priorities:
            # Try A first, then B
            valid_A = True
            for a in A:
                d = abs(x - a)
                if d in all_diffs:
                    valid_A = False
                    break
            if valid_A:
                for a in A:
                    if x + a in sums_A:
                        valid_A = False
                        break
                if valid_A and 2 * x in sums_A:
                    valid_A = False
            if valid_A and len(A) < max_size:
                for a in A:
                    all_diffs.add(abs(x - a))
                    sums_A.add(x + a)
                sums_A.add(2 * x)
                A.append(x)
                continue
            
            valid_B = True
            for b in B:
                d = abs(x - b)
                if d in all_diffs:
                    valid_B = False
                    break
            if valid_B:
                for b in B:
                    if x + b in sums_B:
                        valid_B = False
                        break
                if valid_B and 2 * x in sums_B:
                    valid_B = False
            if valid_B and len(B) < max_size:
                for b in B:
                    all_diffs.add(abs(x - b))
                    sums_B.add(x + b)
                sums_B.add(2 * x)
                B.append(x)
        
        return A, B

    best_A = []
    best_B = []
    best_size = 0
    seed_counter = 0

    # Phase 1: multiple greedy restarts
    while time.time() < deadline - 10:
        A, B = greedy_random(seed=seed_counter)
        A, B = polish(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Phase 2: coloring-based greedy on remaining time
    while time.time() < deadline - 2:
        A, B = greedy_with_coloring(seed=seed_counter)
        A, B = polish(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Final polish on best solution
    if best_size > 0:
        best_A, best_B = polish(best_A, best_B, time_limit=2.0)

    return [best_A, best_B]
=======
"""Structural change: bitmask-accelerated greedy + local search with move/swap."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 58  # safety margin before 60s kill

    # Bitmask for differences (differences range 1..255)
    # Bitmask for sums (sums range 2..512)

    def build_state(A, B):
        """Build internal state from sets A and B."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        for i, a in enumerate(A):
            for j, a2 in enumerate(A):
                sums_A.add(a + a2)
            for a2 in A:
                if a != a2:
                    diffs_A.add(abs(a - a2))
        for i, b in enumerate(B):
            for j, b2 in enumerate(B):
                sums_B.add(b + b2)
            for b2 in B:
                if b != b2:
                    diffs_B.add(abs(b - b2))
        all_diffs = diffs_A | diffs_B
        return sums_A, sums_B, diffs_A, diffs_B, all_diffs

    def can_add_to_set(x, S, sums_S, all_diffs):
        """Check if x can be added to set S with given sums and all_diffs."""
        for s in S:
            d = abs(x - s)
            if d in all_diffs:
                return False
            if x + s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_to_set(x, S, sums_S, diffs_S, all_diffs):
        """Add x to set S, updating all state."""
        for s in S:
            d = abs(x - s)
            all_diffs.add(d)
            diffs_S.add(d)
            sums_S.add(x + s)
        sums_S.add(2 * x)
        S.append(x)

    def remove_from_set(x, S, sums_S, diffs_S, all_diffs):
        """Remove x from set S, rebuilding sums and diffs."""
        S.remove(x)
        # Rebuild sums_S
        new_sums = set()
        for a in S:
            for b in S:
                new_sums.add(a + b)
        sums_S.clear()
        sums_S.update(new_sums)
        # Rebuild diffs_S
        new_diffs = set()
        for a in S:
            for b in S:
                if a != b:
                    new_diffs.add(abs(a - b))
        diffs_S.clear()
        diffs_S.update(new_diffs)
        # Rebuild all_diffs from both sets
        # We need the other set's diffs too; caller handles this
        return new_diffs

    def greedy_bitmask(seed=None):
        """Fast greedy using bitmask-style set operations."""
        if seed is not None:
            random.seed(seed)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try A
            if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                add_to_set(x, A, sums_A, diffs_A, all_diffs)
                continue
            # Try B
            if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                add_to_set(x, B, sums_B, diffs_B, all_diffs)
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B, all_diffs

    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B, all_diffs, time_limit=2.0):
        """Local search: try insertions, moves, and swaps."""
        start = time.time()
        improved = True
        
        while improved and time.time() - start < time_limit:
            improved = False
            
            # Phase 1: Try to insert unused elements
            used = set(A) | set(B)
            remaining = [x for x in range(1, n + 1) if x not in used]
            random.shuffle(remaining)
            
            for x in remaining:
                if time.time() - start > time_limit:
                    break
                if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                    add_to_set(x, A, sums_A, diffs_A, all_diffs)
                    improved = True
                    continue
                if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                    add_to_set(x, B, sums_B, diffs_B, all_diffs)
                    improved = True
            
            # Phase 2: Try moves (element from A to B or vice versa)
            if time.time() - start > time_limit:
                break
            for a in A[:]:
                if time.time() - start > time_limit:
                    break
                # Try moving a to B
                if len(B) < max_size:
                    # Remove a from A
                    old_diffs_A = set(diffs_A)
                    remove_from_set(a, A, sums_A, diffs_A, all_diffs)
                    # Rebuild all_diffs
                    all_diffs_new = diffs_A | diffs_B
                    # Try adding a to B
                    if can_add_to_set(a, B, sums_B, all_diffs_new):
                        add_to_set(a, B, sums_B, diffs_B, all_diffs_new)
                        all_diffs = diffs_A | diffs_B
                        improved = True
                    else:
                        # Revert: rebuild A state
                        A.append(a)
                        new_sums_A = set()
                        for x1 in A:
                            for x2 in A:
                                new_sums_A.add(x1 + x2)
                        sums_A.clear()
                        sums_A.update(new_sums_A)
                        new_diffs_A = set()
                        for x1 in A:
                            for x2 in A:
                                if x1 != x2:
                                    new_diffs_A.add(abs(x1 - x2))
                        diffs_A.clear()
                        diffs_A.update(new_diffs_A)
                        all_diffs = diffs_A | diffs_B
            
            # Phase 3: Try swaps (a in A <-> b in B)
            if time.time() - start > time_limit:
                break
            for a in A[:]:
                if time.time() - start > time_limit:
                    break
                for b in B[:]:
                    if time.time() - start > time_limit:
                        break
                    if a == b:
                        continue
                    # Remove a from A and b from B
                    A.remove(a)
                    B.remove(b)
                    # Rebuild sums and diffs
                    new_sums_A = set()
                    for x1 in A:
                        for x2 in A:
                            new_sums_A.add(x1 + x2)
                    new_sums_B = set()
                    for x1 in B:
                        for x2 in B:
                            new_sums_B.add(x1 + x2)
                    new_diffs_A = set()
                    for x1 in A:
                        for x2 in A:
                            if x1 != x2:
                                new_diffs_A.add(abs(x1 - x2))
                    new_diffs_B = set()
                    for x1 in B:
                        for x2 in B:
                            if x1 != x2:
                                new_diffs_B.add(abs(x1 - x2))
                    
                    # Try adding b to A and a to B
                    all_diffs_tmp = new_diffs_A | new_diffs_B
                    ok1 = can_add_to_set(b, A, new_sums_A, all_diffs_tmp)
                    ok2 = can_add_to_set(a, B, new_sums_B, all_diffs_tmp)
                    
                    if ok1 and ok2:
                        # Do the swap
                        A.append(b)
                        B.append(a)
                        # Rebuild all state
                        for x1 in A:
                            for x2 in A:
                                new_sums_A.add(x1 + x2)
                        for x1 in B:
                            for x2 in B:
                                new_sums_B.add(x1 + x2)
                        for x1 in A:
                            for x2 in A:
                                if x1 != x2:
                                    new_diffs_A.add(abs(x1 - x2))
                        for x1 in B:
                            for x2 in B:
                                if x1 != x2:
                                    new_diffs_B.add(abs(x1 - x2))
                        sums_A = new_sums_A
                        sums_B = new_sums_B
                        diffs_A = new_diffs_A
                        diffs_B = new_diffs_B
                        all_diffs = diffs_A | diffs_B
                        improved = True
                    else:
                        # Revert
                        A.append(a)
                        B.append(b)
            
            # Rebuild all_diffs at end of iteration
            all_diffs = diffs_A | diffs_B
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B, all_diffs

    best_A = []
    best_B = []
    best_size = 0
    seed_counter = 0

    while time.time() < deadline - 5:
        A, B, sums_A, sums_B, diffs_A, diffs_B, all_diffs = greedy_bitmask(seed=seed_counter)
        A, B, _, _, _, _, _ = local_search(A, B, sums_A, sums_B, diffs_A, diffs_B, all_diffs, time_limit=1.5)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Final local search on best
    if best_size > 0:
        sums_A, sums_B, diffs_A, diffs_B, all_diffs = build_state(best_A, best_B)
        best_A, best_B, _, _, _, _, _ = local_search(best_A, best_B, sums_A, sums_B, diffs_A, diffs_B, all_diffs, time_limit=3.0)

    return [best_A, best_B]
>>>>>>> REPLACE
