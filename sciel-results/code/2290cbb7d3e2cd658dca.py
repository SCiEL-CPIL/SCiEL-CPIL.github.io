import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    n = 256
    max_size = 64

    def get_diffs(S):
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    class SetState:
        __slots__ = ['elements', 'sums', 'diffs', 'elem_set']

        def __init__(self):
            self.elements = []
            self.sums = set()
            self.diffs = set()
            self.elem_set = set()

        def try_add(self, x, other_diffs):
            if x in self.elem_set:
                return False, None, None
            
            new_sums = set()
            for s in self.elements:
                ns = x + s
                if ns in self.sums or ns in new_sums:
                    return False, None, None
                new_sums.add(ns)
            
            two_x = 2 * x
            if two_x in self.sums or two_x in new_sums:
                return False, None, None
            new_sums.add(two_x)

            new_diffs = set()
            for s in self.elements:
                d = abs(x - s)
                if d in other_diffs or d in self.diffs or d in new_diffs:
                    return False, None, None
                new_diffs.add(d)
            
            return True, new_sums, new_diffs

        def commit(self, x, new_sums, new_diffs):
            self.elements.append(x)
            self.elem_set.add(x)
            self.sums.update(new_sums)
            self.diffs.update(new_diffs)

        def remove(self, x):
            self.elements.remove(x)
            self.elem_set.remove(x)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

        def rebuild(self, elements):
            self.elements = list(elements)
            self.elem_set = set(self.elements)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

    def greedy_with_balance(ordering, rng=None):
        """
        Greedy construction that balances set sizes to avoid one set becoming
        too large (which would block the other due to difference disjointness).
        """
        A = SetState()
        B = SetState()
        
        if rng is None:
            rng = random.Random(0)

        for x in ordering:
            # Determine which set is "smaller" or has more room
            # If sizes are equal, alternate or use random tie-break
            if len(A.elements) < len(B.elements):
                order = [(A, B), (B, A)]
            elif len(B.elements) < len(A.elements):
                order = [(B, A), (A, B)]
            else:
                # Tie-break: try both, or random
                if rng.random() < 0.5:
                    order = [(A, B), (B, A)]
                else:
                    order = [(B, A), (A, B)]
                
            for target, other in order:
                if len(target.elements) >= max_size:
                    continue
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    break
        return A, B

    def standard_greedy(ordering, prefer_A_first=True):
        """Standard greedy for comparison."""
        A = SetState()
        B = SetState()
        for x in ordering:
            if prefer_A_first:
                order = [(A, B), (B, A)]
            else:
                order = [(B, A), (A, B)]
            for target, other in order:
                if len(target.elements) >= max_size:
                    continue
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    break
        return A, B

    def strong_polish(A_init, B_init, max_time=10.0, rng=None):
        """
        Iterative improvement:
        1. Try to add unused elements.
        2. If no adds possible, try swaps to rebalance/unblock.
        3. If still stuck, try removing an element to create space (kick).
        """
        if rng is None:
            rng = random.Random(42)
            
        A = SetState()
        B = SetState()
        A.rebuild(A_init)
        B.rebuild(B_init)

        best_total = len(A.elements) + len(B.elements)
        best_A = list(A.elements)
        best_B = list(B.elements)

        all_elements = set(range(1, n + 1))
        
        def try_add_to(dst, other, x):
            if len(dst.elements) >= max_size:
                return False
            success, new_sums, new_diffs = dst.try_add(x, other.diffs)
            if success:
                dst.commit(x, new_sums, new_diffs)
                return True
            return False

        def try_swap(src, dst, x):
            """Move x from src to dst. Returns True if successful."""
            src.remove(x)
            if try_add_to(dst, src, x):
                return True
            else:
                # Revert
                # Since we just removed x from src, adding it back should be safe 
                # provided no other changes happened. 
                # However, try_add checks against current state.
                success, ns, nd = src.try_add(x, dst.diffs)
                if success:
                    src.commit(x, ns, nd)
                else:
                    # Fallback: manual rebuild if something went wrong
                    src.rebuild(src.elements + [x])
                return False

        start = time.time()
        no_improve_count = 0
        max_no_improve = 500 # After this many failed steps, do a kick
        
        while time.time() - start < max_time:
            curr_total = len(A.elements) + len(B.elements)
            
            # Phase 1: Try to add unused elements
            used = A.elem_set | B.elem_set
            unused = list(all_elements - used)
            rng.shuffle(unused)
            
            added_any = False
            for x in unused:
                if time.time() - start > max_time:
                    break
                if try_add_to(A, B, x):
                    added_any = True
                    no_improve_count = 0
                    # Continue trying to add more
                elif try_add_to(B, A, x):
                    added_any = True
                    no_improve_count = 0
            
            if added_any:
                new_total = len(A.elements) + len(B.elements)
                if new_total > best_total:
                    best_total = new_total
                    best_A = list(A.elements)
                    best_B = list(B.elements)
                continue # Loop back to try adding more
            
            # Phase 2: If no adds, try swaps to unblock
            # We want to move elements from the larger set to the smaller, 
            # or just random swaps to find a valid configuration that allows adds.
            
            if no_improve_count < max_no_improve:
                # Try a few random swaps
                swaps_attempted = 0
                swap_success = False
                while swaps_attempted < 10 and time.time() - start < max_time:
                    swaps_attempted += 1
                    # Pick a source set (prefer larger)
                    if len(A.elements) > len(B.elements):
                        src, dst = A, B
                    elif len(B.elements) > len(A.elements):
                        src, dst = B, A
                    else:
                        if rng.random() < 0.5:
                            src, dst = A, B
                        else:
                            src, dst = B, A
                    
                    if not src.elements:
                        continue
                    
                    x = rng.choice(src.elements)
                    if try_swap(src, dst, x):
                        swap_success = True
                        # After a swap, check if we can add anything
                        used = A.elem_set | B.elem_set
                        unused = list(all_elements - used)
                        rng.shuffle(unused)
                        for x2 in unused:
                            if try_add_to(A, B, x2) or try_add_to(B, A, x2):
                                break
                        break # Restart main loop to re-evaluate adds
                if swap_success:
                    new_total = len(A.elements) + len(B.elements)
                    if new_total > best_total:
                        best_total = new_total
                        best_A = list(A.elements)
                        best_B = list(B.elements)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Phase 3: Kick - Remove a random element to create space
                # This allows us to potentially add a different, more compatible element
                if A.elements or B.elements:
                    if A.elements and (not B.elements or rng.random() < 0.5):
                        x = rng.choice(A.elements)
                        A.remove(x)
                    else:
                        x = rng.choice(B.elements)
                        B.remove(x)
                    no_improve_count = 0
                    # Continue loop to try adding new elements

        return best_A, best_B

    # Generate orderings
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]

    # Interleaved low-high
    inter = []
    for i in range(1, n // 2 + 1):
        inter.append(i)
        inter.append(n + 1 - i)
    orderings.append(inter[:n])

    rng_global = random.Random(42)
    for _ in range(10):
        s = list(range(1, n + 1))
        rng_global.shuffle(s)
        orderings.append(s)

    candidates = []
    
    # 1. Run Balanced Greedy on several orderings
    for i, ordering in enumerate(orderings):
        if time.time() > deadline - 20:
            break
        rng_local = random.Random(100 + i)
        A, B = greedy_with_balance(ordering, rng_local)
        total = len(A.elements) + len(B.elements)
        if total > 0:
            candidates.append((total, A.elements, B.elements))

    # 2. Run Standard Greedy on a few (for diversity)
    for i in range(3):
        if time.time() > deadline - 20:
            break
        ordering = orderings[i]
        for pref in [True, False]:
            A, B = standard_greedy(ordering, pref)
            total = len(A.elements) + len(B.elements)
            if total > 0:
                candidates.append((total, A.elements, B.elements))

    # Sort candidates by initial size
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_total = 0
    best_A = None
    best_B = None
    
    # Polish the top candidates
    # We have ~20s left. Let's spend it on the top 3-5 candidates.
    num_to_polish = 5
    time_per_polish = (deadline - time.time() - 5) / num_to_polish if deadline - time.time() > 10 else 1.0
    
    for i in range(min(num_to_polish, len(candidates))):
        if time.time() > deadline - 5:
            break
        total, A_init, B_init = candidates[i]
        if total <= best_total:
            # If this candidate is already worse than our best polished result, skip?
            # No, because polishing might improve it. But if it's significantly worse, maybe skip.
            # Let's just polish if we have time.
            pass
        
        rng_polish = random.Random(200 + i)
        polished_A, polished_B = strong_polish(A_init, B_init, max_time=time_per_polish, rng=rng_polish)
        
        new_total = len(polished_A) + len(polished_B)
        if new_total > best_total:
            best_total = new_total
            best_A = polished_A
            best_B = polished_B

    # If we didn't find anything (shouldn't happen), fallback
    if best_A is None or best_B is None:
        # Use the best candidate directly
        if candidates:
            best_total, best_A, best_B = candidates[0]
        else:
            best_A = [1]
            best_B = [2]
            best_total = 2

    return [best_A, best_B]
