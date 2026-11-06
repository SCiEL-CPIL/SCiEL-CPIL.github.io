"""
Extended Sidon set construction using systematic backtracking with sum-set pruning
plus heuristic local search.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Fast greedy initialization to get a baseline.
2. Systematic backtracking with aggressive pruning to find larger valid sets.
3. Local search (swaps) for refinement.
4. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
import random
from typing import Set, List, Tuple

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    n = max_val

    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        new_sums = {x + y for y in s} | {2 * x}
        if new_sums & used_sums:
            return False
        return True

    def add_to_sums(x: int, used_sums: Set[int], s: Set[int]) -> Set[int]:
        new_sums = set(used_sums)
        for y in s:
            if y != x:
                new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # --- Phase 1: Greedy initialization ---
    def greedy_init() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                used_sums = add_to_sums(x, used_sums, current_set)
        return sorted(current_set)

    best_set = greedy_init()
    best_size = len(best_set)

    # --- Phase 2: Systematic Backtracking with Pruning ---
    # This is the key improvement: systematic exploration with pruning
    # to find larger sets than greedy can achieve.
    
    def backtracking_search(target_size: int, max_time: float) -> bool:
        nonlocal best_set, best_size
        
        # State: current_set (list, sorted), used_sums (set), next_start (int)
        # We build the set incrementally, choosing the next element to add.
        # Prescribed elements must be included.
        
        # Start with prescribed elements
        initial_set = sorted(prescribed)
        initial_sums = get_sums(set(initial_set))
        
        # We need to add (target_size - len(prescribed)) more elements
        remaining_to_add = target_size - len(prescribed)
        
        if remaining_to_add <= 0:
            return True
        
        # Use a stack-based DFS to avoid recursion depth issues
        # Each stack item: (current_set_tuple, used_sums, next_start, depth)
        # We use tuples for sets in the stack to avoid mutation issues
        
        stack = [(tuple(initial_set), initial_sums, 1, 0)]
        nodes_explored = 0
        
        while stack:
            if time.time() > max_time:
                return False
            
            curr_tuple, curr_sums, next_start, depth = stack.pop()
            nodes_explored += 1
            
            if nodes_explored % 10000 == 0 and time.time() > max_time:
                return False
            
            curr_set = set(curr_tuple)
            
            if depth == remaining_to_add:
                # Found a valid set of target_size
                best_set = list(curr_set)
                best_size = target_size
                return True
            
            # Try adding each candidate from next_start onwards
            # Candidates are values in [1, 284] not already in curr_set
            # We iterate in increasing order for systematic exploration
            
            # Optimization: we can bound the search
            # If we need k more elements, and the max possible value is n,
            # we need enough "room" in the sum space.
            
            for x in range(next_start, n + 1):
                if x in curr_set:
                    continue
                
                if not can_add(x, curr_set, curr_sums):
                    continue
                
                new_set = curr_set | {x}
                new_sums = add_to_sums(x, curr_sums, curr_set)
                new_tuple = tuple(sorted(new_set))
                new_depth = depth + 1
                
                # Push to stack (LIFO, so we explore in reverse order)
                # To explore smaller x first, push larger x first
                # Actually, let's push in reverse so smaller x is explored first
                # Since we want to find any valid set, order doesn't matter much
                # for finding existence, but for finding good sets, 
                # smaller elements might be better.
                
                # Heuristic: prune if remaining capacity is too small
                # With k more elements to add, the smallest possible new elements
                # would be x+1, x+2, ..., x+k
                # The largest sum would be (x+k) + (x+k-1) or similar
                # This is a rough bound; let's keep it simple for now.
                
                stack.append((new_tuple, new_sums, x + 1, new_depth))
                
                # Limit stack size to prevent memory explosion
                if len(stack) > 50000:
                    # Keep only the most promising states
                    # Heuristic: prefer states with smaller max element
                    stack.sort(key=lambda s: max(s[0]) if s[0] else 0)
                    stack = stack[:50000]
            
            # If stack is empty, we've exhausted all possibilities
            if not stack:
                return False
        
        return False

    # Try to find sets of increasing size using backtracking
    for target in range(best_size + 1, min(best_size + 5, 21)):
        if time.time() > deadline - 20:
            break
        
        # Allocate time for this target size
        # Give more time to smaller targets, less to larger
        time_alloc = min(30, (deadline - 20 - time.time()) * 0.5)
        
        if time_alloc < 5:
            break
        
        found = backtracking_search(target, time.time() + time_alloc)
        if found:
            # Found a set of this size, try to go larger
            continue
        else:
            # Didn't find a set of this size, stop
            break

    # --- Helper: greedy extend a set to local maximum ---
    def greedy_extend(s: Set[int], used_sums: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Add as many valid elements as possible in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x in s:
                    continue
                if can_add(x, s, used_sums):
                    s.add(x)
                    used_sums = add_to_sums(x, used_sums, s)
                    changed = True
        return s, used_sums

    # --- Phase 3: Local search refinement ---
    # Use swaps to escape local maxima
    if time.time() < deadline - 10:
        rng = random.Random(42)
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        # First, greedily extend to make sure we're at a local max
        current_set, current_sums = greedy_extend(current_set, current_sums)
        if len(current_set) > best_size:
            best_set = sorted(current_set)
            best_size = len(best_set)

        for iteration in range(50000):
            if time.time() > deadline - 5:
                break

            # Try a swap: remove one non-prescribed element, add one new one
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
            x_remove = rng.choice(removable)
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if not candidates:
                break
            x_add = rng.choice(candidates)

            new_set = set(current_set)
            new_set.remove(x_remove)

            # Check if x_add can be added to the reduced set
            new_sums_temp = get_sums(new_set)
            if can_add(x_add, new_set, new_sums_temp):
                new_set.add(x_add)
                new_sums = get_sums(new_set)

                # Try to extend: add any valid element
                extended = False
                for y in range(1, n + 1):
                    if y in new_set:
                        continue
                    if can_add(y, new_set, new_sums):
                        new_set.add(y)
                        new_sums = get_sums(new_set)
                        if len(new_set) > best_size:
                            best_set = sorted(new_set)
                            best_size = len(best_set)
                        extended = True
                        break

                # Accept swap if it led to extension or with some probability
                if extended or rng.random() < 0.3:
                    current_set = new_set
                    current_sums = new_sums

    # --- Phase 4: Randomized greedy with perturbation ---
    if time.time() < deadline - 5:
        rng2 = random.Random(12345)
        for trial in range(30):
            if time.time() > deadline - 3:
                break
            current_set = set(prescribed)
            used_sums = get_sums(current_set)
            candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
            shuffled = [(rng2.random(), x) for x in candidates]
            shuffled.sort(key=lambda t: t[0] * 0.7 + t[1] / n * 0.3)
            for _, x in shuffled:
                if can_add(x, current_set, used_sums):
                    current_set.add(x)
                    used_sums = add_to_sums(x, used_sums, current_set)
            current_set, used_sums = greedy_extend(current_set, used_sums)
            if len(current_set) > best_size:
                best_set = sorted(current_set)
                best_size = len(best_set)

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
