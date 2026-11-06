"""
Extended Sidon set construction using systematic backtracking CSP with pruning.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Fast greedy initialization to get a good baseline.
2. Systematic backtracking search with strong pruning (sum-set tracking +
   forward checking) to prove optimality or find larger sets.
3. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
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

    # --- Phase 1: Greedy initialization ---
    def greedy_init() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                for y in list(current_set):
                    if y != x:
                        used_sums.add(x + y)
                used_sums.add(2 * x)
        return sorted(current_set)

    best_set = greedy_init()
    best_size = len(best_set)

    # --- Phase 2: Systematic backtracking with pruning ---
    # We search for a valid set of size >= best_size + 1.
    # State: (current_set, used_sums, next_candidate_index)
    # Candidates are processed in increasing order.
    # Pruning:
    #   - If adding x creates a duplicate sum, skip.
    #   - Upper bound: current_size + (n - next_candidate_index + 1) <= best_size -> prune.
    #   - More aggressive: if remaining candidates can't fill the gap, prune.

    # Precompute candidate list (all numbers 1..284 except prescribed, sorted)
    all_candidates = [x for x in range(1, n + 1) if x not in prescribed_set]

    # We'll do a DFS that tries to build a set of size target_size.
    # To be efficient, we use the fact that the order matters for pruning.
    # We try to find ANY valid set of size best_size + 1.

    found_larger = False
    target_size = best_size + 1

    # Use iterative deepening on target size? No, just try best_size + 1 first.
    # If found, update best and try best_size + 2, etc.

    def backtrack(current_set: Set[int], used_sums: Set[int], start_idx: int, target: int, depth: int) -> bool:
        nonlocal best_set, best_size, found_larger
        
        if time.time() > deadline:
            return False

        if len(current_set) == target:
            # Found a valid set of size target
            found_larger = True
            best_set = sorted(current_set)
            best_size = target
            return True

        # Pruning: even if we add all remaining candidates, can we reach target?
        remaining = len(all_candidates) - start_idx
        if len(current_set) + remaining < target:
            return False

        # Forward checking / optimistic bound
        # A tighter bound: how many of the remaining candidates can actually be added?
        # This is expensive, so we use a simpler bound first.
        
        for i in range(start_idx, len(all_candidates)):
            if time.time() > deadline:
                return False
            
            x = all_candidates[i]
            if not can_add(x, current_set, used_sums):
                continue
            
            # Add x
            current_set.add(x)
            new_sums = set(used_sums)
            for y in current_set:
                if y != x:
                    new_sums.add(x + y)
            new_sums.add(2 * x)
            
            # Recurse
            if backtrack(current_set, new_sums, i + 1, target, depth + 1):
                return True
            
            # Undo
            current_set.remove(x)

        return False

    # Try to find a set of size best_size + 1
    initial_set = set(prescribed)
    initial_sums = get_sums(initial_set)
    
    # The backtracking above starts from scratch. This might be too slow if the space is large.
    # Let's use the greedy result as a guide or start point?
    # Actually, the backtracking explores subsets of all_candidates.
    # To speed up, we can order candidates by "potential" or just use the natural order.
    # The natural order (1..284) is often good for Sidon sets because small numbers conflict more.
    
    # Let's run the backtracking for target_size = best_size + 1
    if not backtrack(initial_set, initial_sums, 0, target_size, 0):
        # If not found, try best_size + 2? Unlikely if +1 failed, but possible if +1 was blocked by specific conflicts.
        # Usually, if you can't find +1, you can't find +2 either in the same search tree.
        # But the search might have been cut off by time.
        pass

    # --- Phase 3: Local search refinement if backtracking didn't find a larger set quickly ---
    # If we still have time, try to improve via swaps (simulated annealing style)
    if time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        
        T = 5.0
        T_min = 0.1
        alpha = 0.999
        
        while time.time() < deadline - 5 and T > T_min:
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
                
                # Evaluate: we want to maintain or increase size.
                # Since we swapped, size is constant.
                # We accept if valid (which it is by construction) to explore the space.
                # This helps escape local minima where no single add works, but a swap does.
                # However, we are looking for LARGER sets. Swaps don't increase size.
                # So this phase is only useful if we got stuck at a local optimum for size,
                # and a swap might open up a path to an add?
                # Actually, if we swap, the sum set changes, potentially freeing up sums for future adds.
                
                # Let's check if this new configuration allows an immediate add
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
                
                # Accept the swap if it led to an extension or just to explore
                if extended:
                    current_set = new_set
                    current_sums = new_sums
                else:
                    # Accept with probability based on temperature?
                    # Since size didn't change, we accept to explore.
                    if rng.random() < 0.5: # High acceptance rate for exploration
                        current_set = new_set
                        current_sums = new_sums
            
            T *= alpha

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
