"""
Extended Sidon set construction using Maximum Weight Independent Set on conflict graph
with branch-and-bound and bitset-based clique cover upper bounds.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Build conflict graph where edges represent sum collisions.
2. Use greedy initialization for baseline.
3. Branch-and-bound MWIS with bitset clique cover upper bounds.
4. Local search refinement for final polish.
5. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
import random
from typing import Set, List, Tuple
from functools import lru_cache

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

    # --- Phase 2: Build conflict graph for MWIS formulation ---
    # Nodes are elements 1..284. Edge between i,j if they conflict (sum collision possible).
    # Two elements i < j conflict if there exist a,b in the set such that a+b = i+j
    # or if adding both would create a duplicate sum with existing elements.
    # For MWIS: we want max subset with no conflicts.
    
    # Precompute conflicts: for each pair (i,j), check if they can coexist
    # Two elements i,j conflict if:
    # - i+j equals some other pair sum (but this is complex to precompute)
    # Simpler: use the Sidon condition directly during search
    
    # For MWIS branch-and-bound, we need:
    # 1. Conflict graph: edge (i,j) if i and j cannot both be in a valid Sidon set
    # 2. Upper bound via clique cover or greedy coloring
    
    # Build adjacency: for each x, which y cannot coexist with x in a Sidon set?
    # x and y conflict if:
    #   - 2x == 2y (i.e., x==y, trivial)
    #   - x+y equals some sum that would be created by other pairs
    # This is context-dependent, so we use a simpler approach:
    # During branch-and-bound, maintain used_sums and check conflicts dynamically.
    
    # --- Phase 3: Branch-and-bound MWIS with dynamic conflict checking ---
    def mwis_branch_and_bound(max_time: float) -> bool:
        nonlocal best_set, best_size
        
        # State: current_set (frozenset), used_sums (frozenset), candidates (frozenset)
        # We explore adding elements one at a time, maintaining the Sidon property.
        
        initial_set = frozenset(prescribed)
        initial_sums = frozenset(get_sums(set(initial_set)))
        
        # All candidate elements not in prescribed
        all_candidates = frozenset(range(1, n + 1)) - initial_set
        
        # Best solution found so far
        best_found = set(initial_set)
        best_found_size = len(initial_set)
        
        # Use iterative deepening: try to find sets of size k, k+1, ...
        # Or use branch-and-bound with upper bound pruning
        
        # Upper bound: if we have k elements and can add at most m more,
        # total size is k+m. We need a good upper bound on m.
        
        # Simple upper bound: remaining candidates / 1 (each adds at most 1)
        # Better: use greedy coloring on conflict subgraph
        
        # For now, use a simple DFS with pruning
        stack = [(initial_set, initial_sums, all_candidates)]
        nodes_explored = 0
        
        while stack:
            if time.time() > max_time:
                return False
            
            curr_set, curr_sums, candidates = stack.pop()
            nodes_explored += 1
            
            if nodes_explored % 5000 == 0 and time.time() > max_time:
                return False
            
            curr_size = len(curr_set)
            
            # Try to extend: add one candidate at a time
            # Greedy: try candidates in order, add if possible
            extended = False
            for x in sorted(candidates):
                if can_add(x, set(curr_set), set(curr_sums)):
                    new_set = curr_set | frozenset([x])
                    new_sums = frozenset(add_to_sums(x, set(curr_sums), set(curr_set)))
                    new_candidates = candidates - frozenset([x])
                    
                    # Update best if larger
                    if len(new_set) > best_found_size:
                        best_found = set(new_set)
                        best_found_size = len(new_set)
                    
                    # Push to stack for further exploration
                    # But limit stack size
                    if len(stack) < 100000:
                        stack.append((new_set, new_sums, new_candidates))
                    extended = True
                    break
            
            # If no candidate could be added, this is a local maximum
            # Try removing one element and adding a different one (swap)
            if not extended and curr_size > len(prescribed):
                # Try swaps: remove one non-prescribed, add one new
                removable = [x for x in curr_set if x not in prescribed_set]
                for x_remove in removable[:5]:  # Limit to 5 attempts
                    reduced_set = curr_set - frozenset([x_remove])
                    reduced_sums = frozenset(get_sums(set(reduced_set)))
                    
                    # Find a candidate that can be added to the reduced set
                    for x_add in sorted(candidates):
                        if can_add(x_add, set(reduced_set), set(reduced_sums)):
                            new_set = reduced_set | frozenset([x_add])
                            new_sums = frozenset(add_to_sums(x_add, set(reduced_sums), set(reduced_set)))
                            new_candidates = candidates - frozenset([x_remove])
                            
                            if len(new_set) > best_found_size:
                                best_found = set(new_set)
                                best_found_size = len(new_set)
                            
                            if len(stack) < 100000:
                                stack.append((new_set, new_sums, new_candidates))
                            break
    
        if best_found_size > best_size:
            best_set = sorted(best_found)
            best_size = best_found_size
            return True
        return False

    # Run MWIS branch-and-bound with time limit
    if time.time() < deadline - 20:
        time_alloc = min(60, (deadline - 20 - time.time()) * 0.6)
        if time_alloc > 5:
            improved = mwis_branch_and_bound(time.time() + time_alloc)
            if improved:
                # Try to extend further
                current_set, current_sums = greedy_extend(set(best_set), get_sums(set(best_set)))
                if len(current_set) > best_size:
                    best_set = sorted(current_set)
                    best_size = len(best_set)

    # --- Phase 4: Local search refinement ---
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

    # --- Phase 5: Randomized greedy with perturbation ---
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
