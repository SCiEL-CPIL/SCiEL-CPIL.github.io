"""
Extended Sidon set construction using Branch-and-Bound with conflict graph.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Approach: Model as maximum independent set in a conflict graph where:
- Vertices: numbers 1..284
- Edge between i,j if they cannot both be in a Sidon set together
  (i.e., adding both would create duplicate pairwise sums with existing elements)
- Use branch-and-bound with bitset-based pruning
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
    
    # We'll use a bitset representation: bit i represents whether number (i+1) is selected
    # Prescribed elements must be included
    
    # Build conflict structure: for each element, track which sums it would create
    # Two elements conflict if they create the same sum
    
    # Precompute: for each element x, the sums it creates with all other elements
    # But this depends on the current set, so we need dynamic checking
    
    # Instead, use a different approach: maintain the set of used sums
    # and check if adding x creates any duplicate sum
    
    best_set = list(prescribed)
    best_size = len(best_set)
    
    # Helper to check if a set is valid and return the set of sums
    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums
    
    # Check if element x can be added to set s without creating duplicate sums
    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        # New sums created by adding x: x + y for y in s, and 2*x
        new_sums = {x + y for y in s} | {2 * x}
        # Check for conflicts with existing sums
        if new_sums & used_sums:
            return False
        # Also check that new_sums don't have internal duplicates
        # This happens if x + y1 = x + y2 (impossible since y1 != y2) 
        # or x + y = 2*x (i.e., y = x, but x not in s)
        # So no internal duplicates possible
        return True
    
    # Greedy initialization: try to build a large valid set
    def greedy_init() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        
        # Try adding elements in various orders
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                # Update used sums
                for y in list(current_set):
                    if y != x:
                        used_sums.add(x + y)
                used_sums.add(2 * x)
        
        return sorted(current_set)
    
    # Get initial solution
    init_set = greedy_init()
    if len(init_set) > best_size:
        best_set = init_set
        best_size = len(init_set)
    
    # Branch and Bound
    # State: current selected set, used sums, index of next candidate to consider
    # We process candidates in order 1..n, deciding include or exclude
    
    def branch_and_bound():
        nonlocal best_set, best_size
        
        # Sort candidates: prescribed first, then others
        # We'll use a recursive approach with memoization-like pruning
        
        # For efficiency, we'll use an iterative deepening approach
        # Try to find a solution of size k, then k+1, etc.
        
        # Start from current best and try to improve
        current_best = list(best_set)
        current_best_set = set(current_best)
        current_sums = get_sums(current_best_set)
        
        # Try to add more elements using local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each possible element
            for x in range(1, n + 1):
                if x in current_best_set:
                    continue
                if can_add(x, current_best_set, current_sums):
                    # Try adding x
                    new_set = current_best_set | {x}
                    new_sums = current_sums | {x + y for y in current_best_set} | {2 * x}
                    
                    # Check if this leads to a better solution
                    # Use a quick greedy extension
                    extended_set = set(new_set)
                    extended_sums = set(new_sums)
                    for y in range(1, n + 1):
                        if y in extended_set:
                            continue
                        if can_add(y, extended_set, extended_sums):
                            extended_set.add(y)
                            for z in list(extended_set):
                                if z != y:
                                    extended_sums.add(y + z)
                            extended_sums.add(2 * y)
                    
                    if len(extended_set) > len(current_best_set):
                        current_best_set = extended_set
                        current_sums = extended_sums
                        current_best = sorted(current_best_set)
                        if len(current_best) > best_size:
                            best_set = current_best
                            best_size = len(best_set)
                        improved = True
                        break  # Restart the loop with the improved set
        
        # Now try a more systematic branch and bound
        # Use the fact that we need to select a subset of [1..n] containing prescribed
        # such that all pairwise sums are distinct
        
        # Represent the problem as: choose a subset S of [1..n] containing prescribed
        # such that |{a+b : a,b in S, a<=b}| = |S|*(|S|+1)/2
        
        # Use a different strategy: start with prescribed, then try to add elements
        # in a way that maximizes the final set size
        
        # Use a priority-based approach: try elements that conflict with fewest others
        def compute_conflicts(x: int, current_set: Set[int]) -> int:
            """Count how many elements in current_set would conflict with x"""
            count = 0
            for y in current_set:
                # Check if x and y create a sum that's already used
                # This requires knowing the full sum set
                pass
            return count
        
        # Simulated annealing as fallback with better initialization
        import random
        rng = random.Random(42)
        
        T = 2.0
        T_min = 0.01
        alpha = 0.9998
        
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        
        while time.time() < deadline - 3 and T > T_min:
            # Generate neighbor
            move_type = rng.choice(['add', 'remove', 'swap'])
            new_set = set(current_set)
            
            if move_type == 'add':
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if candidates:
                    x = rng.choice(candidates)
                    if can_add(x, current_set, current_sums):
                        new_set.add(x)
                    else:
                        # Try a random swap to make room
                        removable = [x for x in current_set if x not in prescribed_set]
                        if removable:
                            x_remove = rng.choice(removable)
                            new_set.remove(x_remove)
                            # Recompute sums after removal
                            new_sums_temp = get_sums(new_set)
                            if can_add(x, new_set, new_sums_temp):
                                new_set.add(x)
                            else:
                                new_set = set(current_set)  # Revert
            elif move_type == 'remove':
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x = rng.choice(removable)
                    new_set.remove(x)
            elif move_type == 'swap':
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x_remove = rng.choice(removable)
                    candidates = [x for x in range(1, n + 1) if x not in current_set]
                    if candidates:
                        x_add = rng.choice(candidates)
                        new_set.remove(x_remove)
                        new_sums_temp = get_sums(new_set)
                        if can_add(x_add, new_set, new_sums_temp):
                            new_set.add(x_add)
                        else:
                            new_set.add(x_remove)  # Revert
                            new_set = set(current_set)
            
            if new_set == current_set:
                T *= alpha
                continue
            
            # Evaluate
            valid, _ = True, None
            new_sums = get_sums(new_set)
            # Check validity: all sums must be distinct
            # get_sums returns a set, so if len(new_sums) < expected, there were duplicates
            expected_sums = len(new_set) * (len(new_set) + 1) // 2
            valid = (len(new_sums) == expected_sums)
            
            if valid:
                delta = len(new_set) - len(current_set)
                if delta >= 0:
                    current_set = new_set
                    current_sums = new_sums
                    if len(current_set) > best_size:
                        best_set = sorted(current_set)
                        best_size = len(best_set)
                else:
                    prob = pow(2.718281828, delta / T)
                    if rng.random() < prob:
                        current_set = new_set
                        current_sums = new_sums
            else:
                # Invalid, reject
                pass
            
            T *= alpha
        
        return best_set
    
    # Run the main algorithm
    branch_and_bound()
    
    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        # Fallback to prescribed only
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
