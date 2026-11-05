import numpy as np
import time
import random

N = 256
MAX_SUM = 2 * N  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Precompute conflict masks for each element x in [1, N]
    # conflict_mask[x] is a bitset of all elements y in [1, N] such that x+y is a "sum" that conflicts.
    # Actually, it's easier to think of it as:
    # If we have a set S, and we want to add x, we need to check if 2*x or x+a for a in S is in Sums(S).
    # Let's use a different representation:
    # For each x, precompute the set of "forbidden" pairs? No.
    
    # Better approach:
    # Represent the set of used sums as a bitset of size 2*N+1 (sums from 2 to 512).
    # When we add x, we add sums: 2*x, x+1, x+2, ..., x+N (but only for a in S).
    # This is still O(|S|) per addition.
    
    # Let's try a different strategy:
    # 1. Generate a random maximal Sidon set using a fast greedy method with bitsets.
    # 2. Use local search to minimize it.
    
    # Fast Greedy:
    # Maintain a bitset of available elements.
    # Maintain a bitset of used sums.
    # At each step, pick an element x that blocks the most available elements.
    
    # To make it fast, we can approximate "blocks most" by just picking the smallest or largest,
    # or by sampling. But let's try to be smart.
    
    # Actually, the key insight from inspiration 2 is vectorized bitmask.
    # Let's precompute for each x, the list of sums it generates with all possible y.
    # sums_with[x][y] = x + y.
    
    # Let's stick to a robust local search on a good initial solution.
    # Initial solution: Random Greedy
    # 1. Shuffle [1..N]
    # 2. Iterate, add if valid.
    # 3. This gives a maximal Sidon set.
    
    def is_valid_addition(x, selected, used_sums):
        # Check if 2*x is in used_sums
        if used_sums[2*x]:
            return False
        for a in selected:
            if used_sums[x + a]:
                return False
        return True

    def add_element(x, selected, used_sums):
        used_sums[2*x] = True
        for a in selected:
            used_sums[x + a] = True
        selected.append(x)

    def remove_element(x, selected, used_sums):
        # This is tricky because we need to know which sums are no longer used.
        # We can rebuild used_sums from scratch, which is O(|S|^2).
        # Or we can count how many times each sum is used.
        # Let's use a count array.
        pass

    # Let's use a count array for sums.
    # sum_count[s] = number of pairs (a,b) in S with a<=b such that a+b=s.
    
    def get_sum_counts(selected):
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for i, a in enumerate(selected):
            for b in selected[i:]:
                counts[a + b] += 1
        return counts

    def can_add_fast(x, selected, sum_counts):
        if sum_counts[2*x] > 0:
            return False
        for a in selected:
            if sum_counts[x + a] > 0:
                return False
        return True

    def add_fast(x, selected, sum_counts):
        sum_counts[2*x] += 1
        for a in selected:
            sum_counts[x + a] += 1
        selected.append(x)

    def remove_fast(x, selected, sum_counts):
        sum_counts[2*x] -= 1
        for a in selected:
            if a != x:
                sum_counts[x + a] -= 1
        selected.remove(x)

    # Initial Solution: Random Greedy
    def random_greedy():
        selected = []
        sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        available = list(range(1, N + 1))
        random.shuffle(available)
        
        for x in available:
            if can_add_fast(x, selected, sum_counts):
                add_fast(x, selected, sum_counts)
        
        # Ensure maximality (it should be maximal by construction if we iterate all available)
        # Wait, if we iterate all available and add if possible, the result is maximal.
        # Because if any y was not added, it must have been blocked by some element added later?
        # No, if y was skipped, it means it was not addable at that time.
        # But elements added later might block even more.
        # Actually, the standard greedy "add if possible" produces a maximal set.
        # Proof: If a set S is not maximal, there exists x not in S such that S U {x} is Sidon.
        # When we considered x, S_subset was a subset of S.
        # If x was not addable to S_subset, it means some sum conflict existed in S_subset.
        # But S is a superset of S_subset, so the conflict would still exist in S.
        # Thus x cannot be added to S.
        # So the set is maximal.
        
        return selected, sum_counts

    # Local Search: Minimize Cardinality
    # Move 1: Remove x if S\{x} is still maximal.
    # Move 2: Swap x for y if S\{x} U {y} is Sidon and maximal.
    
    def is_maximal(selected, sum_counts):
        # Check if any element can be added
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if can_add_fast(x, selected, sum_counts):
                return False
        return True

    def local_search(initial_selected, initial_sum_counts):
        selected = list(initial_selected)
        sum_counts = initial_sum_counts.copy()
        
        # Verify initial is maximal
        if not is_maximal(selected, sum_counts):
            # Repair: add elements until maximal
            # This shouldn't happen if random_greedy is correct, but just in case
            pass # random_greedy guarantees maximality
        
        best_selected = list(selected)
        best_size = len(selected)
        
        no_improvement = 0
        time_limit = time.time() + 45
        
        while time.time() < time_limit and no_improvement < 100:
            improved = False
            
            # Try Removals
            for i in range(len(selected)):
                x = selected[i]
                
                # Remove x
                remove_fast(x, selected, sum_counts)
                
                # Check if still maximal
                # Note: removing x might make some other element addable.
                # If x was the only thing blocking y, then y is now addable.
                # So we must check full maximality.
                if is_maximal(selected, sum_counts):
                    # Success
                    best_selected = list(selected)
                    best_size = len(selected)
                    improved = True
                    no_improvement = 0
                    break # Restart loop
                else:
                    # Restore x
                    add_fast(x, selected, sum_counts)
            
            if improved:
                continue
                
            # Try Swaps
            # Sample a few swaps
            if len(selected) > 1:
                for _ in range(10): # Try 10 random swaps
                    i = random.randint(0, len(selected) - 1)
                    x = selected[i]
                    # Pick a random missing element y
                    missing = [y for y in range(1, N + 1) if y not in selected]
                    if not missing:
                        break
                    y = random.choice(missing)
                    
                    # Remove x
                    remove_fast(x, selected, sum_counts)
                    # Add y
                    if can_add_fast(y, selected, sum_counts):
                        add_fast(y, selected, sum_counts)
                        
                        # Check maximality
                        if is_maximal(selected, sum_counts):
                            best_selected = list(selected)
                            best_size = len(selected)
                            improved = True
                            no_improvement = 0
                            break
                        else:
                            # Restore
                            remove_fast(y, selected, sum_counts)
                            add_fast(x, selected, sum_counts)
                    else:
                        # Restore x
                        add_fast(x, selected, sum_counts)
                    
                    if improved:
                        break
            
            if not improved:
                no_improvement += 1
                # Random restart? Or just break?
                # Let's try a few random restarts if stuck
                if no_improvement % 20 == 0:
                    # Do a random restart of local search?
                    # Or just accept current best and break?
                    # Let's try to find a different local minimum by randomizing the removal order
                    pass
        
        return best_selected

    # Run multiple random greedy starts and keep the best
    best_overall = None
    best_overall_size = float('inf')
    
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        selected, sum_counts = random_greedy()
        
        # Local search
        refined = local_search(selected, sum_counts)
        
        if len(refined) < best_overall_size:
            best_overall_size = len(refined)
            best_overall = refined
            
        if best_overall_size <= 10: # Heuristic lower bound?
            break

    # Final validation
    if best_overall is None:
        # Fallback to a single greedy
        selected, _ = random_greedy()
        best_overall = selected

    # Double check validity and maximality
    def final_check(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        
        # Check maximality
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            if 2*x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if valid:
                return False
        return True

    if not final_check(best_overall):
        # This should not happen, but if it does, try to repair or return a known valid set
        # For now, return the greedy result which is always valid and maximal
        selected, _ = random_greedy()
        best_overall = selected

    return best_overall
