"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    
    def _validate_sidon(S):
        """Check if S is a Sidon set (unique pair sums including a+a)."""
        if not S:
            return True
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _diffs_of(S):
        """Get nonzero differences of S."""
        d = set()
        n = len(S)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d.add(S[j] - S[i])
        return d
    
    def _can_add(x, S, set_sums, set_diffs, other_diffs):
        """Check if x can be added to set S with given sums/diffs."""
        # Check Sidon condition
        for a in S:
            if (a + x) in set_sums:
                return False
        if (x + x) in set_sums:
            return False
        
        # Check difference disjointness
        for a in S:
            if (x - a) in other_diffs or (a - x) in other_diffs:
                return False
        return True
    
    def _greedy_one_ordering(order, deadline, strategy="first_fit"):
        """Run greedy construction with a given ordering of candidates."""
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            if strategy == "first_fit":
                sides = [0, 1]
            elif strategy == "best_fit":
                # Prefer the side with more room (fewer conflicts)
                conflicts = []
                for side in (0, 1):
                    S = sets[side]
                    other_diffs = set_diffs[1 - side]
                    conf = 0
                    for a in S:
                        if (a + x) in set_sums[side] or (x - a) in other_diffs or (a - x) in other_diffs:
                            conf += 1
                    conflicts.append(conf)
                sides = [0, 1] if conflicts[0] <= conflicts[1] else [1, 0]
            else:  # random
                sides = [0, 1]
                random.shuffle(sides)
            
            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if not _can_add(x, S, set_sums[side], set_diffs[side], other_diffs):
                    continue
                
                # Accept x into sets[side]
                sets[side].append(x)
                for a in S:
                    set_sums[side].add(a + x)
                    set_diffs[side].add(x - a)
                    set_diffs[side].add(a - x)
                set_sums[side].add(x + x)
                break
        
        return sets
    
    def _local_search(sets, deadline, max_iter=200):
        """Local search: try to add more elements or swap to improve size."""
        sets = [list(sets[0]), list(sets[1])]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        # Rebuild sums and diffs
        for side in (0, 1):
            S = sets[side]
            for i in range(len(S)):
                for j in range(i, len(S)):
                    set_sums[side].add(S[i] + S[j])
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        set_diffs[side].add(S[j] - S[i])
        
        current_size = len(sets[0]) + len(sets[1])
        best_size_ls = current_size
        best_sets_ls = [list(sets[0]), list(sets[1])]
        
        # Try to add elements not in either set
        all_used = set(sets[0]) | set(sets[1])
        candidates = [x for x in range(1, 257) if x not in all_used]
        random.shuffle(candidates)
        
        iter_count = 0
        while iter_count < max_iter and time.time() < deadline - 1.0:
            iter_count += 1
            improved = False
            
            for x in candidates:
                if time.time() > deadline - 1.0:
                    break
                
                for side in (0, 1):
                    if _can_add(x, sets[side], set_sums[side], set_diffs[side], set_diffs[1 - side]):
                        # Add it
                        S = sets[side]
                        for a in S:
                            set_sums[side].add(a + x)
                            set_diffs[side].add(x - a)
                            set_diffs[side].add(a - x)
                        set_sums[side].add(x + x)
                        sets[side].append(x)
                        candidates.remove(x)
                        
                        new_size = len(sets[0]) + len(sets[1])
                        if new_size > best_size_ls:
                            best_size_ls = new_size
                            best_sets_ls = [list(sets[0]), list(sets[1])]
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try swapping: remove an element from one set and try to add to the other
                swapped = False
                for side in (0, 1):
                    if len(sets[side]) > 1:
                        for idx in range(len(sets[side])):
                            x = sets[side][idx]
                            # Remove x from sets[side]
                            sets[side].pop(idx)
                            
                            # Rebuild sums and diffs for side
                            S = sets[side]
                            new_sums = set()
                            for i in range(len(S)):
                                for j in range(i, len(S)):
                                    new_sums.add(S[i] + S[j])
                            new_diffs = set()
                            for i in range(len(S)):
                                for j in range(len(S)):
                                    if i != j:
                                        new_diffs.add(S[j] - S[i])
                            set_sums[side] = new_sums
                            set_diffs[side] = new_diffs
                            
                            # Try to add x to other side
                            other = 1 - side
                            if _can_add(x, sets[other], set_sums[other], set_diffs[other], set_diffs[side]):
                                S2 = sets[other]
                                for a in S2:
                                    set_sums[other].add(a + x)
                                    set_diffs[other].add(x - a)
                                    set_diffs[other].add(a - x)
                                set_sums[other].add(x + x)
                                sets[other].append(x)
                                swapped = True
                                break
                            
                            # If didn't work, put x back
                            if not swapped:
                                sets[side].append(x)
                                S = sets[side]
                                for i in range(len(S)):
                                    for j in range(i, len(S)):
                                        set_sums[side].add(S[i] + S[j])
                                for i in range(len(S)):
                                    for j in range(len(S)):
                                        if i != j:
                                            set_diffs[side].add(S[j] - S[i])
                        
                        if swapped:
                            break
                
                if not swapped and not improved:
                    break
        
        return best_sets_ls
    
    # Try multiple strategies and orderings
    random.seed(42)
    base_order = list(range(1, 257))
    
    # Strategy 1: Natural order with first-fit
    sets = _greedy_one_ordering(base_order, deadline, "first_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets
    
    # Strategy 2: Natural order with best-fit
    sets = _greedy_one_ordering(base_order, deadline, "best_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets
    
    # Strategy 3: Descending order
    desc_order = list(range(256, 0, -1))
    sets = _greedy_one_ordering(desc_order, deadline, "first_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets
    
    # Strategy 4: Random permutations with different strategies
    strategies = ["first_fit", "best_fit", "random"]
    num_trials = 0
    while time.time() < deadline - 5.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        strat = strategies[num_trials % len(strategies)]
        
        sets = _greedy_one_ordering(random_order, deadline, strat)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = sets
        
        num_trials += 1
    
    # Local search polish on the best solution
    if best_size > 0 and time.time() < deadline - 3.0:
        polished = _local_search(best_sets, deadline, max_iter=500)
        polished_size = len(polished[0]) + len(polished[1])
        if polished_size > best_size:
            best_size = polished_size
            best_sets = polished
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
