"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        # Vectorized check using numpy for speed
        arr = np.array(S)
        # Compute all sums a+b
        # This creates an N x N array. For N=240, that's 57600 elements. Very fast.
        sums = arr[:, None] + arr[None, :]
        # Check if any sum is in S_set
        # We need to check if sums[i,j] == S[k] for any i,j,k
        # A faster way: for each s in S_set, check if s is in sums.
        # Even faster: check if the intersection of sums and S is non-empty.
        # But sums is a 2D array.
        # Let's just iterate over S_set and check if s is in the set of sums.
        # Actually, since N is small (<=240), a simple loop with set lookup is fine.
        # The current O(N^2) loop in Python might be slow if done naively, 
        # but with set lookup it's O(N^2) set lookups which is fast.
        # Let's stick to the simple set-based check for clarity and correctness, 
        # optimized by using a set.
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def max_sum_free(elements, max_time=30.0):
        n = len(elements)
        if n <= 3:
            return list(elements)
        
        # Strategy: Simulated Annealing / Random Restart Local Search
        # We will maintain a set S.
        # Moves:
        # 1. Add a random element from remaining if it doesn't violate sum-free.
        # 2. Remove a random element from S.
        # 3. Swap: Remove one from S, add one from remaining (if valid after removal).
        # Objective: Maximize |S|.
        
        best_S = []
        best_size = 0
        rng = Random(12345)
        
        # Pre-compute conflict graph?
        # Conflict(i, j, k) if elements[i] + elements[j] == elements[k].
        # It's a 3-uniform hypergraph.
        # For local search, we need to know if adding x to S is valid.
        # x is valid if:
        #   - No a, b in S such that a + b = x
        #   - No a in S such that a + x in S
        #   - 2x not in S
        
        # To make this fast, we can maintain:
        # S_set: set of values in S
        # S_indices: list of indices in S (for removal)
        
        # Let's implement a simple hill-climbing with random restarts.
        
        def get_conflicts(x, S_set):
            # Returns list of elements in S that conflict with x
            # Conflict if a+b=x, a+x in S, or 2x in S
            # We just need to know if it's valid.
            # For efficiency in SA, we might want to know which elements to remove.
            pass

        def can_add(x, S_set):
            # Check if x can be added to S_set
            # 1. a + b = x
            #    for a in S_set: if x - a in S_set: return False
            # 2. a + x in S_set
            #    for a in S_set: if a + x in S_set: return False
            # 3. 2x in S_set
            if 2 * x in S_set:
                return False
            for a in S_set:
                if (x - a) in S_set:
                    return False
                if (a + x) in S_set:
                    return False
            return True

        def get_blockers(x, S_set):
            # Return set of elements in S that prevent x from being added
            blockers = set()
            if 2 * x in S_set:
                blockers.add(2 * x)
            for a in S_set:
                # a + b = x => b = x - a
                if (x - a) in S_set:
                    blockers.add(a)
                    blockers.add(x - a)
                # a + x = c => c = a + x
                if (a + x) in S_set:
                    blockers.add(a)
                    blockers.add(a + x)
            return blockers

        # Initial solution: Greedy
        S = []
        S_set = set()
        # Sort by absolute value descending (heuristic)
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        for x in sorted_elems:
            if can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        
        best_size = len(S)
        best_S = S.copy()
        
        current_time = time.time()
        time_limit = current_time + max_time
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        current_S = S.copy()
        current_S_set = S_set.copy()
        current_size = len(current_S)
        
        while time.time() < time_limit:
            # Move selection
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add a random element from remaining
                remaining = [x for x in elements if x not in current_S_set]
                if not remaining:
                    continue
                x = rng.choice(remaining)
                if can_add(x, current_S_set):
                    new_S = current_S + [x]
                    new_S_set = current_S_set | {x}
                    new_size = current_size + 1
                    # Always accept improvement
                    current_S = new_S
                    current_S_set = new_S_set
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = current_S.copy()
                else:
                    # Try to resolve conflicts by removing blockers
                    # Pick a random blocker to remove
                    blockers = get_blockers(x, current_S_set)
                    if blockers:
                        b = rng.choice(list(blockers))
                        # Remove b, add x
                        new_S_set = current_S_set - {b} | {x}
                        new_S = [y for y in current_S if y != b] + [x]
                        # Check if new set is sum-free (it should be if b was the only blocker, 
                        # but removing b might not be enough if x conflicts with others? 
                        # No, get_blockers returns ALL elements involved in conflicts with x.
                        # If we remove one blocker, x might still conflict with others.
                        # So we need to check can_add(x, new_S_set) after removal.
                        # Actually, if we remove b, and b was part of a conflict (a,b,x), 
                        # then that specific conflict is gone. But x might conflict with c,d.
                        # So we must verify.
                        if can_add(x, new_S_set - {x}): # Wait, new_S_set already has x.
                            # Let's re-verify properly.
                            temp_set = current_S_set - {b}
                            if can_add(x, temp_set):
                                current_S_set = temp_set | {x}
                                current_S = [y for y in current_S if y != b] + [x]
                                current_size = len(current_S)
                                if current_size > best_size:
                                    best_size = current_size
                                    best_S = current_S.copy()
                                continue
                        # If adding x after removing b fails, try another blocker? 
                        # For simplicity, let's just try to add x if we can remove a small set of blockers.
                        # This is getting complex. Let's simplify:
                        # Move: Remove a random element from S.
                        pass
            elif move_type < 0.8:
                # Remove a random element
                if current_S:
                    b = rng.choice(current_S)
                    new_S_set = current_S_set - {b}
                    new_S = [y for y in current_S if y != b]
                    new_size = current_size - 1
                    # Accept with probability exp(-delta/T)
                    delta = current_size - new_size # 1
                    if rng.random() < np.exp(-delta / T):
                        current_S = new_S
                        current_S_set = new_S_set
                        current_size = new_size
            else:
                # Swap: Remove one, add one
                if current_S:
                    b = rng.choice(current_S)
                    remaining = [x for x in elements if x not in current_S_set]
                    if remaining:
                        x = rng.choice(remaining)
                        temp_set = current_S_set - {b}
                        if can_add(x, temp_set):
                            new_S_set = temp_set | {x}
                            new_S = [y for y in current_S if y != b] + [x]
                            new_size = current_size # Size stays same
                            # Accept with probability 1 (neutral move) or slightly less
                            current_S = new_S
                            current_S_set = new_S_set
                            # current_size unchanged
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_S

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time
    # Total ~140s.
    # 60: 20s
    # 120: 40s
    # 240: 80s
    # Sum = 140s.
    time_budgets = [20.0, 40.0, 80.0]
    
    for i, (size, seed) in enumerate(instances):
        if time.time() > deadline:
            results.append([])
            continue
            
        elems = instance(size, seed)
        t_start = time.time()
        subset = max_sum_free(elems, max_time=time_budgets[i])
        results.append(subset)
        
        # Check time
        if time.time() > deadline - 5.0:
             # Break if we are running out of time for subsequent instances
             # But we must return 3 lists.
             # If we run out, we return empty for the rest.
             pass

    return results
# EVOLVE-BLOCK-END
