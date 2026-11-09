import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    def is_valid(subset):
        # Check if any subset of 'subset' sums to 200
        # Using DP: reachable[s] is True if some subset sums to s
        reachable = [False] * 201
        reachable[0] = True
        for x in subset:
            # Update reachable sums
            # We only care up to 200
            # Iterate backwards to avoid using x multiple times
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
                    if s + x == 200:
                        return False
        return True

    def find_conflict(subset, x):
        # Returns a subset of 'subset' + [x] that sums to 200, or None
        # We know is_valid(subset + [x]) is False.
        # We want to find WHICH elements cause it.
        # Backtrack DP.
        # For simplicity and speed, we can just return [x] + list(subset) as a "conflict group"
        # and let the local search try removing any element from it.
        # But a tighter conflict set is better for swaps.
        # Let's just use the whole set for now, or try to find a minimal one?
        # Minimal subset sum is NP-hard, but 200 is small.
        # Let's try to find a small subset summing to 200 from subset + [x].
        # Since x is the new element, the conflict must involve x.
        # So we need a subset of 'subset' summing to 200 - x.
        target = 200 - x
        if target < 0:
            return None # Should not happen if is_valid failed due to x
        
        # Find subset of 'subset' summing to target
        # DP to find one such subset
        # dp[s] = list of elements used to reach s, or None
        # To save memory/time, we just store the previous state.
        # Given small target, we can do this.
        
        # Actually, we can just return the whole 'subset' + [x] as the conflict.
        # The local search will try removing any y from 'subset'.
        # This is effective enough.
        return list(subset) + [x]

    def greedy_build(order):
        selected = []
        for x in order:
            if is_valid(selected + [x]):
                selected.append(x)
        return selected

    def local_search(initial_set):
        S = set(initial_set)
        U = set(range(1, 101)) - S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add elements
            for x in list(U):
                if time.time() > deadline:
                    return S
                if is_valid(list(S) + [x]):
                    S.add(x)
                    U.remove(x)
                    improved = True
            
            # Phase 2: Try swaps if no additions were made
            if not improved:
                swap_made = False
                # Randomize order of U to escape local minima
                u_list = list(U)
                random.shuffle(u_list)
                for x in u_list:
                    if time.time() > deadline:
                        return S
                    
                    # Check if adding x causes conflict
                    # We know it does, otherwise it would have been added in Phase 1
                    conflict_set = find_conflict(list(S), x)
                    if not conflict_set:
                        continue
                        
                    # Try removing any y from S that is in the conflict
                    # The conflict is a subset of S U {x} summing to 200.
                    # Actually, find_conflict returns S + [x].
                    # We need to remove a y from S such that (S - {y}) + {x} is valid.
                    # To be safe, we should only consider y that is part of the actual conflict.
                    # But we don't have the minimal conflict.
                    # Let's try removing each y in S.
                    s_list = list(S)
                    random.shuffle(s_list)
                    for y in s_list:
                        new_S = (S - {y}) | {x}
                        if is_valid(list(new_S)):
                            S = new_S
                            U = set(range(1, 101)) - S
                            swap_made = True
                            improved = True
                            break
                    if improved:
                        break
        return S

    best_set = set()
    
    # Run multiple random restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            # First run: Greedy descending (like seed)
            order = list(range(100, 0, -1))
        else:
            # Subsequent runs: Randomized greedy
            order = list(range(1, 101))
            random.shuffle(order)
            # Bias towards larger numbers? 
            # Sorting by random weight?
            # Let's just use pure random, or weighted.
            # Larger numbers are "safer" (fewer combinations sum to 200).
            # Let's mix: 50% random, 50% descending.
            if random.random() < 0.5:
                pass
            else:
                order = list(range(100, 0, -1))
                # Perturb
                for _ in range(10):
                    if len(order) > 1:
                        idx = random.randint(0, len(order)-1)
                        jdx = random.randint(0, len(order)-1)
                        order[idx], order[jdx] = order[jdx], order[idx]
        
        current_set = greedy_build(order)
        current_set = local_search(current_set)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            # If we hit a very high score, we might stop early?
            # Max possible is 100.
            # If we get > 80, it's very good.
            
        if time.time() > deadline - 5.0:
            break

    return list(best_set)
