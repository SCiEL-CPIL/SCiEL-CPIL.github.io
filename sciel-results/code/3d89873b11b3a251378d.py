import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 128
    TARGETS = list(range(128, 257))  # 129 targets
    TARGETS_SET = set(TARGETS)
    NUM_TARGETS = 129
    
    # Precompute for fast multiplicity calculation
    # We represent A as a boolean array of size 129 (indices 0..128)
    # But for set operations in local search, we use Python sets for clarity and speed on small sets.
    
    def compute_mult(A_set):
        """Return max multiplicity over targets in [128,256]. Returns inf if not covered."""
        max_mult = 0
        for t in TARGETS:
            count = 0
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    count += 1
            if count == 0:
                return float('inf')
            if count > max_mult:
                max_mult = count
        return max_mult

    def is_complete(A_set):
        """Check if all targets are covered."""
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True

    def get_missing_targets(A_set):
        missing = []
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                missing.append(t)
        return missing

    def local_search(A_set, max_iters=3000, time_limit=None):
        """
        Local search to minimize max multiplicity.
        Moves:
        1. Remove element (if coverage maintained and mult doesn't increase)
        2. Swap element (if coverage maintained and mult decreases)
        3. Add element (if it reduces max mult by balancing? Unlikely, but maybe helps escape?)
           Actually, adding increases total pairs. It can only increase or keep max mult.
           So we primarily rely on removal and swaps.
        """
        if time_limit is None:
            time_limit = deadline - 2  # safety
        
        current_mult = compute_mult(A_set)
        if current_mult == float('inf'):
            return A_set, current_mult
            
        # Sort A_set for deterministic iteration? No, random is better for search.
        
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
                
            improved = False
            
            # 1. Try removing an element
            # We want to remove an element that is "least useful" or causes high multiplicity.
            # If we remove 'a', we lose all pairs involving 'a'.
            # This reduces multiplicity for targets t where a+b=t.
            # It might break coverage if 'a' was the only partner for some target.
            
            # Strategy: Try removing each element, check if still complete. If yes, compute new mult.
            # To speed up, we can first check if removal breaks coverage.
            
            elements_to_try = list(A_set)
            random.shuffle(elements_to_try)
            
            for a in elements_to_try:
                if time.time() > time_limit:
                    break
                new_set = A_set - {a}
                if not new_set:
                    continue
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
                    break
                elif new_mult == current_mult:
                    # If mult is same, prefer smaller set? It might allow future removals.
                    # But removing might make it harder to cover if we add back later?
                    # Let's accept removal if mult is same to reduce set size (less "clutter").
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swapping an element
            # Pick a random element 'a' in A, and a random 'b' not in A.
            # Replace a with b.
            if len(A_set) > 0:
                a_in = random.choice(list(A_set))
                # Choose a_out from [0, 128] \ A_set
                # To bias towards useful swaps, we could pick a_out such that it covers a "hot" target?
                # For now, random.
                a_out = random.randint(0, N)
                if a_out in A_set:
                    continue
                new_set = A_set - {a_in} | {a_out}
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
            
            if not improved:
                # Stuck. Try a "kick" or random restart of local search?
                # Or try a swap that doesn't improve mult but keeps it same, to escape plateau?
                escaped = False
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    a_in = random.choice(list(A_set))
                    a_out = random.randint(0, N)
                    if a_out in A_set:
                        continue
                    new_set = A_set - {a_in} | {a_out}
                    if not is_complete(new_set):
                        continue
                    new_mult = compute_mult(new_set)
                    if new_mult <= current_mult:
                        A_set = new_set
                        current_mult = new_mult
                        escaped = True
                        break
                if not escaped:
                    # If we can't even keep mult same, break
                    break
                    
        return A_set, current_mult

    def greedy_complete(A_set, time_limit):
        """
        Greedily add elements to A_set until all targets are covered.
        """
        while time.time() < time_limit:
            missing = get_missing_targets(A_set)
            if not missing:
                break
            
            t = missing[0]
            # Find x, y such that x+y=t, and adding x (and possibly y) covers t and other missing targets.
            # We can add x if y is already in A_set.
            # Or add both x and y.
            
            best_x = None
            best_y = None
            best_gain = 0
            
            # Try adding a single element x such that t-x is in A_set
            for x in range(0, N+1):
                if x in A_set:
                    continue
                y = t - x
                if y < 0 or y > N:
                    continue
                if y in A_set:
                    # Adding x covers t.
                    # Check how many other missing targets this x covers.
                    gain = 1
                    for t2 in missing[1:]:
                        y2 = t2 - x
                        if 0 <= y2 <= N and y2 in A_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = None # y is already in set
            
            if best_x is not None:
                A_set.add(best_x)
                continue
            
            # If no single element covers t, we must add a pair (x, y) such that x+y=t.
            # To minimize size increase, we pick a pair that covers the most other missing targets.
            best_pair_gain = 0
            best_x_pair = None
            best_y_pair = None
            
            for x in range(0, N+1):
                y = t - x
                if y < 0 or y > N:
                    continue
                if x > y:
                    continue # avoid duplicates
                # If we add x and y, we cover t.
                # Check other missing targets.
                gain = 1
                for t2 in missing[1:]:
                    # Covered if t2-x in A_set OR t2-y in A_set (or t2-x == y or t2-y == x)
                    # Actually, if we add x and y to A_set, then for a target t2:
                    # It is covered if there exists a in A_set U {x,y} such that t2-a in A_set U {x,y} and a <= t2-a.
                    # This is complex to check quickly.
                    # Approximation: Check if t2-x is in A_set or t2-y is in A_set.
                    # Or if t2-x == y (which implies t2 = x+y=t, already counted) 
                    # Or if t2-y == x.
                    
                    # Let's just count how many missing targets t2 have t2-x in A_set or t2-y in A_set.
                    # This is a lower bound on coverage.
                    if (0 <= t2-x <= N and (t2-x in A_set or t2-x == y)) or \
                       (0 <= t2-y <= N and (t2-y in A_set or t2-y == x)):
                        gain += 1
                        
                if gain > best_pair_gain:
                    best_pair_gain = gain
                    best_x_pair = x
                    best_y_pair = y
            
            if best_x_pair is not None:
                A_set.add(best_x_pair)
                if best_y_pair != best_x_pair:
                    A_set.add(best_y_pair)
            else:
                # Should not happen if N is large enough, but just in case
                break
                
        return A_set

    best_A = None
    best_mult = float('inf')
    
    # Strategy:
    # 1. Generate diverse random seeds of small size (k=15..25).
    # 2. Complete them greedily.
    # 3. Run local search.
    # 4. Keep the best.
    
    # Also try some structured seeds?
    # e.g. A = {0, 1, ..., k-1} U {N-k+1, ..., N}?
    # But random with good local search is often better for this size.
    
    num_seeds = 0
    while time.time() < deadline - 15:
        num_seeds += 1
        
        # Random size
        k = random.randint(15, 22)
        
        # Random initial set
        # To improve coverage chances, maybe bias towards spread out numbers?
        # Uniform random is okay.
        A_set = set(random.sample(range(0, N+1), k))
        
        # Greedy completion
        A_set = greedy_complete(A_set, deadline - 15)
        
        if not is_complete(A_set):
            continue
            
        # Local search
        A_set, mult = local_search(A_set, max_iters=2000, time_limit=deadline - 5)
        
        if mult < best_mult:
            best_mult = mult
            best_A = set(A_set)
            
        # If we found a very good solution, we can try to polish it more in subsequent iterations?
        # Or just keep generating new seeds.
        # If best_mult <= 4, it's hard to beat. Let's keep searching for 3.
        
        # Optional: If we have time, run local search on the best_A again with more iterations?
        # This might help escape local minima.
        if num_seeds % 10 == 0 and time.time() < deadline - 10:
            # Polish the best solution found so far
            if best_A is not None:
                # Make a copy
                polished_A = set(best_A)
                polished_mult = compute_mult(polished_A)
                polished_A, polished_mult = local_search(polished_A, max_iters=5000, time_limit=deadline - 5)
                if polished_mult < best_mult:
                    best_mult = polished_mult
                    best_A = polished_A

    # Fallback if nothing found (should not happen)
    if best_A is None:
        A_set = set(range(0, N+1, 2)) | {1}
        A_set = greedy_complete(A_set, deadline - 5)
        A_set, mult = local_search(A_set, max_iters=1000, time_limit=deadline - 2)
        best_A = A_set
        best_mult = mult

    return list(best_A)
