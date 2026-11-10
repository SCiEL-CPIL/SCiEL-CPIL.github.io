# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Uses a combination of:
    1. Multiple greedy orderings to get good initial solutions
    2. Local improvement (add/remove/swap) to refine
    3. Randomized restarts with time budget
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, leave safety margin
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute conflict graph: conflict[i][j] = True if (i-j) % m in forbidden
    # This is symmetric, so we can use a set of frozensets or just check on the fly
    # For efficiency, precompute which residues are forbidden
    forbidden_set = set(forbidden)
    
    def is_conflict(a, b):
        d = (a - b) % m
        return d in forbidden_set
    
    def greedy_fill(order, initial=None):
        """Greedy fill using a given ordering of candidates."""
        selected = list(initial) if initial else []
        selected_set = set(selected)
        for candidate in order:
            if candidate in selected_set:
                continue
            ok = True
            for s in selected:
                if is_conflict(candidate, s):
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected
    
    def local_improve(selected, max_iters=1000):
        """Try to improve the set by adding, removing, or swapping elements."""
        selected = list(selected)
        selected_set = set(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding elements
            for cand in range(m):
                if cand in selected_set:
                    continue
                ok = True
                for s in selected:
                    if is_conflict(cand, s):
                        ok = False
                        break
                if ok:
                    selected.append(cand)
                    selected_set.add(cand)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add two
            # For efficiency, try removing each element and see if we can add more
            for i in range(len(selected)):
                removed = selected[i]
                temp_selected = selected[:i] + selected[i+1:]
                temp_set = set(temp_selected)
                
                # Try to add as many elements as possible after removal
                added_count = 0
                for cand in range(m):
                    if cand in temp_set:
                        continue
                    ok = True
                    for s in temp_selected:
                        if is_conflict(cand, s):
                            ok = False
                            break
                    if ok:
                        temp_selected.append(cand)
                        temp_set.add(cand)
                        added_count += 1
                
                if added_count > 1:
                    selected = temp_selected
                    selected_set = temp_set
                    improved = True
                    break
        
        return selected
    
    # Generate different orderings
    def natural_order():
        return list(range(m))
    
    def reverse_order():
        return list(range(m-1, -1, -1))
    
    def conflict_degree_order():
        """Order by degree in conflict graph (lowest first)."""
        degrees = []
        for i in range(m):
            deg = 0
            for j in range(m):
                if i != j and is_conflict(i, j):
                    deg += 1
            degrees.append((deg, i))
        degrees.sort()
        return [d[1] for d in degrees]
    
    def golden_ratio_shuffle(seed=0):
        """Deterministic shuffle using golden ratio."""
        phi = 0.6180339887498949
        arr = list(range(m))
        for i in range(m-1, 0, -1):
            j = int((i * phi + seed) % (i+1))
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    def random_shuffle(seed):
        """Random shuffle with given seed."""
        arr = list(range(m))
        import random
        rng = random.Random(seed)
        rng.shuffle(arr)
        return arr
    
    # Get initial solutions from different orderings
    best_solution = []
    best_size = 0
    
    orderings = [
        natural_order(),
        reverse_order(),
        conflict_degree_order(),
    ]
    
    # Add some golden ratio shuffles
    for s in range(10):
        orderings.append(golden_ratio_shuffle(s))
    
    # Add some random shuffles
    for s in range(20):
        orderings.append(random_shuffle(s))
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        sol = greedy_fill(order)
        sol = local_improve(sol, max_iters=200)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
    
    # More random restarts with local improvement
    import random
    rng = random.Random(42)
    
    while time.time() < deadline - 60:
        # Random ordering
        order = list(range(m))
        rng.shuffle(order)
        sol = greedy_fill(order)
        sol = local_improve(sol, max_iters=500)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
    
    # Final local improvement on best solution
    if time.time() < deadline - 30:
        best_solution = local_improve(best_solution, max_iters=2000)
    
    return np.array(best_solution, dtype=np.int64)


# EVOLVE-BLOCK-END
