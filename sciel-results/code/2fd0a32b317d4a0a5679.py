import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict sets: conflict_sets[i] = set of j such that (i*j)%(i+j)==0
    conflict_sets = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)

    def is_valid(S):
        """Check if set S is valid (all distinct pairs compatible)."""
        S_list = list(S)
        for idx, a in enumerate(S_list):
            for b in S_list[idx+1:]:
                if b in conflict_sets[a]:
                    return False
        return True

    def greedy_min_degree(remaining, selected, rng):
        """
        Greedy minimum-degree independent set:
        At each step, pick the vertex with minimum degree in the remaining graph,
        add it to selected, and remove it and its neighbors from remaining.
        """
        selected = set(selected)
        remaining = set(remaining)
        
        while remaining:
            # Find vertex with minimum degree within remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflict_sets[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            selected.add(best_v)
            # Remove best_v and all its neighbors from remaining
            neighbors = conflict_sets[best_v]
            remaining.discard(best_v)
            for n in neighbors:
                remaining.discard(n)
        
        return selected

    def greedy_max_independence(remaining, selected, rng):
        """
        Alternative greedy: pick vertices that maximize the number of remaining vertices
        they don't conflict with (i.e., minimize degree in remaining).
        This is similar to min-degree but with a slight heuristic variation.
        """
        selected = set(selected)
        remaining = set(remaining)
        
        while remaining:
            # Find vertex with minimum degree in remaining
            candidates = list(remaining)
            rng.shuffle(candidates)
            
            min_deg = float('inf')
            best_v = None
            for v in candidates:
                deg = len(conflict_sets[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                elif deg == min_deg:
                    # Tie-break randomly
                    if rng.random() < 0.5:
                        best_v = v
            
            if best_v is None:
                break
            
            selected.add(best_v)
            neighbors = conflict_sets[best_v]
            remaining.discard(best_v)
            for n in neighbors:
                remaining.discard(n)
        
        return selected

    def local_search_swap(S, rng, max_iters=100):
        """
        Local search: try to swap out an element for another that increases size.
        """
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            S_list = list(S)
            if not S_list:
                break
            
            remove_elem = rng.choice(S_list)
            S.remove(remove_elem)

            # Try to add a random element not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                S.add(remove_elem)
                continue

            rng.shuffle(candidates)
            added = False
            for add_elem in candidates[:30]:
                compatible = True
                for s in S:
                    if add_elem in conflict_sets[s]:
                        compatible = False
                        break
                if compatible:
                    S.add(add_elem)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break

            if not added:
                S.add(remove_elem)

        return best_S

    def two_opt_improve(S, rng, max_iters=50):
        """
        Try to improve by removing 2 elements and adding 3 (net +1).
        """
        S = set(S)
        best_S = set(S)
        best_size = len(S)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            S_list = list(S)
            if len(S_list) < 2:
                break

            # Pick two random elements to remove
            pair = rng.sample(S_list, 2)
            S.remove(pair[0])
            S.remove(pair[1])

            # Try to add up to 3 new elements
            candidates = [x for x in range(1, N + 1) if x not in S]
            rng.shuffle(candidates)
            
            added_count = 0
            for add_elem in candidates[:50]:
                if added_count >= 3:
                    break
                compatible = True
                for s in S:
                    if add_elem in conflict_sets[s]:
                        compatible = False
                        break
                if compatible:
                    S.add(add_elem)
                    added_count += 1

            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            else:
                # Revert
                for elem in list(S):
                    if elem not in best_S:
                        S.discard(elem)
                S.update(pair)

        return best_S

    # Main search: multiple trials with different strategies
    best_solution = set()
    best_size = 0

    # Strategy 1: Pure min-degree greedy from scratch
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 104729 + 17)
        remaining = set(range(1, N + 1))
        S = greedy_min_degree(remaining, set(), rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_solution = set(S)
        
        # Local search improvement
        if time.time() < deadline - 5:
            S_ls = local_search_swap(S, rng, max_iters=30)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_solution = set(S_ls)

    # Strategy 2: Start with odds, then add evens via min-degree on remaining
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 31337 + 41)
        S = set(odds)
        # Find evens compatible with all odds
        compatible_evens = [e for e in evens if not (conflict_sets[e] & set(odds))]
        
        if compatible_evens:
            # Add compatible evens greedily
            remaining = set(compatible_evens)
            while remaining:
                min_deg = float('inf')
                best_v = None
                for v in remaining:
                    deg = len(conflict_sets[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    elif deg == min_deg and rng.random() < 0.5:
                        best_v = v
                if best_v is None:
                    break
                S.add(best_v)
                remaining.discard(best_v)
                for n in conflict_sets[best_v]:
                    remaining.discard(n)
        
        if len(S) > best_size:
            best_size = len(S)
            best_solution = set(S)
        
        if time.time() < deadline - 5:
            S_ls = local_search_swap(S, rng, max_iters=30)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_solution = set(S_ls)

    # Strategy 3: Randomized min-degree with different tie-breaking
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 6151 + 7)
        S = greedy_max_independence(set(range(1, N + 1)), set(), rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_solution = set(S)
        
        if time.time() < deadline - 5:
            S_ls = local_search_swap(S, rng, max_iters=30)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_solution = set(S_ls)

    # Final verification and fallback
    if not is_valid(best_solution):
        # Fallback: all odds
        best_solution = set(odds)
        best_size = len(best_solution)

    return list(best_solution)
