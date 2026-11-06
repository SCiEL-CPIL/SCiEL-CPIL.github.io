import time
import random
import sys
import heapq


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    base = [1, 2, 5, 11]
    best_size = len(base)
    best_solution = base[:]

    def valid_and_sums(lst):
        n = len(lst)
        s = set()
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                b = lst[j]
                v = a + b
                if v in s:
                    return False, None
                s.add(v)
        return True, s

    def greedy_extend(lst, order, max_iter=None):
        cur = lst[:]
        sums = set()
        n = len(cur)
        for i in range(n):
            a = cur[i]
            for j in range(i, n):
                sums.add(a + cur[j])
        used = set(cur)
        it = 0
        for x in order:
            if max_iter is not None and it >= max_iter:
                break
            it += 1
            if x in used:
                continue
            new_ok = True
            for y in cur:
                if x + y in sums:
                    new_ok = False
                    break
            if new_ok and 2 * x in sums:
                new_ok = False
            if new_ok:
                for y in cur:
                    sums.add(x + y)
                sums.add(2 * x)
                cur.append(x)
                used.add(x)
        cur.sort()
        return cur

    def kick_and_grow(lst):
        n = len(lst)
        non_base = [x for x in lst if x not in (1, 2, 5, 11)]
        if not non_base:
            return lst
        k = min(3, len(non_base))
        to_remove = random.sample(non_base, k)
        removed = set(to_remove)
        kept = [x for x in lst if x not in removed]
        cands = [x for x in range(1, 285) if x not in set(kept)]
        random.shuffle(cands)
        grown = greedy_extend(kept, cands)
        return grown

    # Phase 1: randomized greedy constructions from scratch
    base_set = set(base)
    for trial in range(400):
        if time.time() > deadline:
            break
        cands = [x for x in range(1, 285) if x not in base_set]
        random.shuffle(cands)
        built = greedy_extend(base[:], cands)
        if len(built) > best_size:
            best_size = len(built)
            best_solution = built[:]

    # Phase 2: CSP backtracking with forward checking and MRV
    # We try to extend the current best solution to see if we can do better.
    # The CSP variables are the candidate elements (1..284 not in base).
    # A variable is "assigned" if we decide to include it.
    # Constraints: all pairwise sums must be distinct.
    
    def csp_extend(initial_set, target_size, time_limit):
        """
        Try to extend initial_set to reach target_size using CSP backtracking.
        Returns a list of size target_size if successful, None otherwise.
        """
        if time.time() > time_limit:
            return None
        
        initial_set = sorted(initial_set)
        current_sums = set()
        n = len(initial_set)
        for i in range(n):
            a = initial_set[i]
            for j in range(i, n):
                current_sums.add(a + initial_set[j])
        
        used = set(initial_set)
        candidates = [x for x in range(1, 285) if x not in used]
        
        # For each candidate, precompute which existing elements it conflicts with
        # via sum collision. We'll use forward checking dynamically.
        
        def backtrack(current_list, current_sums_set, used_set, remaining_to_add):
            if time.time() > time_limit:
                return None
            
            if remaining_to_add == 0:
                return current_list[:]
            
            # MRV heuristic: for each candidate, count how many "future" candidates
            # it would block. Pick the one with minimum remaining values (most constrained).
            # Actually, for maximization, we want to pick the element that is "easiest" 
            # to place first? No, MRV usually picks the most constrained variable.
            # In this context, adding an element removes other elements from consideration 
            # if they conflict. So an element that conflicts with many others is "more constrained" 
            # in the sense that placing it reduces options more. 
            # However, we want to find *any* valid extension of size target_size.
            # Let's use a heuristic: pick the candidate that has the fewest conflicts 
            # with other *unplaced* candidates? Or the one with the most conflicts?
            
            # Let's try: pick the candidate that is compatible with the current set.
            # Among compatible candidates, pick one that minimizes the number of 
            # *other* compatible candidates it would block?
            
            compatible = []
            for x in candidates:
                if x in used_set:
                    continue
                # Check if x can be added to current_list
                ok = True
                for y in current_list:
                    if x + y in current_sums_set:
                        ok = False
                        break
                if ok and 2 * x in current_sums_set:
                    ok = False
                if ok:
                    compatible.append(x)
            
            if not compatible:
                return None
            
            # Heuristic: sort compatible by how many other compatible elements they conflict with
            # We want to place elements that don't block too many others? 
            # Or place elements that block many others (to prune early)?
            # Let's try placing the one that blocks the fewest others first (greedy-like),
            # but backtrack if it fails.
            
            # Calculate "conflict count" for each compatible x:
            # How many other compatible y would become incompatible if we add x?
            # This is expensive. Let's just use a simple heuristic: 
            # Randomize or sort by value?
            
            # Let's try MRV properly:
            # The "domain" of the problem is the set of possible extensions.
            # This is complex. Let's stick to a simpler randomized backtracking 
            # with forward checking (pruning incompatible candidates early).
            
            # Shuffle compatible to randomize order
            random.shuffle(compatible)
            
            for x in compatible:
                # Try adding x
                new_list = current_list + [x]
                new_used = used_set | {x}
                
                # Compute new sums
                new_sums = current_sums_set.copy()
                for y in current_list:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                # Forward checking: ensure that for all remaining candidates, 
                # they are still compatible with the new state?
                # Actually, we just recurse. The recursion will check compatibility.
                
                result = backtrack(new_list, new_sums, new_used, remaining_to_add - 1)
                if result is not None:
                    return result
            
            return None
        
        # Start backtracking
        # We need to add (target_size - len(initial_set)) more elements
        to_add = target_size - len(initial_set)
        if to_add <= 0:
            return initial_set
        
        result = backtrack(initial_set, current_sums, used, to_add)
        return result

    # Use CSP to try to improve the best solution found so far
    # Try to extend best_solution to size 20, 19, etc.
    target_sizes = list(range(20, best_size, -1))
    for target in target_sizes:
        if time.time() > deadline - 5.0:
            break
        # Try a few random initial subsets of the best solution to break symmetry
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            # Start from best_solution, maybe remove some random non-base elements
            # to give CSP more room? Or just try to extend directly.
            # Let's try extending directly first.
            result = csp_extend(best_solution, target, deadline - 5.0)
            if result is not None:
                if len(result) > best_size:
                    best_size = len(result)
                    best_solution = result[:]
                break
            # If direct extension failed, try removing 1-2 random non-base elements
            # and then extending
            non_base_in_best = [x for x in best_solution if x not in base]
            if non_base_in_best:
                # Remove 1 random non-base element
                to_remove = random.choice(non_base_in_best)
                reduced = [x for x in best_solution if x != to_remove]
                result = csp_extend(reduced, target, deadline - 5.0)
                if result is not None:
                    if len(result) > best_size:
                        best_size = len(result)
                        best_solution = result[:]
                    break

    # Phase 3: local search (kick + grow) around the best
    cur = best_solution[:]
    cur_size = len(cur)
    no_improve = 0
    while time.time() < deadline - 5.0:
        grown = kick_and_grow(cur)
        if len(grown) > cur_size:
            cur = grown
            cur_size = len(grown)
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_solution = cur[:]
        else:
            no_improve += 1
            if no_improve > 120:
                # occasionally try a fresh random start
                cands = [x for x in range(1, 285) if x not in base_set]
                random.shuffle(cands)
                cur = greedy_extend(base[:], cands)
                cur_size = len(cur)
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
                no_improve = 0

    return best_solution


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
    v, s = valid_and_sums(res)
    print("valid:", v)
