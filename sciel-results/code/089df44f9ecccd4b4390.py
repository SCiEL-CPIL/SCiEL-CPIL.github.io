import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 50  # safety margin before 60s kill
    
    def greedy_topdown(elements):
        """Greedy from top down, avoiding subset sum 200."""
        selected, reachable = [], {0}
        for x in sorted(elements, reverse=True):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    def greedy_bottomup(elements):
        """Greedy from bottom up, avoiding subset sum 200."""
        selected, reachable = [], {0}
        for x in sorted(elements):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    def greedy_priority(elements):
        """
        Priority-based greedy: process elements in order of 'difficulty' to combine.
        Larger elements are harder to combine into 200, so try them first.
        But also consider that elements near 200/2 = 100 are critical.
        Use a heuristic: sort by (200 - 2*x) descending, then x descending.
        """
        # Elements closer to 100 are more constrained
        # Sort: prefer elements that are harder to pair with others to make 200
        # An element x is 'hard' if few other elements y satisfy x+y=200
        # For x in [L,100], y = 200-x must also be in [L,100] and y != x
        # So x is hard if 200-x is NOT in the set or 200-x == x
        
        def difficulty(x):
            y = 200 - x
            if y == x:
                return -1  # self-complement, very hard to use
            if y < L or y > 100:
                return 1  # no complement in range, easy to use
            return 0  # has complement, harder
        
        sorted_elems = sorted(elements, key=lambda x: (-difficulty(x), -x))
        selected, reachable = [], {0}
        for x in sorted_elems:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    def greedy_with_lookahead(elements, look_ahead=3):
        """
        Greedy with limited lookahead: for each position, try each remaining element
        and pick the one that leads to the best local outcome.
        This is expensive, so limit the search depth.
        """
        remaining = list(elements)
        selected = []
        reachable = {0}
        
        while remaining:
            if time.time() - start_time > timeout - 5:
                break
            
            best_choice = None
            best_score = -1
            best_reachable = None
            best_selected = None
            
            # Try each remaining element as next choice
            for i, x in enumerate(remaining):
                new_reachable = {s + x for s in reachable if s + x <= 200}
                if 200 in new_reachable:
                    continue
                
                # Score: how many more elements can we potentially add?
                # Approximate by counting how many remaining elements can be added
                score = 0
                temp_reachable = new_reachable
                for j, y in enumerate(remaining):
                    if j == i:
                        continue
                    new2 = {s + y for s in temp_reachable if s + y <= 200}
                    if 200 not in new2:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_choice = x
                    best_reachable = new_reachable
                    best_selected = remaining[:i] + remaining[i+1:]
            
            if best_choice is None:
                break
            
            selected.append(best_choice)
            reachable = best_reachable
            remaining = best_selected
        
        return selected
    
    def ilp_solve(elements, time_limit=15):
        """Solve the ILP for a given set of elements."""
        n = len(elements)
        if n == 0:
            return []
        
        L = min(elements)
        idx_map = {val: i for i, val in enumerate(elements)}
        
        # Generate constraints: subsets summing to 200
        max_k = 1
        while max_k < n:
            min_sum = max_k * L + max_k * (max_k - 1) // 2
            if min_sum > 200:
                max_k -= 1
                break
            max_k += 1
        
        constraints = []
        
        for k in range(3, max_k + 1):
            for comb in itertools.combinations(elements, k):
                if sum(comb) == 200:
                    row = [0] * n
                    for val in comb:
                        row[idx_map[val]] = 1
                    constraints.append(row)
            
            if time.time() - start_time > timeout - 5:
                break
        
        if not constraints:
            return list(elements)
        
        c = -np.ones(n)
        A_ub = np.array(constraints)
        b_ub = np.ones(len(constraints)) * 2
        
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
                      integrality=integrality, bounds=bounds,
                      options={"time_limit": time_limit})
            
            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                return [elements[i] for i in selected_indices]
            else:
                return None
        except Exception:
            return None
    
    # Try multiple L values and multiple greedy strategies
    best_result = []
    best_card = 0
    
    # First, get a baseline from full-range greedy
    full_range = list(range(1, 101))
    baseline = greedy_topdown(full_range)
    best_result = baseline
    best_card = len(baseline)
    
    # Try different L values
    for L in [40, 50, 60, 70, 80, 90, 95]:
        if time.time() - start_time > timeout - 10:
            break
        
        elements = list(range(L, 101))
        n = len(elements)
        
        if n <= best_card:
            continue
        
        # Try multiple greedy strategies
        for strategy_name, strategy_func in [
            ("topdown", greedy_topdown),
            ("bottomup", greedy_bottomup),
            ("priority", greedy_priority),
        ]:
            if time.time() - start_time > timeout - 10:
                break
            
            result = strategy_func(elements)
            if len(result) > best_card:
                best_result = result
                best_card = len(result)
        
        # Try ILP if we have time
        if time.time() - start_time < timeout - 20 and n <= 61:
            ilp_result = ilp_solve(elements, time_limit=10)
            if ilp_result is not None and len(ilp_result) > best_card:
                best_result = ilp_result
                best_card = len(ilp_result)
    
    # Try lookahead greedy on the best-performing L range
    if time.time() - start_time < timeout - 15:
        # Use the L that gave the best result so far
        best_L = 40
        for L in [40, 50, 60, 70, 80, 90, 95]:
            elements = list(range(L, 101))
            if len(elements) > best_card:
                best_L = L
                break
        
        elements = list(range(best_L, 101))
        if len(elements) > best_card:
            lookahead_result = greedy_with_lookahead(elements)
            if len(lookahead_result) > best_card:
                best_result = lookahead_result
                best_card = len(lookahead_result)
    
    # Final safety: ensure we return something valid and nonempty
    if not best_result:
        best_result = greedy_topdown(list(range(1, 101)))
    
    return best_result
