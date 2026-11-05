"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.optimize import linprog

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one_lp(vals, time_limit=10.0):
        """
        Solve the sum-free subset problem using LP relaxation followed by local search.
        """
        n = len(vals)
        if n == 0:
            return []
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # Identify all sum-free violating triples (a, b, c) such that a + b = c
        # Note: a, b, c must be distinct elements from vals? 
        # The problem says "a, b, c in it". It doesn't say distinct.
        # If a+b=c, and a,b,c are in the set, it's a violation.
        # Since vals are distinct, a, b, c are distinct indices unless a=b.
        # If a=b, then 2a=c.
        
        # We need constraints x_a + x_b + x_c <= 2 for all such triples.
        # To keep the LP small, we generate these constraints.
        
        # Efficient triple generation:
        # For each c in vals, check if there exist a, b in vals such that a+b=c.
        # Actually, we just need to add constraints for every valid triple found.
        
        # Precompute a set for fast lookup
        vals_set = set(vals)
        
        # Generate constraints matrix A_ub and b_ub
        # We will store rows as lists of (col, coeff) to build sparse matrix later
        # But scipy linprog expects dense or sparse. For n=240, dense 240x240 is fine.
        # However, number of constraints M can be large.
        
        # Let's estimate M. For random integers, probability of a+b=c is low.
        # But let's be safe.
        
        rows = []
        # We iterate over all pairs (a, b) and check if a+b is in vals.
        # If yes, add constraint x_a + x_b + x_c <= 2.
        # To avoid duplicates, we can just add them all. Redundant constraints are okay for LP.
        
        # Optimization: Only add constraint if a, b, c are distinct?
        # If a=b, constraint is 2x_a + x_c <= 2.
        # If a!=b, constraint is x_a + x_b + x_c <= 2.
        
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                s = a + b
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Constraint: x_a + x_b + x_k <= 2
                    # We store this to build the matrix
                    rows.append((i, j, k))
        
        # If no constraints, any subset is sum-free. Max size is n.
        if not rows:
            return vals[:]
        
        # Build A_ub and b_ub
        # A_ub is M x n
        # b_ub is M
        M = len(rows)
        # For n=240, M might be a few thousand. Dense matrix is fine.
        A_ub = np.zeros((M, n))
        b_ub = np.ones(M) * 2
        
        for idx, (i, j, k) in enumerate(rows):
            A_ub[idx, i] += 1
            A_ub[idx, j] += 1
            A_ub[idx, k] += 1
            
        # Objective: Maximize sum(x_i)
        # linprog minimizes, so minimize -sum(x_i)
        c_obj = -np.ones(n)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * n
        
        # Solve LP
        # Use method 'highs' which is fast and robust
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            # Fallback to greedy if LP fails
            return greedy_solve(vals)
            
        if not res.success:
            return greedy_solve(vals)
            
        # res.x is the fractional solution
        x_sol = res.x
        
        # Convert to 0-1 solution
        # Simple rounding: x_i = 1 if x_i > 0.5 else 0
        # But this might not be sum-free.
        # We use this as a seed for local search.
        
        # Heuristic: Sort by x_sol descending, then greedily add if valid?
        # Or just threshold and fix.
        
        # Let's create a candidate set based on the LP solution.
        # Sort indices by LP value descending.
        sorted_indices = np.argsort(x_sol)[::-1]
        
        S_set = set()
        S_list = []
        
        # Greedy add based on LP priority
        for idx in sorted_indices:
            v = vals[idx]
            if can_add(S_set, v):
                S_set.add(v)
                S_list.append(v)
                
        # Now run local search to improve
        S_final = local_search(S_list, vals, time_limit_left=max(0, time_limit - (time.time() - start_time)))
        
        return S_final

    def greedy_solve(vals):
        """Fallback greedy solver."""
        S_set = set()
        S_list = []
        # Sort by magnitude descending
        for v in sorted(vals, key=abs, reverse=True):
            if can_add(S_set, v):
                S_set.add(v)
                S_list.append(v)
        return S_list

    def can_add(S_set, x):
        """Check if adding x to S keeps it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
        return True

    def local_search(S_init, all_elements, time_limit=5.0):
        """
        Hill Climbing with 1-move (add) and 2-move (swap) operations.
        """
        S_set = set(S_init)
        S_list = list(S_init)
        current_size = len(S_set)
        start_ts = time.time()
        
        improved = True
        while improved and (time.time() - start_ts < time_limit):
            improved = False
            
            # Try Add
            # Iterate over candidates not in S
            # To make it faster, we can shuffle or just iterate
            for x in all_elements:
                if x not in S_set:
                    if can_add(S_set, x):
                        S_set.add(x)
                        S_list.append(x)
                        current_size += 1
                        improved = True
                        break  # Restart search after improvement
            
            if improved:
                continue
                
            # Try Swap: Remove x, Add y
            # We want to swap a "bad" element with a "good" one?
            # Actually, if we are stuck, no single add works.
            # A swap might allow adding more later, but directly swapping doesn't increase size.
            # Wait, the goal is to maximize size.
            # If no add works, we are at a local maximum for 1-add moves.
            # To escape, we need to remove an element that is "blocking" many others.
            # Then add multiple elements?
            # Standard local search for Max Independent Set (or similar) uses 1-swap or 2-swap.
            # Here, "Swap" means Remove x, Add y. Size stays same.
            # Then we try to add more.
            
            # Heuristic: Find an element x in S whose removal allows adding at least 2 new elements?
            # Or just find a swap that leads to a state where an add is possible.
            
            # Simpler: Try all pairs (x in S, y not in S).
            # If we swap x for y, does it allow an add?
            # This is O(n^2 * n) which is slow.
            
            # Alternative: Just try to add elements in random order?
            # If stuck, shuffle the "all_elements" order for the add phase?
            # The current add phase iterates in sorted order (or original).
            # Let's try a few random restarts of the add phase if stuck.
            
            # Actually, for this specific problem, "Stuck" means no single element can be added.
            # This is a maximal sum-free set.
            # To get a larger one, we must change the set structure.
            
            # Let's try a "2-opt" style: Remove x, Add y.
            # We prioritize x that has high "conflict degree" and y that has low conflict.
            
            # Calculate conflict degree for x in S:
            # How many elements z not in S are blocked by x?
            # i.e., z cannot be added because of x.
            
            # This is getting complex. Let's stick to a simpler effective heuristic:
            # If stuck, try swapping each x in S with each y not in S, and check if the new set allows an add.
            # To keep it fast, limit the number of swaps tried.
            
            if time.time() - start_ts > time_limit:
                break
                
            # Try limited swaps
            swaps_tried = 0
            for x in list(S_list):
                if swaps_tried > 50: break # Limit work
                for y in all_elements:
                    if y in S_set: continue
                    if y == x: continue
                    
                    # Simulate swap
                    temp_set = S_set - {x}
                    if can_add(temp_set, y):
                        # Swap is valid
                        # Check if this swap enables an add
                        # Actually, just do the swap and let the next loop try adds.
                        # But we need to ensure we don't loop forever.
                        # If we swap and then add, size increases.
                        # If we swap and no add works, size stays same, but structure changed.
                        
                        # Let's just do the swap if it seems "promising"
                        # Promising if y has larger magnitude? Or random?
                        
                        # Let's just accept the swap if it allows an immediate add.
                        # Check if any z can be added to (temp_set U {y})
                        new_set = temp_set | {y}
                        for z in all_elements:
                            if z in new_set: continue
                            if can_add(new_set, z):
                                # Accept swap and add
                                S_set = new_set
                                S_set.add(z)
                                S_list = list(S_set)
                                current_size += 1
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
                    
        return S_list

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 5:
            # Not enough time, return odd numbers as fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        vals = instance(size, seed)
        
        # Allocate time based on size
        # n=60: 5s, n=120: 15s, n=240: 30s
        if size <= 60:
            t_lim = 5.0
        elif size <= 120:
            t_lim = 15.0
        else:
            t_lim = 30.0
            
        result = solve_one_lp(vals, time_limit=t_lim)
        
        # Final verification
        if not is_sum_free(result):
            # Fallback to greedy
            result = greedy_solve(vals)
            
        results.append(result)

    return results

def is_sum_free(S):
    S_set = set(S)
    for a in S:
        for b in S:
            if a + b in S_set:
                return False
    return True

# EVOLVE-BLOCK-END
