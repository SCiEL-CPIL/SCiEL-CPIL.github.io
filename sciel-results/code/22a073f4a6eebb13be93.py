"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Check if pysat is available, otherwise fallback to simple heuristic
try:
    from pysat.formula import WCNF
    from pysat.solvers import Minisat22
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_with_pysat(values):
        """
        Solve the MaxSAT problem:
        Maximize sum(x_i)
        Subject to: x_i + x_j + x_k <= 2 for all a,b,c in values where a+b=c.
        
        Variables x_i (1..n) indicate if value[i] is in the subset.
        Clauses:
        1. Objective: maximize sum x_i. In MaxSAT, we can add soft clauses (x_i) with weight 1.
           Or use a cardinality constraint approach.
           Simpler: Add soft unit clauses (x_i) with weight 1.
           And hard clauses for constraints: (!x_i or !x_j or !x_k) for a+b=c.
        """
        n = len(values)
        if n == 0:
            return []
        
        # Map value to variable id (1-indexed)
        val_to_var = {val: i+1 for i, val in enumerate(values)}
        
        clauses = []
        
        # Find all sum relations a+b=c
        # To speed up, we can precompute sums
        vals_set = set(values)
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                c = a + b
                if c in vals_set:
                    var_a = val_to_var[a]
                    var_b = val_to_var[b]
                    var_c = val_to_var[c]
                    # Clause: not (a AND b AND c) -> !a OR !b OR !c
                    clauses.append([-var_a, -var_b, -var_c])
        
        # Create WCNF
        # Weight 1 for objective (maximize inclusion)
        # Weight 1000 for hard constraints (or just use hard clauses in SAT and maximize soft)
        # pysat Minisat22 can solve weighted MaxSAT directly?
        # Actually, Minisat22 is a SAT solver. For MaxSAT, we might need MaxSAT solver.
        # Let's check if pysat has a MaxSAT solver. 
        # `pysat.solvers` has `Glucose3` etc which are SAT.
        # There is `pysat.card` for cardinality.
        # For MaxSAT, we can use `pysat.formula.WCNF` and a solver that supports it?
        # Standard Minisat does not support WCNF directly in the same way.
        # However, we can transform MaxSAT to SAT with an auxiliary variable T and binary search on T.
        # Or use a specific MaxSAT solver if available.
        
        # Let's try a simpler approach: Binary search on the size k.
        # Is there a sum-free subset of size k?
        # This is a SAT problem with cardinality constraint sum(x_i) >= k.
        # We can use pysat.card.Card to enforce sum(x_i) >= k.
        
        from pysat.card import Card
        
        best_size = 0
        low = 0
        high = n
        
        # Pre-compute hard clauses
        hard_clauses = []
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                c = a + b
                if c in vals_set:
                    var_a = val_to_var[a]
                    var_b = val_to_var[b]
                    var_c = val_to_var[c]
                    hard_clauses.append([-var_a, -var_b, -var_c])
        
        # Binary search for max k
        # To save time, we can start with a lower bound from heuristics
        # But let's just binary search.
        
        while low < high and time.time() < DEADLINE:
            mid = (low + high + 1) // 2
            
            # Solve SAT: sum(x_i) >= mid AND hard_clauses
            # Use Card to create cardinality constraints
            # Card.lte or Card.gte?
            # We want sum >= mid.
            # pysat.card.Card.gte(lits, k) generates clauses for sum >= k.
            
            try:
                card_clauses = Card.gte(list(val_to_var.values()), mid)
            except Exception:
                # If Card fails or is too slow, break
                break
            
            all_clauses = hard_clauses + card_clauses
            
            # Use a SAT solver
            # Minisat22 is fast
            with Minisat22(assumptions=all_clauses) as s:
                if s.solve():
                    low = mid
                else:
                    high = mid - 1
        
        if low > 0:
            # Re-solve with k=low to get the model
            card_clauses = Card.gte(list(val_to_var.values()), low)
            all_clauses = hard_clauses + card_clauses
            with Minisat22(assumptions=all_clauses) as s:
                if s.solve():
                    model = s.get_model()
                    # Extract variables
                    selected = []
                    for var in model:
                        if var > 0:
                            # Map var back to value
                            # We need reverse map
                            # val_to_var is val -> var
                            # Create var_to_val
                            pass
                    
                    # Build var_to_val
                    var_to_val = {v: k for k, v in val_to_var.items()}
                    for var in model:
                        if var > 0 and var in var_to_val:
                            selected.append(var_to_val[var])
                    return selected
        return []

    def greedy_fallback(values):
        """Fast greedy fallback if pysat is too slow or unavailable."""
        if not values:
            return []
        
        best = []
        best_size = 0
        
        # Try a few random orderings
        for trial in range(10):
            if time.time() > DEADLINE:
                break
            
            perm = list(values)
            random.shuffle(perm)
            
            subset = set()
            for x in perm:
                ok = True
                for a in subset:
                    if x + a in subset:
                        ok = False
                        break
                    if x + x in subset:
                        ok = False
                        break
                if ok:
                    subset.add(x)
            
            if len(subset) > best_size:
                best_size = len(subset)
                best = list(subset)
        
        return best

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        
        if HAS_PYSAT and size <= 240: # pysat might be slow for large n, but 240 is manageable?
            # Try pysat with a time limit? pysat doesn't have a built-in wall-clock limit easily.
            # We can run it and check time afterwards, but if it hangs, we fail.
            # To be safe, we can run pysat in a thread? No, standard lib only.
            # Let's try pysat but limit the search space or use a fast solver.
            # Actually, for n=240, the number of clauses is O(n^2) ~ 57k clauses.
            # SAT solvers are very fast. This should be instant.
            
            try:
                sol = solve_with_pysat(vals)
                if is_sum_free(sol):
                    results.append(sol)
                else:
                    # Fallback if solver returns invalid (shouldn't happen)
                    results.append(greedy_fallback(vals))
            except Exception:
                results.append(greedy_fallback(vals))
        else:
            results.append(greedy_fallback(vals))
            
    return results
# EVOLVE-BLOCK-END
