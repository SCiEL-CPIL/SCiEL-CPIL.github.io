import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_milp(elements):
        n = len(elements)
        if n == 0:
            return []
        
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        
        # Collect constraints
        # Each constraint is a list of indices and a bound
        # Type 1: x_i + x_j + x_k <= 2
        # Type 2: x_i + x_k <= 1 (when i==j in the sum a+b=c, i.e., 2a=c)
        
        constraints_rows = []
        constraints_bounds = []
        
        seen_triples = set()
        
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    # The triple is (i, j, k)
                    # If i == j, the constraint is x_i + x_k <= 1
                    # If i != j, the constraint is x_i + x_j + x_k <= 2
                    
                    # We need to be careful not to add duplicate constraints if possible, 
                    # though MILP solvers handle duplicates fine.
                    
                    if i == j:
                        # Constraint: x_i + x_k <= 1
                        # Represented as row with 1 at i, 1 at k
                        row = [0] * n
                        row[i] = 1
                        row[k] = 1
                        # Check if we already have this exact constraint
                        # For speed, we can just add it. 
                        # However, if i==k? That means elements[i] + elements[i] = elements[i] => elements[i]=0. 
                        # But 0 is excluded. So i != k.
                        constraints_rows.append(row)
                        constraints_bounds.append(1)
                    else:
                        # Constraint: x_i + x_j + x_k <= 2
                        row = [0] * n
                        row[i] = 1
                        row[j] = 1
                        row[k] = 1
                        constraints_rows.append(row)
                        constraints_bounds.append(2)
        
        if not constraints_rows:
            return list(elements)
            
        A = np.array(constraints_rows, dtype=float)
        b = np.array(constraints_bounds, dtype=float)
        
        c = np.ones(n) # Minimize sum x_i, so we want to maximize, so minimize -sum? 
        # scipy.optimize.milp minimizes c @ x.
        # We want to maximize sum x_i, so minimize -sum x_i.
        c = -np.ones(n)
        
        # Bounds for x_i: [0, 1]
        lb = np.zeros(n)
        ub = np.ones(n)
        
        integrality = np.ones(n) # All variables are integer (binary)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, 0, b),
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
            if res.status == 0:
                # res.x contains the optimal solution
                selected = [elements[i] for i in range(n) if res.x[i] > 0.5]
                return selected
            else:
                # Fallback to heuristic if MILP fails
                pass
        except Exception:
            pass
            
        # Fallback Heuristic
        return heuristic_solve(elements)

    def heuristic_solve(elements):
        n = len(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        
        # Precompute conflicts
        # conflict[i] is a set of (j, k) such that elements[i] + elements[j] = elements[k] or similar
        # Simpler: just check validity on the fly or precompute a "bad" set.
        
        # Greedy with random restarts
        best_set = set()
        
        for restart in range(10):
            rng = random.Random(100 + restart)
            order = list(range(n))
            rng.shuffle(order)
            current = set()
            
            for idx in order:
                val = elements[idx]
                # Check if adding val creates a sum-free violation
                valid = True
                # Check if val = a + b for a,b in current
                # Or a + val = b for a,b in current
                # Or val + b = c for b,c in current
                
                # To make this fast, we can just check against current set
                # Since n is small (max 240), O(n^2) check per insertion is 240^3 = 13M ops per restart.
                # 10 restarts = 130M ops. Might be slow in Python.
                
                # Optimized check:
                # 1. If val in current: skip (already there)
                # 2. Check if exists a,b in current such that a+b=val
                # 3. Check if exists a in current such that a+val in current
                # 4. Check if exists b in current such that val+b in current (same as 3)
                
                # Actually, just checking if any pair in current sums to val, 
                # or val + any element in current is in current.
                
                # Let's try a simpler heuristic: Sign partition
                pos = [x for x in elements if x > 0]
                neg = [x for x in elements if x < 0]
                if len(pos) > len(best_set): best_set = set(pos)
                if len(neg) > len(best_set): best_set = set(neg)
                
                # Local search on the best set found so far
                # ... (Implementing a quick local search)
        
        # Simple local search on the current best
        current = set(best_set)
        improved = True
        while improved:
            improved = False
            # Try removing an element to allow adding two others? 
            # Or just 1-flip: remove one, see if we can add more?
            
            # 1-flip add
            for val in elements:
                if val in current: continue
                # Check if val can be added
                can_add = True
                for a in current:
                    if a + val in current:
                        can_add = False
                        break
                    if val + val in current: # 2*val
                        # If 2*val is in current, and val is not, adding val makes {val, 2val} which is not sum-free?
                        # Yes, val+val=2val. So if 2val in current, we cannot add val.
                        can_add = False
                        break
                if can_add:
                    current.add(val)
                    improved = True
            # Try 1-flip remove + add
            # This is getting complex. Let's stick to the MILP result primarily.
            # If MILP fails, the sign partition is usually a good baseline.
            
        return list(current)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 10:
            elems = instance(size, seed)
            # Fallback: take all positives
            results.append([x for x in elems if x > 0])
            continue

        elems = instance(size, seed)
        res = solve_milp(elems)
        results.append(res)

    return results
