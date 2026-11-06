import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, Highs
from random import Random

def construct():
    """
    Constructs sum-free subsets for three fixed instances.
    Uses MILP (HiGHS) to find the maximum cardinality sum-free subset.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_milp(values):
        n = len(values)
        if n == 0:
            return []
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # Find all sum-free triples (a, b, c) such that a + b = c
        # Constraints: x_a + x_b + x_c <= 2
        # We need to generate these constraints.
        # To speed up, we can use a set of values for O(1) lookup
        val_set = set(values)
        
        constraints = []
        
        # Iterate over all pairs (a, b)
        # Note: a and b can be the same.
        # We only need to add constraint if c is in the set.
        # To avoid duplicate constraints (e.g., a+b=c and b+a=c), we can just generate them all, 
        # but that's O(n^2). For n=240, n^2 is 57600, which is fine.
        # However, we must be careful not to add redundant constraints if possible, 
        # but HiGHS handles them well.
        
        # Optimization: 
        # For each a, for each b, if a+b is in val_set, add constraint.
        # We can skip if a > b to reduce half, but since a+b=c is symmetric in a,b, 
        # the constraint is the same. So we can assume a <= b.
        
        c_indices = []
        a_indices = []
        b_indices = []
        
        for i, a in enumerate(values):
            for j in range(i, n):
                b = values[j]
                c_val = a + b
                if c_val in val_set:
                    k = val_to_idx[c_val]
                    # Constraint: x_i + x_j + x_k <= 2
                    # If i == j, it's 2*x_i + x_k <= 2
                    # If i == k or j == k, it's x_i + x_j + x_i <= 2 => 2*x_i + x_j <= 2
                    # The linear constraint form handles coefficients.
                    # We build the matrix row by row or collect coefficients.
                    # scipy.optimize.milp takes A_ub, b_ub.
                    # Let's collect rows.
                    c_indices.append(k)
                    a_indices.append(i)
                    b_indices.append(j)
        
        if not c_indices:
            # No constraints, take all
            return values[:]

        # Build constraint matrix A_ub
        # Number of constraints
        num_cons = len(c_indices)
        # A_ub is (num_cons, n)
        # We can use sparse matrix for efficiency
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        for r in range(num_cons):
            i = a_indices[r]
            j = b_indices[r]
            k = c_indices[r]
            
            # Add coefficient for i
            if i == j:
                # 2 * x_i
                data.append(2)
                cols.append(i)
                rows.append(r)
            else:
                data.append(1)
                cols.append(i)
                rows.append(r)
                
                data.append(1)
                cols.append(j)
                rows.append(r)
            
            data.append(1)
            cols.append(k)
            rows.append(r)
        
        A_ub = csr_matrix((data, (rows, cols)), shape=(num_cons, n))
        b_ub = np.full(num_cons, 2)
        
        # Objective: Maximize sum(x_i)
        # milp minimizes, so minimize -sum(x_i)
        c_obj = -np.ones(n)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x_i are integers
        integrality = np.ones(n)
        
        # Solve
        try:
            res = milp(c=c_obj, constraints=LinearConstraint(A_ub, -np.inf, b_ub), 
                       integrality=integrality, bounds=bounds, 
                       options={"presolve": True, "time_limit": 10.0})
            
            if res.status == 0:
                # res.x is the solution
                x_sol = res.x
                # Select indices where x_i is close to 1
                selected = [values[i] for i in range(n) if x_sol[i] > 0.5]
                return selected
            else:
                # Fallback to greedy if solver fails
                pass
        except Exception:
            pass
            
        # Fallback: Greedy Local Search
        return greedy_solve(values)

    def greedy_solve(values):
        """
        Fallback heuristic: Start with all elements, remove those causing conflicts.
        Or start with random/odd and try to add.
        Better: Start with all, while there is a conflict, remove one of the three.
        To maximize size, we should be strategic.
        Simple heuristic: 
        1. Start with all elements.
        2. While there exists a+b=c:
           Remove c (usually largest magnitude or just c) if it doesn't help, or remove a or b?
           Actually, removing c is often good because c is "large".
           Let's try: remove the element with the highest "degree" in the conflict graph.
        """
        S = set(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # Iterate until no conflicts
        changed = True
        while changed:
            changed = False
            # Find a conflict
            conflict_found = False
            # We need to find a+b=c in S
            # To be efficient, we can pick a random a, b and check
            # Or iterate through sorted list
            vals = sorted(S)
            for i, a in enumerate(vals):
                for j in range(i, len(vals)):
                    b = vals[j]
                    c = a + b
                    if c in S:
                        # Conflict (a, b, c)
                        # Which one to remove?
                        # Heuristic: Remove the one that appears in most other conflicts?
                        # For simplicity and speed, let's remove c.
                        # Or remove the one with smallest absolute value?
                        # Let's try removing c first, then a, then b in a random order?
                        # Let's stick to a simple deterministic rule: remove c.
                        # If that leads to suboptimal, we can improve.
                        # Actually, a better heuristic is to remove the element that is part of the most triples.
                        # But computing that is expensive.
                        # Let's just remove c.
                        S.remove(c)
                        changed = True
                        conflict_found = True
                        break
                if conflict_found:
                    break
                    
        return sorted(S)

    instances = [(60, 801), (120, 802), (240, 803)]
    results = []
    
    for size, seed in instances:
        if time.time() - start_time > TIME_LIMIT:
            # Emergency fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        subset = solve_milp(vals)
        results.append(subset)
        
    return results

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Sizes: {[len(r) for r in res]}")
