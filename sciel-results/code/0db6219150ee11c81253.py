import math
import time
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    N = 512
    MAX_A = 511
    deadline = start_time + 110.0
    
    # Precompute prime factors for each number 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(MAX_A + 1)]
    factors[1] = set()
    for i in range(2, MAX_A + 1):
        if not factors[i]:
            # i is prime
            for j in range(i, MAX_A + 1, i):
                factors[j].add(i)
    
    # Weights: w[a] = 1/(512-a). We want to maximize sum of w[a].
    # w[a] is strictly increasing with a.
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)
    
    # Helper: check if a is coprime to all elements in set S
    def is_coprime_to_set(a, S):
        for b in S:
            if factors[a] & factors[b]:
                return False
        return True
    
    # Greedy initialization
    # Iterate from largest a to smallest.
    # If a is coprime to all already chosen, add it.
    # This maximizes the count of large numbers first.
    chosen_set = set()
    for a in range(MAX_A, 0, -1):
        if is_coprime_to_set(a, chosen_set):
            chosen_set.add(a)
    
    best_score = sum(weights[a] for a in chosen_set)
    best_set = set(chosen_set)
    
    # Local Search
    # We try to improve the set by swapping elements.
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_gain = 0.0
        best_swap = None
        
        for x in list(chosen_set):
            best_y_for_x = -1
            for y in range(MAX_A, x, -1):
                if y in chosen_set:
                    continue
                valid = True
                for z in chosen_set:
                    if z == x:
                        continue
                    if factors[y] & factors[z]:
                        valid = False
                        break
                if valid:
                    best_y_for_x = y
                    break
            
            if best_y_for_x != -1:
                gain = weights[best_y_for_x] - weights[x]
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (x, best_y_for_x)
        
        if best_swap:
            x, y = best_swap
            chosen_set.remove(x)
            chosen_set.add(y)
            best_score += best_gain
            best_set = set(chosen_set)
            improved = True
    
    # Attempt MILP for exact solution
    if time.time() < deadline - 10.0:
        try:
            # Build conflict constraints
            # x_a + x_b <= 1 for all pairs (a,b) with gcd(a,b) > 1
            # a ranges from 1 to MAX_A. Index in vector is a-1.
            n_vars = MAX_A
            c = np.zeros(n_vars)
            for a in range(1, MAX_A + 1):
                c[a-1] = weights[a]
            
            # Generate rows for LinearConstraint
            # A x <= b
            # We will build a sparse matrix or list of rows
            # Since MAX_A is 511, the number of conflicts is manageable.
            # Conflict exists if factors[a] & factors[b] is not empty.
            
            rows = []
            cols = []
            vals = []
            
            for a in range(1, MAX_A + 1):
                for b in range(a + 1, MAX_A + 1):
                    if factors[a] & factors[b]:
                        # x_a + x_b <= 1
                        # Index a-1, b-1
                        rows.append(len(rows)) # dummy, we'll build arrays
                        # Actually, let's just build the arrays directly
                        cols.extend([a-1, b-1])
                        vals.extend([1.0, 1.0])
                        
            # We need to structure this for scipy.optimize.milp
            # It accepts A_ub as a sparse matrix or array.
            # Let's create a sparse matrix.
            from scipy.sparse import lil_matrix
            
            A_ub = lil_matrix((len(rows) if rows else 0, n_vars))
            # Wait, the loop above didn't count rows correctly if I just extend.
            # Let's restart the matrix building properly.
            
            A_ub = lil_matrix((0, n_vars))
            b_ub = []
            
            for a in range(1, MAX_A + 1):
                for b in range(a + 1, MAX_A + 1):
                    if factors[a] & factors[b]:
                        A_ub[len(b_ub), a-1] = 1.0
                        A_ub[len(b_ub), b-1] = 1.0
                        b_ub.append(1.0)
            
            if len(b_ub) == 0:
                # No conflicts? Impossible for 1..511
                pass
            else:
                A_ub = A_ub.tocsr()
                constraints = LinearConstraint(A_ub, -np.inf, b_ub)
                integrality = np.ones(n_vars)
                bounds = Bounds(0, 1)
                
                res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
                
                if res.status == 0:
                    # Success
                    milp_set = set()
                    for i in range(n_vars):
                        if res.x[i] > 0.5:
                            milp_set.add(i + 1)
                    
                    milp_score = sum(weights[a] for a in milp_set)
                    
                    if milp_score > best_score:
                        best_score = milp_score
                        best_set = milp_set
                        
        except Exception as e:
            # If MILP fails, fall back to heuristic result
            pass

    return list(best_set)
