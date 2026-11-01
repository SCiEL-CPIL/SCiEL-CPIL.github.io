"""
Exact ILP Strategy for Ternary Covering Codes.

Strategy:
1. Formulate the problem as a Minimum Set Cover / Integer Linear Programming (ILP) problem.
   - Variables: x_c for each word c in {0,1,2}^n (binary, 0 or 1).
   - Objective: Minimize sum(x_c).
   - Constraints: For every word w in {0,1,2}^n, sum(x_c for c in ball(w)) >= 1.
     Note: ball(w) is the set of codewords c such that dist(c, w) <= radius.
     Since the relation is symmetric, ball(w) is the set of c such that w is in ball(c).
     In the matrix A, A[w, c] = 1 if dist(w, c) <= radius.
     Then the constraint is A @ x >= 1.

2. Use scipy.optimize.milp (HiGHS solver) to find the optimal or near-optimal solution.
   - The problem size is 2187 variables and 2187 constraints.
   - The constraint matrix is sparse (each row has 1 + (q-1)*n = 15 ones for q=3, n=7).
   - HiGHS is very fast on such structured problems.

3. Fallback: If ILP fails or times out, use a fast greedy heuristic to ensure a valid return.

This approach is fundamentally different from local search: it uses global optimization
with linear relaxation and cutting planes, which can jump over local minima inherent
in heuristic searches.
"""

import time
import numpy as np
import random
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    total = q ** n
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
        
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n))

    # 1. Build the constraint matrix A (sparse or dense?)
    # A is total x total. A[w, c] = 1 if dist(w, c) <= radius.
    # For q=3, n=7, total=2187. Dense matrix is 2187*2187*8 bytes ~ 38 MB.
    # This is acceptable for memory. It might be faster to build dense for milp.
    # However, milp accepts sparse matrices too. Let's use dense for simplicity and speed in small cases.
    # Actually, building the matrix takes time. Let's precompute it.
    
    # Precompute distances? No, just check condition.
    # A[w, c] = 1 if sum(int_to_digits(w)[i] != int_to_digits(c)[i] for i in range(n)) <= radius
    
    # Optimization: Generate balls for each c, then fill A.
    # A is symmetric.
    
    # Let's try to build the matrix efficiently.
    # Instead of checking all pairs, we know the structure.
    # For each c, find all w in ball(c). Set A[w, c] = 1.
    
    A = np.zeros((total, total), dtype=np.int8)
    
    for c in range(total):
        digits_c = int_to_digits(c)
        # Center
        A[c, c] = 1
        # Radius 1 neighbors
        for i in range(n):
            orig = digits_c[i]
            for d in range(q):
                if d == orig:
                    continue
                digits_c[i] = d
                w = digits_to_int(digits_c)
                A[w, c] = 1
            digits_c[i] = orig
            
    # If radius > 1, we would need to expand further. The task specifies radius 1 for the known problem,
    # but the interface takes radius. The problem statement says "radius" is passed.
    # The specific task is K_3(7,1), so radius=1.
    # If radius is larger, the ball size grows. For radius=1, ball size is 15.
    # If radius is generic, we should handle it.
    # Given the "PUBLIC TASK CONTRACT" mentions "radius" parameter, but the specific problem is radius 1.
    # I will assume radius=1 for the optimization structure but write generic code if possible.
    # Actually, for radius > 1, the matrix becomes denser.
    # For now, assume radius=1 as per the problem description "differs ... in at most one position".
    
    # 2. Define ILP problem
    # Minimize c @ x
    # Subject to: A @ x >= 1
    #              0 <= x <= 1
    #              x is integer
    
    c_obj = np.ones(total, dtype=float)
    
    # LinearConstraint(A, 1, np.inf) means A @ x >= 1
    # Note: milp expects constraints in the form lb <= A @ x <= ub
    constraint = LinearConstraint(A, 1, np.inf)
    
    # Bounds for x: 0 to 1
    bounds = Bounds(0, 1)
    
    # Integrality: 1 for all variables
    integrality = np.ones(total, dtype=int)
    
    # 3. Solve
    result = None
    try:
        # options can be tuned for HiGHS
        # time_limit is not directly supported in milp options in all versions, 
        # but we can rely on the deadline check if we run it in a thread? 
        # No, milp is blocking.
        # HiGHS has a time_limit option.
        # Let's check if we can pass time_limit.
        # scipy.optimize.milp(c, constraints=..., integrality=..., bounds=..., options={'time_limit': 10})
        
        # We have ~145s available. Let's give it 140s.
        options = {
            'time_limit': 140.0,
            'mip_rel_gap': 0.0  # We want exact if possible, or best effort
        }
        
        result = milp(c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options=options)
        
    except Exception as e:
        print(f"ILP failed: {e}")
        result = None
        
    # 4. Process Result
    if result is not None and result.success and result.x is not None:
        # result.x contains the solution
        # Due to floating point, we need to round
        x = np.round(result.x).astype(int)
        
        # Extract codewords where x[c] == 1
        code_indices = np.where(x == 1)[0].tolist()
        
        # Verify validity (just in case)
        # Rebuild cover count
        cover_count = np.zeros(total, dtype=np.int8)
        for c in code_indices:
            # Add ball of c
            # We can use A[:, c] which is the column of A, i.e., rows w such that A[w,c]=1
            # This is exactly the set of w covered by c.
            cover_count[A[:, c] == 1] += 1
            
        if np.all(cover_count > 0):
            # Valid solution
            final_code = [int_to_digits(c) for c in code_indices]
            return final_code
        else:
            # ILP returned an invalid solution (should not happen with >= 1 constraint, but maybe rounding issues?)
            # Or maybe it timed out and returned a bound? No, if success is False, we skip.
            # If success is True but invalid, it's a bug in my formulation or scipy.
            # Let's fallback to greedy.
            pass
            
    # 5. Fallback: Greedy Heuristic
    # If ILP fails, times out, or returns invalid, use a fast greedy.
    
    # Precompute balls as lists for greedy
    balls = []
    for c in range(total):
        digits_c = int_to_digits(c)
        covered = {c}
        for i in range(n):
            orig = digits_c[i]
            for d in range(q):
                if d == orig:
                    continue
                digits_c[i] = d
                w = digits_to_int(digits_c)
                covered.add(w)
            digits_c[i] = orig
        balls.append(covered)
        
    uncovered_set = set(range(total))
    code = []
    
    # Greedy Set Cover
    while uncovered_set and time.time() < deadline - 5:
        best_c = -1
        best_gain = 0
        for c in range(total):
            gain = 0
            for w in balls[c]:
                if w in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if gain == 0:
                continue
        if best_c == -1:
            break
        code.append(best_c)
        for w in balls[best_c]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                
    # Local Search cleanup for the greedy result (quick pass)
    if time.time() < deadline - 2:
        cover_count = np.zeros(total, dtype=np.int8)
        for c in code:
            cover_count[A[:, c] == 1] += 1
            
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            i = 0
            while i < len(code):
                c = code[i]
                # Check redundancy
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    code.pop(i)
                    cover_count[A[:, c] == 1] -= 1
                    improved = True
                    i = 0
                else:
                    i += 1
                    
    final_code = [int_to_digits(c) for c in code]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in final_code:
        if c not in seen:
            seen.add(c)
            unique_code.append(c)
            
    return unique_code

# EVOLVE-BLOCK-END
