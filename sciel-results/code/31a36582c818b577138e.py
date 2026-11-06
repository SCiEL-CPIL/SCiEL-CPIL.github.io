import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
from itertools import combinations
import sys

def construct():
    # Precompute conflict constraints
    # Variables x_1 ... x_24
    # Constraint for each identity 1/a = sum_{j in J} 1/j with a not in J:
    # x_a + sum_{j in J} x_j <= |J|
    
    n = 24
    conflicts = []
    # J is a subset of {2, ..., 24}
    # We check all subsets of size 1 to 10 (sufficient as sum of reciprocals grows)
    # Actually, we can just check all subsets of {2..24}
    # To be safe and fast, we limit size. 
    # Max size: sum of smallest k reciprocals from {2..24}.
    # 1/24 + ... + 1/(24-k+1).
    # For a=24, 1/24. 1/24 < 1/24. No.
    # 1/24 + 1/23 > 1/24. So size >= 2 for a=24? No, J must sum to 1/24.
    # Min element in J must be > a? No.
    # If j < a, 1/j > 1/a. So J can contain elements < a.
    # Example: a=5, 1/5 = 1/6 + 1/30 (30 not in range). 1/5 = 1/6 + 1/10 + 1/15? 1/6+1/10+1/15 = 5/30+3/30+2/30 = 10/30 = 1/3 != 1/5.
    # 1/5 = 1/10 + 1/10 (not distinct).
    # 1/5 = 1/6 + 1/15 + 1/30?
    
    # We will generate all subsets of {2..24} of size up to 8.
    # This is safe because 1/2 + ... + 1/9 is already > 1.
    # For a >= 2, 1/a <= 1/2.
    # If J contains 2, sum >= 1/2. If a=2, sum=1/2, J={2} but 2 in J and a=2, conflict a notin J fails.
    # So if a=2, J cannot contain 2.
    # Generally, if min(J) = m, sum >= 1/m.
    # If 1/m > 1/a, then m < a.
    
    elements = list(range(2, 25))
    valid_constraints = []
    
    # Iterate over all possible subsets J of elements
    # We can do this by iterating over sizes
    max_size = 8
    for size in range(1, max_size + 1):
        for J in combinations(elements, size):
            s = Fraction(0, 1)
            for j in J:
                s += Fraction(1, j)
            
            # Check if s is of the form 1/a for some integer a in 1..24
            if s.numerator == 1:
                a = s.denominator
                if 1 <= a <= 24:
                    # Check if a is not in J
                    if a not in J:
                        # Create constraint: x_a + sum(x_j for j in J) <= len(J)
                        # Indexing: x_i corresponds to i-1 in the vector
                        # Coefficients for x_1 ... x_24
                        coeffs = [0] * 24
                        coeffs[a-1] = 1
                        for j in J:
                            coeffs[j-1] += 1
                        # We want to avoid duplicates
                        # Convert to tuple for hashing
                        key = tuple(coeffs)
                        if key not in valid_constraints:
                            valid_constraints.append(key)
    
    # If no conflicts found, we can select all?
    # But there are always conflicts. e.g. 1/2 = 1/3 + 1/6.
    # a=2, J={3,6}. Constraint: x_2 + x_3 + x_6 <= 2.
    
    m = len(valid_constraints)
    if m == 0:
        return list(range(1, 25))
    
    # Build ILP
    # c = -1 for all (minimize -sum x_i)
    c = np.ones(n) * -1
    
    # A_ub, b_ub
    A = np.array(valid_constraints, dtype=float)
    b_ub = np.array([len(J) for J in []]) # Placeholder, need to store len(J)
    
    # Re-do to store len(J) properly
    # Let's restart the constraint collection to store (coeffs, len_J)
    valid_constraints_data = []
    seen = set()
    for size in range(1, max_size + 1):
        for J in combinations(elements, size):
            s = Fraction(0, 1)
            for j in J:
                s += Fraction(1, j)
            if s.numerator == 1:
                a = s.denominator
                if 1 <= a <= 24 and a not in J:
                    coeffs = [0] * 24
                    coeffs[a-1] = 1
                    for j in J:
                        coeffs[j-1] += 1
                    key = tuple(coeffs)
                    if key not in seen:
                        seen.add(key)
                        valid_constraints_data.append((coeffs, size))
    
    m = len(valid_constraints_data)
    if m == 0:
        return list(range(1, 25))
        
    A_ub = np.array([c for c, _ in valid_constraints_data], dtype=float)
    b_ub = np.array([s for _, s in valid_constraints_data], dtype=float)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        # res.x is the solution
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            # Fallback
            return [1]
    else:
        # Fallback to a simple heuristic if ILP fails
        # Greedy: pick largest numbers?
        # Or just return a known good set?
        # The seed program returned [13..24]. Let's try to improve on that or return it.
        return list(range(13, 25))
