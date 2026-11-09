import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 180
    # Generate all valid triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # This implies c = ab / (a+b) is integer? No.
    # 1/a = 1/b + 1/c => c(a+b) = ab => ca + cb = ab => a(c-b) = -cb => b(a-c) = ac => b = ac / (a-c)?
    # Wait, a is smallest.
    # 1/a = 1/b + 1/c.
    # b = a * c / (c - a).
    # For a fixed a, c > b > a.
    # Iterate a from 1 to n.
    # Iterate c from 2*a to n? (Since b > a, c > b > a).
    # Actually, b = ac / (c-a).
    # Let d = c - a. Then c = a + d.
    # b = a(a+d) / d = a + a^2/d.
    # So d must divide a^2.
    # And b must be integer and b > a.
    # And b < c.
    
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        # d divides a^2
        # d = c - a. c <= n => d <= n - a.
        # Also b = a + a^2/d.
        # We need b < c = a + d => a^2/d < d => a^2 < d^2 => d > a.
        # So d > a and d | a^2 and a + d <= n.
        for d in range(1, int(np.sqrt(a2)) + 1):
            if a2 % d == 0:
                # d is a divisor.
                # Case 1: d itself
                if d > a and a + d <= n:
                    c = a + d
                    b = a + a2 // d
                    if b > a and b < c and b <= n:
                        triples.append((a, b, c))
                # Case 2: a2/d
                d2 = a2 // d
                if d2 != d and d2 > a and a + d2 <= n:
                    c = a + d2
                    b = a + a2 // d2
                    if b > a and b < c and b <= n:
                        triples.append((a, b, c))
    
    # If no triples, return all
    if not triples:
        return list(range(1, n + 1))
    
    # MILP Formulation
    # Maximize sum(x_i)
    # Constraints: x_a + x_b + x_c <= 2 for each triple
    # x_i in {0, 1}
    
    c_obj = np.ones(n + 1) # 1-indexed, ignore index 0
    c_obj[0] = 0
    
    # Build constraint matrix A_ub
    # Rows: triples. Cols: 1..n
    A_ub = np.zeros((len(triples), n + 1))
    b_ub = np.full(len(triples), 2)
    
    for i, (a, b, c) in enumerate(triples):
        A_ub[i, a] = 1
        A_ub[i, b] = 1
        A_ub[i, c] = 1
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(n + 1), ub=np.ones(n + 1))
    bounds.lb[0] = 0
    bounds.ub[0] = 0
    
    # Integrality: x_i integer
    integrality = np.ones(n + 1)
    integrality[0] = 0 # x_0 is continuous (and fixed to 0)
    
    constraints = LinearConstraint(A_ub, ub=b_ub)
    
    res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to greedy
        return list(range(91, 181))
        
    # res.x gives the optimal solution
    # res.x[0] is x_0, res.x[1] is x_1, etc.
    S = [i for i in range(1, n + 1) if res.x[i] > 0.5]
    
    if not S:
        return list(range(91, 181))
        
    return S
