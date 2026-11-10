import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Generate 3-APs
    aps = []
    for d in range(1, (n-1)//2 + 1):
        for a in range(1, n - 2*d + 1):
            b = a + d
            c = a + 2*d
            aps.append((a, b, c))
    
    # Weights
    weights = np.array([1.0/i for i in range(1, n+1)])
    
    # Try ILP
    try:
        A = np.zeros((len(aps), n))
        b = np.zeros(len(aps))
        for idx, (a, b_val, c) in enumerate(aps):
            A[idx, a-1] = 1
            A[idx, b_val-1] = 1
            A[idx, c-1] = 1
            b[idx] = 2
            
        integrality = np.ones(n)
        res = milp(
            c=-weights,
            constraints=LinearConstraint(A, ub=b),
            integrality=integrality,
            bounds=Bounds(0, 1)
        )
        
        if res.status == 0:
            x = res.x
            selected = [i+1 for i in range(n) if x[i] > 0.5]
            if selected:
                return selected
    except Exception:
        pass
        
    # Fallback Heuristic
    S = set(range(1, n+1))
    for a, b_val, c in aps:
        if a in S and b_val in S and c in S:
            to_remove = max(a, b_val, c)
            S.remove(to_remove)
            
    new_S = set(S)
    # Greedy addition of highest weight elements
    for x in range(1, n+1):
        if x not in new_S:
            valid = True
            for a in new_S:
                c1 = 2*x - a
                if c1 in new_S and c1 != x:
                    valid = False
                    break
                c2 = 2*a - x
                if c2 in new_S and c2 != x:
                    valid = False
                    break
            if valid:
                new_S.add(x)
                
    if not new_S:
        return [1]
    return sorted(new_S)
# EVOLVE-BLOCK-END
