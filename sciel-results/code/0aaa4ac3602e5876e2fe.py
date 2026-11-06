import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 180
    deadline = time.time() + 85
    
    # Find all triples (a, b, c) with a < b < c, distinct, and 1/a = 1/b + 1/c
    # This is equivalent to: bc = a(b+c), or a = bc/(b+c)
    # For fixed b, c: a = bc/(b+c). We need a to be an integer in [1, N] and a < b.
    # bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # So for each a, we factor a^2 = x*y where x = b-a, y = c-a, x < y (to ensure b < c)
    # b = a+x, c = a+y
    # Constraints: a+x <= N, a+y <= N, x < y, x*y = a^2, x >= 1
    
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        # x must divide a^2, x < y, x*y = a^2
        # x ranges from 1 to sqrt(a^2) = a
        for x in range(1, int(a2 ** 0.5) + 1):
            if a2 % x == 0:
                y = a2 // x
                if x >= y:
                    continue
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    triples.append((a, b, c))
    
    # Also consider cases where the "a" in 1/a = 1/b + 1/c might not be the smallest
    # Actually, 1/a = 1/b + 1/c implies 1/a > 1/b and 1/a > 1/c, so a < b and a < c.
    # So a is always the smallest. The above enumeration is complete.
    
    # Build conflict graph for MILP:
    # Binary variables x_i for i in 1..N
    # For each triple (a,b,c): x_a + x_b + x_c <= 2
    # Maximize sum x_i
    
    m = len(triples)
    n = N
    
    # Objective: maximize sum x_i
    c_obj = -np.ones(n)  # minimization
    
    # Constraints: for each triple, x_a + x_b + x_c <= 2
    A_ub = np.zeros((m, n))
    for i, (a, b, c) in enumerate(triples):
        A_ub[i, a-1] = 1
        A_ub[i, b-1] = 1
        A_ub[i, c-1] = 1
    
    b_ub = np.ones(m)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    # Linear constraint for MILP: A_ub @ x <= b_ub
    # scipy milp uses: lb <= A @ x <= ub
    lin_cons = LinearConstraint(A_ub, ub=b_ub)
    
    # Run MILP
    try:
        res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=bounds)
        if res.status == 0:  # Optimal
            solution = set(int(x + 0.5) + 1 for x in res.x if x > 0.5)
            best_set = solution
        else:
            best_set = None
    except Exception:
        best_set = None
    
    # Fallback: greedy approach
    def greedy_solution(order):
        S = set()
        for x in order:
            can_add = True
            for a, b, c in triples:
                if x in (a, b, c):
                    others = [v for v in (a, b, c) if v != x]
                    if others[0] in S and others[1] in S:
                        can_add = False
                        break
            if can_add:
                S.add(x)
        return S
    
    # Try multiple greedy orders
    best_size = 0
    best_solution = set()
    
    if best_set is not None:
        best_size = len(best_set)
        best_solution = best_set
    
    # Backward greedy (start from large numbers)
    S_back = greedy_solution(range(N, 0, -1))
    if len(S_back) > best_size:
        best_size = len(S_back)
        best_solution = S_back
    
    # Random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy_solution(order)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_solution = S_rand
    
    # Try to improve best_solution by local search
    # Remove elements that are part of no triple in current set, then re-add
    # Actually, let's try a simple local search: try swapping elements
    
    # Verify solution
    def is_valid(S):
        for a, b, c in triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    if not is_valid(best_solution):
        # Fallback to backward greedy
        best_solution = greedy_solution(range(N, 0, -1))
    
    return list(best_solution)
