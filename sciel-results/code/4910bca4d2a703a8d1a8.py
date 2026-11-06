import time
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    Uses Integer Linear Programming (ILP) via scipy.optimize.milp to find
    the maximum independent set in the 3-uniform hypergraph of forbidden triples.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 180

    # Precompute all forbidden triples: distinct a, b, c in [1, N] with 1/a = 1/b + 1/c
    # Equivalent to: a*(b+c) = b*c, i.e., a = bc/(b+c), so b+c must divide bc.
    # For distinct a, b, c: we need a != b, a != c, b != c.
    # Note: 1/a = 1/b + 1/c means a is the "smaller" one (since 1/a > 1/b and 1/a > 1/c).
    # Actually 1/a = 1/b + 1/c > 1/b, so a < b. Similarly a < c.
    # So a is the smallest of the three.
    
    triples = []
    triple_set = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):  # b != c, assume b < c
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    # Ensure a < b < c for canonical form (a is smallest)
                    # Since 1/a = 1/b + 1/c, a < b and a < c always.
                    # So a is the minimum.
                    t = tuple(sorted((a, b, c)))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    T = len(triples)
    
    # ILP formulation:
    # Variables: x_i in {0, 1} for i = 1..N (x_i = 1 means i is in the set)
    # Maximize: sum(x_i)
    # Subject to: for each triple (a, b, c): x_a + x_b + x_c <= 2
    #             x_i in {0, 1}
    
    # scipy.optimize.milp minimizes c^T x subject to constraints.
    # So minimize -sum(x_i) => c = [-1, -1, ..., -1]
    
    c = np.ones(N)  # minimize sum(x_i) but we want to maximize, so use -c
    c = -np.ones(N)
    
    # Constraints: for each triple, x_a + x_b + x_c <= 2
    # A is T x N matrix, each row has 1s at positions a-1, b-1, c-1
    A = np.zeros((T, N))
    for idx, (a, b, c_val) in enumerate(triples):
        A[idx, a - 1] = 1
        A[idx, b - 1] = 1
        A[idx, c_val - 1] = 1
    
    # LinearConstraint: A @ x <= 2 for all rows
    # scipy expects: lb <= A @ x <= ub
    lb = np.full(T, -np.inf)
    ub = np.full(T, 2)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all variables are integers (binary)
    integrality = np.ones(N)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.success and result.x is not None:
            # Extract the solution
            x = result.x
            # Due to floating point, round to nearest integer
            x_rounded = np.round(x).astype(int)
            # Ensure binary
            x_rounded = np.clip(x_rounded, 0, 1)
            
            solution = set()
            for i in range(N):
                if x_rounded[i] == 1:
                    solution.add(i + 1)
            
            # Verify the solution
            if len(solution) > 0:
                # Check no forbidden triples
                valid = True
                for (a, b, c_val) in triples:
                    if a in solution and b in solution and c_val in solution:
                        valid = False
                        break
                
                if valid:
                    return list(solution)
                else:
                    # Fallback: clean up
                    S = set(solution)
                    for (a, b, c_val) in triples:
                        if a in S and b in S and c_val in S:
                            # Remove the element with highest "conflict degree"
                            # Just remove a for simplicity
                            S.discard(a)
                    if len(S) > 0:
                        return list(S)
        
        # If ILP failed or returned invalid, fall back to a greedy/heuristic approach
        raise Exception("ILP did not yield a valid solution")
        
    except Exception:
        # Fallback: use a simple greedy approach if ILP fails or times out
        if time.time() > deadline:
            # Last resort: return all numbers > 90 (known valid, size 90)
            return list(range(91, N + 1))
        
        # Greedy: sort by conflict degree, add if safe
        conflicts = {i: set() for i in range(1, N + 1)}
        for (a, b, c_val) in triples:
            for x in (a, b, c_val):
                others = tuple(sorted([y for y in (a, b, c_val) if y != x]))
                conflicts[x].add(others)
        
        S = set()
        sorted_nodes = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
        for v in sorted_nodes:
            is_safe = True
            for u, w in conflicts[v]:
                if u in S and w in S:
                    is_safe = False
                    break
            if is_safe:
                S.add(v)
        
        if len(S) == 0:
            S = set(range(91, N + 1))
        
        return list(S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    N = 180
    valid = True
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    if a in S and b in S and c in S:
                        valid = False
                        print(f"Found conflict: {a}, {b}, {c}")
    print("Valid:", valid)
