import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256


def _build_cover():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]
        d_pb = D2[p, A][None, :]
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover


COVER = _build_cover()

# Build conflict graph: edge (i,j) exists if i and j form an isosceles triangle with some third point k
# Actually, for the independent set formulation: two points i,j are incompatible if there exists k such that (i,j,k) is isosceles
# But we need: S is no-isosceles means no 3 points in S form an isosceles triangle.
# So the conflict graph has an edge between i and j if... wait, that's not right either.
# 
# Let me reconsider. The constraint is: no 3 points in S form an isosceles triangle.
# This is a 3-uniform hypergraph constraint, not a graph constraint.
# 
# For the MILP formulation:
# - x_p = 1 if point p is in S
# - Maximality: for every point p not in S, there exist a,b in S such that (p,a,b) is isosceles
#   Equivalently: for every p not in S, NOT (for all a,b, (p,a,b) is not isosceles or a not in S or b not in S)
#   This is tricky to linearize directly.
#
# Alternative: Use the fact that p can be added to S iff for all pairs (a,b) in COVER[p], not both a and b are in S.
# So p is blocked (cannot be added) iff exists (a,b) in COVER[p] such that a in S AND b in S.
#
# For maximality: for every p not in S, exists (a,b) in COVER[p] with a in S and b in S.
#
# Let's use auxiliary variables y_{p,(a,b)} for each p and each pair (a,b) in COVER[p]:
#   y_{p,(a,b)} = 1 if a in S and b in S (i.e., x_a = 1 and x_b = 1)
#   Actually we can use: y_{p,(a,b)} >= x_a + x_b - 1, y_{p,(a,b)} <= x_a, y_{p,(a,b)} <= x_b
#   And maximality: for each p, sum over (a,b) in COVER[p] of y_{p,(a,b)} >= (1 - x_p)
#   (If p not in S, at least one covering pair must have both endpoints in S)
#
# And no-isosceles: for each triple (i,j,k) that forms an isosceles triangle, x_i + x_j + x_k <= 2
#
# This is a valid MILP! Let's build it.

# Precompute all isosceles triples
def _build_isosceles_triples():
    triples = []
    for i in range(M):
        for j in range(i+1, M):
            for k in range(j+1, M):
                pi = POINTS[i]; pj = POINTS[j]; pk = POINTS[k]
                d2_ij = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
                d2_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
                d2_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
                if d2_ij == d2_ik or d2_ij == d2_jk or d2_ik == d2_jk:
                    triples.append((i,j,k))
    return triples


def _shrink(S):
    """Inclusion-minimal maximal: remove any point whose removal keeps maximality."""
    S = set(S)
    inS = [False] * M
    for v in S:
        inS[v] = True
    
    def _is_maximal(inS):
        for p in range(M):
            if inS[p]:
                continue
            for a, b in COVER[p]:
                if inS[a] and inS[b]:
                    break
            else:
                return False
        return True
    
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                S.add(x)
                inS[x] = True
    return S


def _greedy_maximal():
    S = set()
    inS = [False] * M
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if inS[v]:
            continue
        # Check if v is compatible with S (no isosceles triangle formed)
        compatible = True
        for a, b in COVER[v]:
            if inS[a] and inS[b]:
                compatible = False
                break
        if compatible:
            S.add(v)
            inS[v] = True
    return S


def _repair(S):
    """Greedy-complete S to a maximal no-isosceles set."""
    S = set(S)
    inS = [False] * M
    for v in S:
        inS[v] = True
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if inS[v]:
            continue
        compatible = True
        for a, b in COVER[v]:
            if inS[a] and inS[b]:
                compatible = False
                break
        if compatible:
            S.add(v)
            inS[v] = True
    return S


def _try_milp(deadline):
    """Try to solve via MILP. Returns a set if successful, else None."""
    t0 = time.time()
    
    # Build isosceles triples
    triples = _build_isosceles_triples()
    
    # Variables:
    # x[0..M-1] : binary, whether point i is in S
    # y[m] : binary, for each covering pair (p, (a,b)), whether a and b are both in S
    #         We only need y for pairs that appear in COVER
    
    # Collect all distinct (a,b) pairs from all COVER lists
    all_cover_pairs = {}  # (a,b) -> list of p's that cover with this pair
    pair_id = {}
    num_pairs = 0
    for p in range(M):
        for a, b in COVER[p]:
            key = (a, b)
            if key not in pair_id:
                pair_id[key] = num_pairs
                num_pairs += 1
            all_cover_pairs.setdefault(key, []).append(p)
    
    # MILP:
    # Minimize sum(x_i)
    # Subject to:
    # 1. For each isosceles triple (i,j,k): x_i + x_j + x_k <= 2
    # 2. For each pair (a,b) with id q: y_q >= x_a + x_b - 1
    # 3. For each pair (a,b) with id q: y_q <= x_a, y_q <= x_b
    # 4. For each point p: sum_{(a,b) in COVER[p]} y_{id(a,b)} >= 1 - x_p
    
    n_vars = M + num_pairs
    
    # Objective: minimize sum of x
    c = np.zeros(n_vars)
    for i in range(M):
        c[i] = 1.0
    
    # Constraints
    A_rows = []
    A_cols = []
    A_vals = []
    lb = []
    ub = []
    
    def add_constraint(row_coeffs, lower, upper):
        """row_coeffs: dict of var_index -> coefficient"""
        global row_num
        for var, coeff in row_coeffs.items():
            A_rows.append(len(lb))
            A_cols.append(var)
            A_vals.append(coeff)
        lb.append(lower)
        ub.append(upper)
    
    row_num = 0
    
    # Constraint 1: isosceles triples
    for (i, j, k) in triples:
        coeffs = {i: 1.0, j: 1.0, k: 1.0}
        add_constraint(coeffs, 0, 2)
    
    # Constraint 2: y_q >= x_a + x_b - 1  =>  x_a + x_b - y_q <= 1
    for (a, b), q in pair_id.items():
        coeffs = {a: 1.0, b: 1.0, q: -1.0}
        add_constraint(coeffs, None, 1.0)
        # For lower bound, we need x_a + x_b - y_q >= -1 => -x_a - x_b + y_q <= 1
        # Actually let's use: -x_a - x_b + y_q >= -1 => as upper bound form: y_q - x_a - x_b <= 1 is wrong
        # Let me redo: y >= x_a + x_b - 1 => x_a + x_b - y <= 1
        # That's already what I have. Good.
    
    # Constraint 3: y_q <= x_a  =>  y_q - x_a <= 0
    for (a, b), q in pair_id.items():
        coeffs = {q: 1.0, a: -1.0}
        add_constraint(coeffs, None, 0.0)
        coeffs2 = {q: 1.0, b: -1.0}
        add_constraint(coeffs2, None, 0.0)
    
    # Constraint 4: for each p, sum of y over pairs in COVER[p] >= 1 - x_p
    # => sum(y) + x_p >= 1
    for p in range(M):
        coeffs = {p: 1.0}
        for a, b in COVER[p]:
            q = pair_id[(a,b)]
            coeffs[q] = coeffs.get(q, 0.0) + 1.0
        add_constraint(coeffs, 1.0, None)
    
    A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(len(lb), n_vars))
    
    # Handle None in lb/ub
    lb_arr = np.array([v if v is not None else -np.inf for v in lb])
    ub_arr = np.array([v if v is not None else np.inf for v in ub])
    
    # Integer variables: all are binary
    integrality = np.ones(n_vars)
    
    # Bounds: all variables in [0, 1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    constraints = LinearConstraint(A, lb_arr, ub_arr)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={'time_limit': max(10, deadline - time.time() - 5)})
        
        if res.status == 0 and res.x is not None:
            S = set()
            for i in range(M):
                if res.x[i] > 0.5:
                    S.add(i)
            return S
    except Exception:
        pass
    
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 280.0
    
    best = None
    best_size = M + 1
    
    # Phase 1: Greedy + shrink to get initial good solution
    for _ in range(12):
        S = _shrink(_greedy_maximal())
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    # Phase 2: Try MILP to see if we can do better
    if time.time() < deadline - 120.0:
        milp_result = _try_milp(deadline)
        if milp_result is not None:
            milp_shrunk = _shrink(milp_result)
            if len(milp_shrunk) < best_size:
                best = milp_shrunk
                best_size = len(milp_shrunk)
    
    # Phase 3: Iterated Local Search
    plateau = 0
    strength = 1
    while time.time() < deadline:
        S = set(best)
        r = random.random()
        if r < 0.4:
            k = random.randint(1, max(1, len(S) // 4 + strength))
            for _ in range(k):
                if not S:
                    break
                x = random.choice(list(S))
                S.discard(x)
        elif r < 0.7:
            if S:
                x = random.choice(list(S))
                S.discard(x)
                inS = [False] * M
                for v in S:
                    inS[v] = True
                for _ in range(15):
                    y = random.randrange(M)
                    if y in S:
                        continue
                    compatible = True
                    for a, b in COVER[y]:
                        if inS[a] and inS[b]:
                            compatible = False
                            break
                    if compatible:
                        S.add(y)
                        inS[y] = True
                        break
        else:
            if len(S) >= 2:
                x1 = random.choice(list(S))
                S.discard(x1)
                x2 = random.choice(list(S))
                S.discard(x2)
                inS = [False] * M
                for v in S:
                    inS[v] = True
                for _ in range(20):
                    y = random.randrange(M)
                    if y in S:
                        continue
                    compatible = True
                    for a, b in COVER[y]:
                        if inS[a] and inS[b]:
                            compatible = False
                            break
                    if compatible:
                        S.add(y)
                        inS[y] = True
                        break
        
        S = _repair(S)
        S = _shrink(S)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            plateau = 0
            strength = 1
        else:
            plateau += 1
            if plateau % 15 == 0:
                strength = min(strength + 1, 5)
    
    # Final shrink
    if time.time() < deadline - 2.0:
        best = _shrink(best)
    
    return [POINTS[i] for i in sorted(best)]
