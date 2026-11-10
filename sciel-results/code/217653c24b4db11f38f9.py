import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 610
    
    p = 7
    k = 3
    q = p * p
    n_points = q * q
    n_lines = q * (q + 1)
    
    def to_ab(idx):
        return idx % p, idx // p
    
    def from_ab(a, b):
        return a + p * b
    
    add_t = np.zeros((q, q), dtype=np.int32)
    sub_t = np.zeros((q, q), dtype=np.int32)
    mul_t = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        a1, b1 = to_ab(i)
        for j in range(q):
            a2, b2 = to_ab(j)
            add_t[i, j] = from_ab((a1+a2)%p, (b1+b2)%p)
            sub_t[i, j] = from_ab((a1-a2)%p, (b1-b2)%p)
            ra = (a1*a2 + b1*b2*k) % p
            rb = (a1*b2 + a2*b1) % p
            mul_t[i, j] = from_ab(ra, rb)
    
    point_lines = np.zeros((n_points, q+1), dtype=np.int32)
    for x_idx in range(q):
        for y_idx in range(q):
            pt = x_idx + q * y_idx
            point_lines[pt, 0] = q*q + x_idx
            for m in range(q):
                mx = mul_t[m, x_idx]
                b = sub_t[y_idx, mx]
                point_lines[pt, m+1] = m * q + b
    
    line_pts = [[] for _ in range(n_lines)]
    for pt in range(n_points):
        for lid in point_lines[pt]:
            line_pts[lid].append(pt)
    line_pts = [np.array(lp, dtype=np.int32) for lp in line_pts]
    
    def greedy_construct(order, deadline):
        line_count = np.zeros(n_lines, dtype=np.int32)
        safe_lines = np.full(n_points, q+1, dtype=np.int32)
        in_S = np.zeros(n_points, dtype=bool)
        S = []
        
        for pt in order:
            if time.time() > deadline:
                break
            if in_S[pt] or safe_lines[pt] <= 0:
                continue
            
            in_S[pt] = True
            S.append(pt)
            
            for lid in point_lines[pt]:
                old = line_count[lid]
                line_count[lid] = old + 1
                if old == 0:
                    others = line_pts[lid]
                    for op in others:
                        if op != pt and not in_S[op]:
                            safe_lines[op] -= 1
        
        return in_S, len(S)
    
    def validate(in_S):
        for x in range(n_points):
            found = False
            for lid in point_lines[x]:
                others = line_pts[lid]
                has_other = False
                for op in others:
                    if op != x and in_S[op]:
                        has_other = True
                        break
                if not has_other:
                    found = True
                    break
            if not found:
                return False
        return True
    
    best_S_size = 0
    best_in_S = np.zeros(n_points, dtype=bool)
    
    rng = np.random.RandomState(42)
    orders = [
        list(range(n_points)),
        list(range(n_points, 0, -1)),
    ]
    for _ in range(3):
        o = list(range(n_points))
        rng.shuffle(o)
        orders.append(o)
    
    for trial, order in enumerate(orders):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        in_S, S_size = greedy_construct(order, DEADLINE - 30)
        
        if S_size > best_S_size:
            if validate(in_S):
                best_S_size = S_size
                best_in_S = in_S.copy()
    
    # Fallback: original construction if no valid S found
    if best_S_size == 0:
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        
        N_points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        N_points.append([[xa, xb], [ya, yb]])
        return np.array(N_points, dtype=np.int64)
    
    N_points = []
    for pt in range(n_points):
        if not best_in_S[pt]:
            x_idx = pt % q
            y_idx = pt // q
            xa, xb = to_ab(x_idx)
            ya, yb = to_ab(y_idx)
            N_points.append([[xa, xb], [ya, yb]])
    
    if len(N_points) == 0:
        N_points = [[[0, 0], [0, 0]]]
    
    return np.array(N_points, dtype=np.int64)
