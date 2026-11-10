# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    alpha_sq = 3

    # Precompute field operations
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Precompute line_other[idx, d, t]
    line_other = np.zeros((q * q, 50, 48), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            for d in range(49):
                for t_idx in range(48):
                    t = t_idx + 1
                    xp = add_table[x, t]
                    yd = mul_table[t, d]
                    yp = add_table[y, yd]
                    line_other[idx, d, t_idx] = xp * q + yp
            for t_idx in range(48):
                t = t_idx + 1
                yp = add_table[y, t]
                line_other[idx, 49, t_idx] = x * q + yp

    # Precompute reverse_map: for each point p, list of (x, d) where p is on line(x,d)
    # Use a more compact representation
    reverse_x = []  # For each p, list of x values
    reverse_d = []  # For each p, list of d values
    
    for x in range(q * q):
        for d in range(50):
            for t_idx in range(48):
                p = line_other[x, d, t_idx]
                if p != x:
                    if not reverse_x or len(reverse_x) <= p:
                        while len(reverse_x) <= p:
                            reverse_x.append([])
                            reverse_d.append([])
                    reverse_x[p].append(x)
                    reverse_d[p].append(d)
    
    # Convert to numpy arrays for faster access
    reverse_x_arr = []
    reverse_d_arr = []
    for p in range(q * q):
        reverse_x_arr.append(np.array(reverse_x[p] if p < len(reverse_x) else [], dtype=np.int32))
        reverse_d_arr.append(np.array(reverse_d[p] if p < len(reverse_d) else [], dtype=np.int32))

    start_time = time.time()
    deadline = start_time + 580

    def try_greedy_remove(order, deadline):
        N = np.ones(q * q, dtype=bool)
        cover_count = np.full(q * q, 50, dtype=np.int32)
        missing = np.zeros((q * q, 50), dtype=np.int32)
        
        for p in order:
            if time.time() > deadline:
                break
            if not N[p]:
                continue
            
            # Check if removing p is safe
            xs = reverse_x_arr[p]
            ds = reverse_d_arr[p]
            
            if len(xs) == 0:
                can_remove = True
            else:
                # Check: for any (x,d) in reverse_map[p], if missing[x,d]==0 and cover_count[x]==1, then not safe
                # Vectorized check
                m_vals = missing[xs, ds]
                c_vals = cover_count[xs]
                unsafe = np.any((m_vals == 0) & (c_vals == 1))
                can_remove = not unsafe
            
            if can_remove:
                N[p] = False
                # Update missing and cover_count
                m_vals = missing[xs, ds]
                was_zero = (m_vals == 0)
                missing[xs, ds] += 1
                cover_count[xs[was_zero]] -= 1
        
        return N

    # Try multiple random orders
    best_N = None
    best_complement = 0
    
    rng = np.random.default_rng(123)
    
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = rng.permutation(q * q)
        N = try_greedy_remove(order, deadline)
        complement = np.sum(~N)
        
        if complement > best_complement:
            best_complement = complement
            best_N = N.copy()

    if best_N is None:
        best_N = np.ones(q * q, dtype=bool)

    # Verify
    valid = True
    for x in range(q * q):
        has_cover = False
        for d in range(50):
            if np.all(best_N[line_other[x, d, :]]):
                has_cover = True
                break
        if not has_cover:
            valid = False
            break

    if not valid:
        best_N = np.ones(q * q, dtype=bool)

    points = []
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            if best_N[idx]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
