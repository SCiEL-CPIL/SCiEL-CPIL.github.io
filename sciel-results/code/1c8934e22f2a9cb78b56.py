"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_3block(k1, k2, a, b, c):
    """Compute max overlap for 3-block: [a]*k1 + [b]*k2 + [c]*k3."""
    k3 = N - k1 - k2
    if k1 <= 0 or k2 <= 0 or k3 <= 0:
        return float('inf')
    vals = []
    if k1 >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if k2 >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b * (1.0 - c))
    if k3 >= 2:
        vals.append(c * (1.0 - c))
    vals.append(c)
    return max(vals)


def _overlap_for_4block(k1, k2, k3, a, b, c, d):
    """Compute max overlap for 4-block: [a]*k1 + [b]*k2 + [c]*k3 + [d]*k4."""
    k4 = N - k1 - k2 - k3
    if k1 <= 0 or k2 <= 0 or k3 <= 0 or k4 <= 0:
        return float('inf')
    vals = []
    if k1 >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if k2 >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b * (1.0 - c))
    if k3 >= 2:
        vals.append(c * (1.0 - c))
    vals.append(c * (1.0 - d))
    if k4 >= 2:
        vals.append(d * (1.0 - d))
    vals.append(d)
    return max(vals)


def _optimize_3block(k1, k2):
    """Optimize 3-block with given k1, k2. Returns (score, a, b, c) or None."""
    k3 = N - k1 - k2
    if k3 <= 0:
        return None
    
    # Constraint: k1*a + k2*b + k3*c = N/2
    # Two free parameters, one constraint -> 2D optimization
    # Use grid search + refinement
    
    best_score = float('inf')
    best_abc = None
    
    # Grid over a and b, compute c from constraint
    n_grid = 50
    a_vals = np.linspace(0.0, 1.0, n_grid)
    b_vals = np.linspace(0.0, 1.0, n_grid)
    
    for a in a_vals:
        for b in b_vals:
            c = (N / 2.0 - k1 * a - k2 * b) / k3
            if c < -1e-12 or c > 1.0 + 1e-12:
                continue
            c = np.clip(c, 0.0, 1.0)
            s = _overlap_for_3block(k1, k2, a, b, c)
            if s < best_score:
                best_score = s
                best_abc = (a, b, c)
    
    if best_abc is None:
        return None
    
    # Refine around best
    a0, b0, c0 = best_abc
    step = 1.0 / n_grid
    for da in np.linspace(-step*2, step*2, 30):
        for db in np.linspace(-step*2, step*2, 30):
            a = a0 + da
            b = b0 + db
            if a < 0 or a > 1 or b < 0 or b > 1:
                continue
            c = (N / 2.0 - k1 * a - k2 * b) / k3
            if c < -1e-12 or c > 1.0 + 1e-12:
                continue
            c = np.clip(c, 0.0, 1.0)
            s = _overlap_for_3block(k1, k2, a, b, c)
            if s < best_score:
                best_score = s
                best_abc = (a, b, c)
    
    return (best_score, best_abc)


def _optimize_4block(k1, k2, k3):
    """Optimize 4-block with given k1, k2, k3. Returns (score, abcd) or None."""
    k4 = N - k1 - k2 - k3
    if k4 <= 0:
        return None
    
    best_score = float('inf')
    best_abcd = None
    
    # Grid over a, b, c, compute d from constraint
    n_grid = 30
    a_vals = np.linspace(0.0, 1.0, n_grid)
    b_vals = np.linspace(0.0, 1.0, n_grid)
    c_vals = np.linspace(0.0, 1.0, n_grid)
    
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                d = (N / 2.0 - k1 * a - k2 * b - k3 * c) / k4
                if d < -1e-12 or d > 1.0 + 1e-12:
                    continue
                d = np.clip(d, 0.0, 1.0)
                s = _overlap_for_4block(k1, k2, k3, a, b, c, d)
                if s < best_score:
                    best_score = s
                    best_abcd = (a, b, c, d)
    
    if best_abcd is None:
        return None
    
    # Refine
    a0, b0, c0, d0 = best_abcd
    step = 1.0 / n_grid
    for da in np.linspace(-step*1.5, step*1.5, 15):
        for db in np.linspace(-step*1.5, step*1.5, 15):
            for dc in np.linspace(-step*1.5, step*1.5, 15):
                a = a0 + da
                b = b0 + db
                c = c0 + dc
                if a < 0 or a > 1 or b < 0 or b > 1 or c < 0 or c > 1:
                    continue
                d = (N / 2.0 - k1 * a - k2 * b - k3 * c) / k4
                if d < -1e-12 or d > 1.0 + 1e-12:
                    continue
                d = np.clip(d, 0.0, 1.0)
                s = _overlap_for_4block(k1, k2, k3, a, b, c, d)
                if s < best_score:
                    best_score = s
                    best_abcd = (a, b, c, d)
    
    return (best_score, best_abcd)


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill

    best_h = None
    best_score = float('inf')

    # --- 3-block search ---
    # Try various (k1, k2) splits
    for k1 in range(1, N // 2 + 1):
        for k2 in range(1, N - k1):
            k3 = N - k1 - k2
            if k3 <= 0:
                continue
            if time.time() > deadline:
                break
            
            result = _optimize_3block(k1, k2)
            if result is None:
                continue
            score, (a, b, c) = result
            if score < best_score:
                best_score = score
                h = np.zeros(N)
                h[:k1] = a
                h[k1:k1+k2] = b
                h[k1+k2:] = c
                # Fix sum
                s = h.sum()
                diff = N / 2.0 - s
                if abs(diff) > 1e-15:
                    h[k1 + k2 - 1] += diff
                    h = np.clip(h, 0.0, 1.0)
                best_h = h.copy()
        
        if time.time() > deadline:
            break

    # --- 4-block search (fewer splits to keep within time) ---
    for k1 in range(1, N // 4 + 1):
        for k2 in range(1, N // 3):
            for k3 in range(1, N - k1 - k2):
                k4 = N - k1 - k2 - k3
                if k4 <= 0:
                    continue
                if time.time() > deadline:
                    break
                
                result = _optimize_4block(k1, k2, k3)
                if result is None:
                    continue
                score, (a, b, c, d) = result
                if score < best_score:
                    best_score = score
                    h = np.zeros(N)
                    h[:k1] = a
                    h[k1:k1+k2] = b
                    h[k1+k2:k1+k2+k3] = c
                    h[k1+k2+k3:] = d
                    s = h.sum()
                    diff = N / 2.0 - s
                    if abs(diff) > 1e-15:
                        h[k1+k2+k3-1] += diff
                        h = np.clip(h, 0.0, 1.0)
                    best_h = h.copy()
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_h is None:
        # Fallback: 2-block with golden ratio
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = h

    return best_h
