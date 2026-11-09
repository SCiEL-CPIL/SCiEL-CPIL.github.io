import time
import numpy as np
from itertools import combinations

def _count_bad_4sets(orient, n):
    """Count 4-sets without a common dominator using vectorized ops."""
    # Build adjacency matrix: A[i,j]=1 if i->j
    A = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if orient[idx] == 0:
                A[i, j] = 1
            else:
                A[j, i] = 1
            idx += 1
    
    bad_count = 0
    for a, b, c, d in combinations(range(n), 4):
        # Check if any of a,b,c,d dominates the other three
        # a dominates b,c,d if A[a,b]=1, A[a,c]=1, A[a,d]=1
        if A[a,b] and A[a,c] and A[a,d]:
            continue
        if A[b,a] and A[b,c] and A[b,d]:
            continue
        if A[c,a] and A[c,b] and A[c,d]:
            continue
        if A[d,a] and A[d,b] and A[d,c]:
            continue
        bad_count += 1
    return bad_count

def _try_repair(n, max_time, rng):
    """Try to find a valid tournament on n vertices via random init + local repair."""
    m = n * (n - 1) // 2
    orient = rng.integers(0, 2, size=m, dtype=np.uint8).tolist()
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if orient[idx] == 0:
                A[i, j] = 1
            else:
                A[j, i] = 1
            idx += 1
    
    def count_bad():
        bad = 0
        for a, b, c, d in combinations(range(n), 4):
            if A[a,b] and A[a,c] and A[a,d]: continue
            if A[b,a] and A[b,c] and A[b,d]: continue
            if A[c,a] and A[c,b] and A[c,d]: continue
            if A[d,a] and A[d,b] and A[d,c]: continue
            bad += 1
        return bad
    
    current_bad = count_bad()
    if current_bad == 0:
        return orient
    
    # Local search: flip edges that reduce bad count
    # Precompute which 4-sets each edge belongs to
    # Edge (i,j) with i<j is at index: i*n - i*(i+1)//2 + (j - i - 1)
    def edge_index(i, j):
        if i > j: i, j = j, i
        return i * n - i*(i+1)//2 + (j - i - 1)
    
    # For each edge, list of 4-sets it's in
    edge_to_4sets = [[] for _ in range(m)]
    for a, b, c, d in combinations(range(n), 4):
        for (u, v) in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]:
            ei = edge_index(min(u,v), max(u,v))
            edge_to_4sets[ei].append((a,b,c,d))
    
    # Greedy local search
    no_improve = 0
    while time.time() < max_time and current_bad > 0:
        improved = False
        # Try flipping each edge, pick best improvement
        best_ei = -1
        best_new_bad = current_bad
        for ei in range(m):
            i = ei
            # Find i,j from ei
            # ei = i*n - i*(i+1)//2 + (j-i-1)
            # Solve for i: quadratic
            # Simpler: iterate
            ii = 0
            found = False
            for ii in range(n):
                base = ii * n - ii*(ii+1)//2
                if base <= ei < base + (n - ii - 1):
                    jj = ii + 1 + (ei - base)
                    found = True
                    break
            if not found:
                continue
            
            # Flip edge ei
            old_val = orient[ei]
            if old_val == 0:
                A[i if ii<=jj else ii, jj if ii<=jj else ii] = 0
                A[jj if ii<=jj else ii, ii if ii<=jj else ii] = 1
            else:
                A[ii, jj] = 1
                A[jj, ii] = 0
            
            # Count change in bad 4sets
            delta = 0
            for (a,b,c,d) in edge_to_4sets[ei]:
                # Check if this 4set was bad before and after
                was_bad = not (A[a,b] and A[a,c] and A[a,d] or
                             A[b,a] and A[b,c] and A[b,d] or
                             A[c,a] and A[c,b] and A[c,d] or
                             A[d,a] and A[d,b] and A[d,c])
                # Note: A has already been flipped, so this checks "after"
                # We need "before" too - this is complex. Let's just recount.
                pass
            
            # Undo flip
            if old_val == 0:
                A[ii, jj] = 1
                A[jj, ii] = 0
            else:
                A[ii, jj] = 0
                A[jj, ii] = 1
            
            # Just do a full recount for simplicity (slow but correct)
            # Actually this is too slow. Let's use a different approach.
            break  # Skip this approach
        
        # Simpler: just try random flips and keep if improves
        for _ in range(100):
            if time.time() >= max_time:
                break
            ei = rng.integers(0, m)
            ii = 0
            for ii in range(n):
                base = ii * n - ii*(ii+1)//2
                if base <= ei < base + (n - ii - 1):
                    jj = ii + 1 + (ei - base)
                    break
            
            old_val = orient[ei]
            orient[ei] = 1 - old_val
            if old_val == 0:
                A[ii, jj] = 0
                A[jj, ii] = 1
            else:
                A[ii, jj] = 1
                A[jj, ii] = 0
            
            new_bad = count_bad()
            if new_bad < current_bad:
                current_bad = new_bad
                improved = True
                no_improve = 0
            else:
                # Undo
                orient[ei] = old_val
                if old_val == 0:
                    A[ii, jj] = 1
                    A[jj, ii] = 0
                else:
                    A[ii, jj] = 0
                    A[jj, ii] = 1
        
        if not improved:
            no_improve += 1
            if no_improve > 20:
                break
    
    if current_bad == 0:
        return orient
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try probabilistic construction for small n
    rng = np.random.default_rng(42)
    
    for n in range(31, 50):
        if time.time() > deadline:
            break
        # Allocate proportional time
        remaining = deadline - time.time()
        alloc = min(remaining * 0.5, 20)  # At most 20s per n
        if alloc < 5:
            break
        result = _try_repair(n, start_time + alloc, rng)
        if result is not None:
            return {'n': n, 'orientation': result}
    
    # Fallback: Paley tournament of order 67
    p = 67
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
