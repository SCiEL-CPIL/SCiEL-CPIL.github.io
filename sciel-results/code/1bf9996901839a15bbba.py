import time
import random
import math
import numpy as np

_DEADLINE = 0.0
_BEST = None

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _global_cost(matrix, k, n_rows, target_scope, lam, mu):
    """Compute cost: scope + lam*collisions + mu*(max-target)^2 using vectorized diffs."""
    # matrix shape: (n_rows, k+1)
    max_val = int(matrix.max())
    scope = max_val
    
    # Compute all pairwise differences within each row
    # For row i, diffs are matrix[i, j] - matrix[i, i] for j > i
    all_diffs = []
    for i in range(n_rows):
        row = matrix[i]
        for a in range(k + 1):
            for b in range(a + 1, k + 1):
                all_diffs.append(row[b] - row[a])
    all_diffs = np.array(all_diffs, dtype=np.int32)
    
    # Count collisions
    if len(all_diffs) > 0:
        _, counts = np.unique(all_diffs, return_counts=True)
        collisions = int(np.sum(np.maximum(0, counts - 1)))
    else:
        collisions = 0
    
    # Soft penalty for exceeding target
    excess = max(0, scope - target_scope)
    scope_penalty = mu * excess * excess
    
    return scope + lam * collisions + scope_penalty

def _sa_vectorized(rows, k, n_rows, deadline, rng, iters_cap=200000):
    """Global vectorized SA over the full matrix."""
    global _BEST
    
    matrix = np.array(rows, dtype=np.int32)
    current_scope = int(matrix.max())
    
    # Initialize target for soft penalty
    target_scope = current_scope - 2  # encourage going below current
    
    lam = 200.0  # collision penalty weight
    mu = 5.0     # scope soft penalty weight
    
    def compute_cost(mat):
        return _global_cost(mat, k, n_rows, target_scope, lam, mu)
    
    current_cost = compute_cost(matrix)
    
    best_matrix = matrix.copy()
    best_cost = current_cost
    best_scope = current_scope
    
    # Precompute diff indices for quick update
    # For each row r, positions (a, b) with a < b
    diff_pairs = []
    for a in range(k + 1):
        for b in range(a + 1, k + 1):
            diff_pairs.append((a, b))
    
    T = 10.0
    cooling = 0.9997
    min_T = 0.05
    
    for it in range(iters_cap):
        if time.time() > deadline:
            break
        
        r = rng.random()
        
        if r < 0.30:
            # Single element move: change one element in one row
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)  # don't change position 0 (must be 0)
            
            old_val = int(matrix[i, j])
            prev_val = int(matrix[i, j - 1])
            next_val = int(matrix[i, j + 1]) if j < k else int(matrix[i, k]) + 100
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            # Gaussian perturbation
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.3)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
            
            if new_val == old_val:
                continue
            
            matrix[i, j] = new_val
            new_cost = compute_cost(matrix)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = int(matrix.max())
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_matrix = matrix.copy()
                    best_scope = new_scope
                    # Adapt target
                    if new_scope < target_scope:
                        target_scope = new_scope
            else:
                matrix[i, j] = old_val
                
        elif r < 0.50:
            # Row pair swap: exchange two entire rows
            if n_rows < 2:
                continue
            i, j = rng.sample(range(n_rows), 2)
            
            matrix[[i, j]] = matrix[[j, i]]
            new_cost = compute_cost(matrix)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = int(matrix.max())
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_matrix = matrix.copy()
                    best_scope = new_scope
            else:
                matrix[[i, j]] = matrix[[j, i]]
                
        elif r < 0.70:
            # Row rebuild: reconstruct one row from scratch using greedy with awareness
            i = rng.randint(0, n_rows - 1)
            old_row = matrix[i].copy()
            
            # Remove row i's diffs from consideration (we'll rebuild)
            # Build a "used" set from all other rows
            used = set()
            for ri in range(n_rows):
                if ri == i:
                    continue
                row = matrix[ri]
                for a in range(k + 1):
                    for b in range(a + 1, k + 1):
                        used.add(int(row[b] - row[a]))
            
            # Greedily build new row
            new_row = [0]
            x = 0
            ok = True
            for _pos in range(1, k + 1):
                found = False
                for t in range(x + 1, x + 200):
                    conflict = False
                    for a in new_row:
                        d = t - a
                        if d <= 0:
                            continue
                        if d in used:
                            conflict = True
                            break
                    if not conflict:
                        new_row.append(t)
                        for a in new_row[:-1]:
                            used.add(t - a)
                        x = t
                        found = True
                        break
                if not found:
                    ok = False
                    break
            
            if not ok:
                continue
            
            new_row_arr = np.array(new_row, dtype=np.int32)
            matrix[i] = new_row_arr
            new_cost = compute_cost(matrix)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = int(matrix.max())
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_matrix = matrix.copy()
                    best_scope = new_scope
            else:
                matrix[i] = old_row
                
        elif r < 0.85:
            # Scope compression: try to reduce the max element
            max_val = int(matrix.max())
            # Find all positions with max value
            max_positions = np.argwhere(matrix == max_val)
            if len(max_positions) == 0:
                continue
            
            ri, pj = max_positions[0]
            old_val = int(matrix[ri, pj])
            prev_val = int(matrix[ri, pj - 1]) if pj > 0 else 0
            
            new_val = rng.randint(prev_val + 1, old_val - 1)
            if new_val <= prev_val or new_val >= old_val:
                continue
            
            matrix[ri, pj] = new_val
            new_cost = compute_cost(matrix)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = int(matrix.max())
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_matrix = matrix.copy()
                    best_scope = new_scope
            else:
                matrix[ri, pj] = old_val
                
        else:
            # Multi-element row shift: shift a contiguous block in a row
            i = rng.randint(0, n_rows - 1)
            start = rng.randint(1, k)
            end = rng.randint(start, k)
            shift = rng.randint(-3, 3)
            if shift == 0:
                continue
            
            old_block = matrix[i, start:end + 1].copy()
            new_block = old_block + shift
            
            # Check constraints: increasing within row
            if start > 0 and new_block[0] <= int(matrix[i, start - 1]):
                continue
            if end < k and new_block[-1] >= int(matrix[i, end + 1]):
                continue
            # Check strictly increasing within block
            for bi in range(1, len(new_block)):
                if new_block[bi] <= new_block[bi - 1]:
                    break
            else:
                matrix[i, start:end + 1] = new_block
                new_cost = compute_cost(matrix)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = int(matrix.max())
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_matrix = matrix.copy()
                        best_scope = new_scope
                else:
                    matrix[i, start:end + 1] = old_block
        
        T = max(T * cooling, min_T)
        
        # Check if best is valid and update global
        if best_scope < current_scope and it % 1000 == 0:
            best_rows = best_matrix.tolist()
            if _valid(best_rows, k):
                if _BEST is None or best_scope < _BEST[0]:
                    _BEST = (best_scope, best_rows)
    
    # Final check
    best_rows = best_matrix.tolist()
    if _valid(best_rows, k):
        if _BEST is None or best_scope < _BEST[0]:
            _BEST = (best_scope, best_rows)
    
    return best_scope, best_rows

def _polish_pass(rows, k, deadline, rng):
    """Final high-precision polish: try small perturbations on valid solution."""
    global _BEST
    
    if not _valid(rows, k):
        return rows
    
    current_scope = _scope(rows)
    matrix = np.array(rows, dtype=np.int32)
    n_rows = len(rows)
    
    # For each element, try small shifts
    for i in range(n_rows):
        for j in range(1, k + 1):
            old_val = int(matrix[i, j])
            prev_val = int(matrix[i, j - 1])
            next_val = int(matrix[i, j + 1]) if j < k else old_val + 100
            
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= prev_val or new_val >= next_val:
                    continue
                
                matrix[i, j] = new_val
                new_rows = matrix.tolist()
                if _valid(new_rows, k):
                    new_scope = _scope(new_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if _BEST is None or new_scope < _BEST[0]:
                            _BEST = (new_scope, new_rows)
                else:
                    matrix[i, j] = old_val
                if time.time() > deadline:
                    return matrix.tolist()
    
    return matrix.tolist()

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate multiple initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
    
    candidates.sort(key=_scope)
    
    # Run vectorized SA on top candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE - 5:
            break
        
        remaining = _DEADLINE - 5 - time.time()
        if remaining < 10:
            break
        
        sa_deadline = time.time() + remaining * (0.5 if idx == 0 else 0.35 if idx == 1 else 0.15)
        iters = int((sa_deadline - time.time()) * 5000)
        iters = max(iters, 5000)
        
        s, rows = _sa_vectorized(cand, k, n_rows, sa_deadline, rng, iters)
        
        if _valid(rows, k):
            sc = _scope(rows)
            if _BEST is None or sc < _BEST[0]:
                _BEST = (sc, [list(r) for r in rows])
    
    # Polish pass
    if _BEST is not None and time.time() < _DEADLINE - 3:
        polish_deadline = _DEADLINE - 3
        rows = _polish_pass(_BEST[1], k, polish_deadline, rng)
        if _valid(rows, k):
            sc = _scope(rows)
            if _BEST is None or sc < _BEST[0]:
                _BEST = (sc, [list(r) for r in rows])
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
