import time
import math
import numpy as np

DEADLINE = 135.0

def _score(U):
    """Compute score efficiently."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 1.0
    maxU = int(U.max())
    if maxU == 0:
        return 1.0
    # Use set-based computation for small sets, numpy for larger
    if n <= 50:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(int(a) + int(b))
                diffs.add(int(a) - int(b))
        ns = len(sums)
        nd = len(diffs)
    else:
        sums = np.add.outer(U, U).ravel()
        diffs = np.subtract.outer(U, U).ravel()
        ns = len(np.unique(sums))
        nd = len(np.unique(diffs))
    ratio = nd / ns
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * maxU + 1)

def construct():
    t0 = time.time()
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Strategy 1: Geometric progression with tunable base b
    # U = {floor(b^k) | k=0,...,m}
    # Sweep over b and m to find the best structured set
    
    def eval_geo(b, m):
        """Evaluate geometric progression {floor(b^k) | k=0..m}"""
        U = []
        val = 1
        for k in range(m + 1):
            U.append(int(val))
            val *= b
            if val > 10**7:
                break
        U = sorted(set(U))
        if len(U) < 2 or max(U) == 0:
            return 1.0, U
        return _score(U), U
    
    # Sweep over base b (continuous) and length m
    # For each m, find optimal b via ternary search
    
    for m in range(2, 30):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # b ranges from just above 1 to some upper bound
        # max element ~ b^m, want max element reasonable
        # Try a range of b values
        b_low = 1.01
        b_high = 10.0
        
        # First, do a coarse grid search to find the best region
        best_b = None
        best_s = -1
        
        # Log-spaced grid for b
        n_grid = 50
        b_values = np.geomspace(b_low, b_high, n_grid)
        
        for b in b_values:
            if time.time() - t0 > DEADLINE - 30:
                break
            s, U = eval_geo(b, m)
            if s > best_s:
                best_s = s
                best_b = b
        
        if best_b is not None and best_s > best_score:
            # Refine with ternary search around best_b
            # Determine neighborhood
            idx = np.argmin(np.abs(b_values - best_b))
            if idx > 0:
                left = b_values[max(0, idx - 1)]
            else:
                left = b_low
            if idx < len(b_values) - 1:
                right = b_values[min(len(b_values) - 1, idx + 1)]
            else:
                right = b_high
            
            # Ternary search
            a, c = left, right
            for _ in range(20):
                if time.time() - t0 > DEADLINE - 30:
                    break
                m1 = a + (c - a) / 3
                m2 = c - (c - a) / 3
                s1, _ = eval_geo(m1, m)
                s2, _ = eval_geo(m2, m)
                if s1 < s2:
                    a = m1
                else:
                    c = m2
            
            b_opt = (a + c) / 2
            s, U = eval_geo(b_opt, m)
            if s > best_score:
                best_score = s
                best_set = U
        
        # Also try integer bases
        for b_int in range(2, 12):
            if time.time() - t0 > DEADLINE - 30:
                break
            s, U = eval_geo(float(b_int), m)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Strategy 2: Structured families from the paper's constructions
    # The paper mentions sizes 2003 and 54265
    # Try sets based on modular arithmetic patterns
    
    # Family: U = {k * step for k in range(n)} union some offsets
    for step in range(1, 20):
        for n in range(5, 50):
            if time.time() - t0 > DEADLINE - 20:
                break
            U = list(range(0, n * step, step))
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Family: U = {floor(a * b^k) for k in range(m)} with offset
    for a in [1, 2, 3]:
        for b in [2, 3, 4, 5]:
            for m in range(2, 25):
                if time.time() - t0 > DEADLINE - 20:
                    break
                U = []
                val = a
                for k in range(m):
                    U.append(int(val))
                    val *= b
                    if val > 10**6:
                        break
                U = sorted(set(U))
                if 0 not in U:
                    U.insert(0, 0)
                if len(U) < 2:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # Strategy 3: Local search refinement
    if time.time() - t0 < DEADLINE - 15:
        current = best_set[:]
        cur_score = best_score
        max_iter = 50
        for _ in range(max_iter):
            if time.time() - t0 > DEADLINE - 15:
                break
            improved = False
            maxU = max(current)
            
            # Try adding elements
            for new_el in range(1, maxU + 5):
                if new_el not in current:
                    U_new = current + [new_el]
                    s = _score(U_new)
                    if s > cur_score + 1e-12:
                        current = U_new
                        cur_score = s
                        improved = True
                        break
            
            # Try removing elements
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    U_new = current[:i] + current[i+1:]
                    if len(U_new) < 2:
                        continue
                    s = _score(U_new)
                    if s > cur_score + 1e-12:
                        current = U_new
                        cur_score = s
                        improved = True
                        break
            
            # Try swapping elements
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(1, maxU + 3):
                        if new_el not in current:
                            U_new = current[:]
                            U_new[i] = new_el
                            s = _score(U_new)
                            if s > cur_score + 1e-12:
                                current = U_new
                                cur_score = s
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = current
    
    # Strategy 4: Try paper-inspired constructions
    # Size 2003 construction
    if time.time() - t0 < DEADLINE - 10:
        # Try a set with ~2000 elements using a structured pattern
        # U = {k * d for k in range(2000)} for various d
        for d in [1, 2, 3, 5, 7, 11]:
            U = list(range(0, 2000 * d, d))
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Strategy 5: Random restarts with geometric structure
    if time.time() - t0 < DEADLINE - 5:
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() - t0 > DEADLINE - 5:
                break
            # Random geometric progression
            b = random.uniform(1.1, 5.0)
            m = random.randint(3, 20)
            U = []
            val = 1
            for k in range(m):
                U.append(int(val))
                val *= b
                if val > 10**5:
                    break
            U = sorted(set(U))
            if 0 not in U:
                U.insert(0, 0)
            if len(U) < 2:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    return best_set
