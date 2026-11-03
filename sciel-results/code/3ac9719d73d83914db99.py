import time
import math
import random
import numpy as np
from itertools import product

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    # Compute all sums and differences using broadcasting
    # For small n, this is fast
    n = len(U)
    if n > 30:
        # Fall back to set-based for large n
        sums_set = set()
        diffs_set = set()
        for a in U:
            for b in U:
                sums_set.add(a + b)
                diffs_set.add(a - b)
        ratio = len(diffs_set) / len(sums_set)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    # Vectorized: outer sum and difference
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def geometric_score(b, m):
    """Score for geometric progression U = {floor(b^k) : k=0..m}"""
    if b <= 1 or m < 1:
        return -1.0
    U = list(set([int(b**k) for k in range(m+1)]))
    U.sort()
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    return fast_score(U)

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Systematic exploration of geometric progressions with ternary search
    # For each m, find optimal b via ternary search
    
    for m in range(2, 30):
        if time.time() > deadline:
            return best_set
        
        # Determine reasonable range for b
        # max element ~ b^m, want b^m < 500000
        b_max = 500000 ** (1.0/m)
        b_min = 1.01
        
        # Coarse grid search first
        n_coarse = 50
        best_b_coarse = b_min
        best_score_coarse = -1.0
        
        for i in range(n_coarse + 1):
            b = b_min + (b_max - b_min) * i / n_coarse
            s = geometric_score(b, m)
            if s > best_score_coarse:
                best_score_coarse = s
                best_b_coarse = b
        
        # Ternary search refinement
        lo = max(b_min, best_b_coarse - (b_max - b_min) / n_coarse * 2)
        hi = min(b_max, best_b_coarse + (b_max - b_min) / n_coarse * 2)
        
        for _ in range(30):
            if time.time() > deadline:
                return best_set
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            s1 = geometric_score(m1, m)
            s2 = geometric_score(m2, m)
            if s1 < s2:
                lo = m1
            else:
                hi = m2
        
        b_opt = (lo + hi) / 2
        s_opt = geometric_score(b_opt, m)
        if s_opt > best_score:
            best_score = s_opt
            U = list(set([int(b_opt**k) for k in range(m+1)]))
            U.sort()
            if 0 not in U:
                U.insert(0, 0)
            best_set = sorted(set(U))
    
    # Phase 2: Fine-tune around the best geometric progression found
    # Local search on the geometric sequence parameters
    for _ in range(100):
        if time.time() > deadline:
            return best_set
        
        # Try small perturbations of each element
        U = sorted(best_set)
        improved = False
        for i in range(len(U)):
            if time.time() > deadline:
                return best_set
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                new_val = U[i] + delta
                if new_val < 0:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 3: Simulated annealing on the best set
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.998
    
    for iteration in range(30000):
        if time.time() > deadline:
            return best_set
        
        if T < T_min:
            T = T_min
        
        n = len(U)
        
        # Random move: modify one element, add, or remove
        move_type = random.random()
        
        if move_type < 0.6 and n > 2:
            # Modify a random element (not 0)
            idx = random.randint(1, n-1)
            old_val = U[idx]
            # Propose new value near old or random
            if random.random() < 0.5:
                new_val = max(0, old_val + random.randint(-3, 3))
            else:
                new_val = random.randint(0, max(U) + 5)
            
            if new_val == old_val:
                continue
            
            new_U = U.copy()
            new_U[idx] = new_val
            # Ensure uniqueness
            if len(set(new_U)) != len(new_U):
                continue
            
            s = fast_score(new_U)
            delta = s - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
        
        elif move_type < 0.85:
            # Add a new element
            if n < 20:
                new_val = random.randint(1, max(U) + 10)
                if new_val not in U:
                    new_U = np.append(U, new_val)
                    new_U = np.sort(new_U)
                    s = fast_score(new_U)
                    delta = s - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        U = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = sorted(U.tolist())
        
        else:
            # Remove an element (not 0)
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
        
        T *= cooling
    
    # Phase 4: Final local polish
    U = sorted(best_set)
    for iteration in range(1000):
        if time.time() > deadline:
            return best_set
        
        improved = False
        n = len(U)
        
        # Try modifying each element
        for i in range(1, n):
            if time.time() > deadline:
                return best_set
            old_val = U[i]
            for new_val in range(max(0, old_val - 2), min(max(U) + 2, old_val + 3)):
                if new_val == old_val:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_set
