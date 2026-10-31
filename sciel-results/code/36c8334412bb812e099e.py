"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def compute_score_fast(U):
        """Compute score using vectorized NumPy operations for speed."""
        U = np.array(sorted(set(U)), dtype=np.int64)
        if len(U) == 0 or U[0] != 0:
            return -1
        if len(U) == 1:
            return 0.0
            
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        # Vectorized computation of sumset and diffset sizes
        # This is O(n^2) but very fast in C/NumPy
        sums = np.add.outer(U, U)
        n_sum = len(np.unique(sums))
        
        diffs = np.subtract.outer(U, U)
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0
    best_set = [0, 1, 3]
    best_score = compute_score_fast([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small sets
    max_u_limit_small = 20
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 5:
            break
        elements = list(range(1, max_u + 1))
        for size in range(3, 8):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)

    # Phase 2: Generate Golomb Rulers
    # A Golomb ruler has distinct differences. This maximizes |U-U| for a given size.
    # We try to construct them greedily.
    def generate_golomb_ruler(n):
        U = [0]
        diffs = set([0])
        current_max = 0
        for _ in range(n - 1):
            found = False
            x = current_max + 1
            while not found:
                ok = True
                new_diffs = []
                for u in U:
                    d1 = x - u
                    d2 = u - x
                    if d1 in diffs or d2 in diffs:
                        ok = False
                        break
                    new_diffs.append(d1)
                    new_diffs.append(d2)
                if ok:
                    U.append(x)
                    diffs.update(new_diffs)
                    current_max = x
                    found = True
                else:
                    x += 1
        return U

    # Try generating Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline - 10:
            break
        try:
            gr = generate_golomb_ruler(n)
            try_set(gr)
        except Exception:
            pass

    # Phase 3: Random Sparse Sets
    # Sets with large gaps often have good properties.
    for _ in range(500):
        if time.time() > deadline - 10:
            break
        n = random.randint(3, 15)
        max_u = random.randint(50, 500)
        # Generate random points, ensure 0 is included
        points = [0]
        for _ in range(n - 1):
            points.append(random.randint(1, max_u))
        points = sorted(set(points))
        try_set(points)

    # Phase 4: Local Search with Fast Evaluator
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 30:
                max_cur = max(current)
                candidates = set()
                # Candidates near existing elements
                for x in current:
                    for dx in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                        candidates.add(x + dx)
                # Candidates above max
                candidates.update(range(max_cur + 1, max_cur + 20))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current or new_el < 0:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    # Try moving element i
                    for new_el in range(max(1, current[i] - 20), current[i] + 21):
                        if new_el in current or new_el < 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score_fast(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
