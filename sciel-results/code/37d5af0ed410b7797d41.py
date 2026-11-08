import time
from math import log
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U. Fast version using NumPy."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return -1.0
    
    # Compute sumset and diffset using NumPy for speed
    arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums (i <= j)
    sums_arr = arr[:, None] + arr[None, :]
    # Only take upper triangle including diagonal
    mask = np.triu(np.ones((n, n), dtype=bool))
    sums_unique = np.unique(sums_arr[mask])
    
    # Diffset: all pairwise differences (i < j), plus 0
    diffs_arr = arr[None, :] - arr[:, None]
    # Take strict upper triangle
    mask_strict = np.triu(np.ones((n, n), dtype=bool), k=1)
    diffs_unique = np.unique(np.concatenate([diffs_arr[mask_strict].ravel(), [0]]))
    
    sc = len(sums_unique)
    dc = len(diffs_unique)
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1.0:
        return -1.0
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 150s
    
    best_set = [0, 1, 4, 9, 16, 25]
    best_score = score_set(best_set)
    
    # Generate many candidate structures
    candidates = []
    
    # Geometric-like sets with various bases
    for b in range(2, 40):
        for m in range(3, 15):
            U = [0]
            val = 1
            for k in range(1, m):
                val *= b
                U.append(val)
            if max(U) < 200000:
                candidates.append(U)
    
    # Squares
    for m in range(3, 25):
        U = [0] + [i*i for i in range(1, m+1)]
        if max(U) < 200000:
            candidates.append(U)
    
    # Triangular numbers
    for m in range(3, 20):
        U = [0] + [i*(i+1)//2 for i in range(1, m+1)]
        if max(U) < 200000:
            candidates.append(U)
    
    # Fibonacci-like
    for m in range(3, 25):
        U = [0, 1]
        a, b = 1, 1
        for _ in range(m-2):
            a, b = b, a+b
            U.append(b)
        if max(U) < 200000:
            candidates.append(U)
    
    # Mixed: small dense + sparse tail
    for small_n in range(3, 10):
        for tail_base in [3, 4, 5, 7, 10]:
            U = list(range(small_n))
            val = small_n
            for _ in range(3, 10):
                val *= tail_base
                U.append(val)
            if max(U) < 200000:
                candidates.append(sorted(set(U)))
    
    # Powers of 2 with offsets
    for offset in [0, 1, 2, 3]:
        for m in range(4, 15):
            U = [offset]
            val = 1
            for k in range(1, m):
                val *= 2
                U.append(offset + val)
            if 0 in U and max(U) < 200000:
                candidates.append(sorted(set(U)))
    
    # Random geometric with perturbation
    random.seed(42)
    for _ in range(50):
        b = random.uniform(2.5, 8.0)
        m = random.randint(5, 12)
        U = [0]
        val = 1
        for k in range(1, m):
            val *= b
            U.append(int(val) + random.randint(-1, 1))
        U = sorted(set(U))
        if 0 in U and len(U) >= 3 and max(U) < 200000:
            candidates.append(U)
    
    # Evaluate all candidates
    for U in candidates:
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Local search from best
    def local_search(U, deadline):
        U = sorted(set(U))
        cur_score = score_set(U)
        best_local = list(U)
        best_local_score = cur_score
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            improved = False
            n = len(U)
            max_val = max(U)
            
            # Try adding elements
            if n < 30:
                # Try values near existing ones and some random
                trial_vals = set()
                for x in U:
                    for d in range(1, 8):
                        trial_vals.add(x + d)
                        if x - d > 0:
                            trial_vals.add(x - d)
                # Add some random values in different ranges
                for _ in range(30):
                    trial_vals.add(random.randint(1, max_val + 100))
                # Try values that are gaps in the set
                for i in range(len(U) - 1):
                    gap_mid = (U[i] + U[i+1]) // 2
                    if gap_mid not in U and gap_mid > 0:
                        trial_vals.add(gap_mid)
                
                for v in sorted(trial_vals):
                    if v in U or v <= 0:
                        continue
                    new_U = sorted(U + [v])
                    new_s = score_set(new_U)
                    if new_s > cur_score:
                        U = new_U
                        cur_score = new_s
                        improved = True
                        break
            
            # Try removing elements
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = [U[j] for j in range(n) if j != i]
                    new_s = score_set(new_U)
                    if new_s > cur_score:
                        U = new_U
                        cur_score = new_s
                        improved = True
                        break
            
            # Try replacing elements with larger deltas
            if not improved and n > 2:
                for i in range(1, n):
                    old = U[i]
                    for delta in range(-10, 11):
                        if delta == 0:
                            continue
                        new_val = old + delta
                        if new_val <= 0:
                            continue
                        if new_val in U:
                            continue
                        new_U = [U[j] if j != i else new_val for j in range(n)]
                        new_U = sorted(set(new_U))
                        if len(new_U) != n:
                            continue
                        new_s = score_set(new_U)
                        if new_s > cur_score:
                            U = new_U
                            cur_score = new_s
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                no_improve = 0
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_local = list(U)
            else:
                no_improve += 1
        
        return best_local_score, best_local
    
    # Run local search from multiple seeds
    seeds = [best_set]
    # Add some diverse seeds
    for U in candidates[:10]:
        if U != best_set:
            seeds.append(U)
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        ls_score, ls_set = local_search(list(seed), deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Final high-precision polish pass with tighter tolerances
    if time.time() < deadline - 2:
        U = list(best_set)
        cur_score = best_score
        # More iterations and wider search for final polish
        for iteration in range(500):
            if time.time() > deadline - 1:
                break
            improved = False
            n = len(U)
            # Try all single-element moves with wider delta range
            for i in range(1, n):
                old = U[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    nv = old + delta
                    if nv <= 0 or nv in U:
                        continue
                    new_U = [U[j] if j != i else nv for j in range(n)]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    ns = score_set(new_U)
                    if ns > cur_score:
                        U = new_U
                        cur_score = ns
                        improved = True
                        break
                if improved:
                    break
            
            # Also try adding one element
            if not improved and n < 25:
                trial_vals = set()
                for x in U:
                    for d in range(1, 4):
                        trial_vals.add(x + d)
                        if x - d > 0:
                            trial_vals.add(x - d)
                for v in sorted(trial_vals):
                    if v in U or v <= 0:
                        continue
                    new_U = sorted(U + [v])
                    ns = score_set(new_U)
                    if ns > cur_score:
                        U = new_U
                        cur_score = ns
                        improved = True
                        break
            
            if not improved:
                break
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = U
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
