"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) on C6.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def compute_score(U):
        U = sorted(set(U))
        n = len(U)
        if n == 0 or max(U) == 0:
            return 0.0
        # Compute sumset U+U
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set U-U
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        size_diff = len(diffset)
        size_sum = len(sumset)
        max_u = max(U)
        if max_u == 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        ratio = size_diff / size_sum
        if ratio <= 0:
            return 0.0
        score = 1 + math.log(ratio) / denom
        return score
    
    # Baseline score
    baseline = [0, 1, 3]
    best_score = compute_score(baseline)
    best_set = list(baseline)
    
    random.seed(42)
    
    # Exhaustive search over small sets: subsets of {0, 1, ..., M} of size 3..8
    # This is key: small sets with high asymmetry ratio
    for M in range(4, 16):
        if time.time() > deadline - 30.0:
            break
        universe = list(range(0, M + 1))
        # Try subsets of size 3 to 8
        for size in range(3, min(9, M + 1)):
            if time.time() > deadline - 30.0:
                break
            for subset in combinations(universe, size):
                if 0 not in subset:
                    continue
                s = list(subset)
                score = compute_score(s)
                if score > best_score:
                    best_score = score
                    best_set = list(s)
    
    # Try structured sets: arithmetic progressions
    for n in range(3, 25):
        for d in range(1, 30):
            if time.time() > deadline - 25.0:
                break
            s = [i * d for i in range(n)]
            score = compute_score(s)
            if score > best_score:
                best_score = score
                best_set = list(s)
    
    # Try geometric-like sets
    for base in range(2, 15):
        for n in range(3, 20):
            if time.time() > deadline - 25.0:
                break
            s = [0]
            val = 1
            for i in range(1, n):
                val *= base
                s.append(val)
            score = compute_score(s)
            if score > best_score:
                best_score = score
                best_set = list(s)
    
    # Try mixed constructions: combine small dense sets with sparse extensions
    # e.g., {0, 1, 2, ..., k} union {k*m, (k+1)*m, ...}
    for k in range(2, 10):
        for m in range(k + 1, k * 5 + 1):
            if time.time() > deadline - 25.0:
                break
            s = set(range(0, k + 1))
            for j in range(1, 6):
                s.add(j * m)
            s = sorted(s)
            score = compute_score(s)
            if score > best_score:
                best_score = score
                best_set = list(s)
    
    # Random search with various parameters
    for trial in range(8000):
        if time.time() > deadline - 20.0:
            break
        n = random.randint(3, 18)
        max_val = random.randint(5, 60)
        s = {0}
        for _ in range(n - 1):
            s.add(random.randint(1, max_val))
        s = sorted(s)
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Local search: try to modify the best set
    def local_search(s, max_iter=200):
        s = sorted(set(s))
        current_score = compute_score(s)
        no_improve = 0
        for _ in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            improved = False
            # Try adding a random element
            if len(s) < 35:
                max_val = max(s) + random.randint(1, 15)
                new_s = s + [random.randint(1, max_val)]
                new_score = compute_score(new_s)
                if new_score > current_score + 1e-12:
                    s = sorted(set(new_s))
                    current_score = new_score
                    improved = True
                    no_improve = 0
            
            # Try removing a random element (but keep 0)
            if len(s) > 3:
                idx = random.choice(range(1, len(s)))
                new_s = s[:idx] + s[idx+1:]
                new_score = compute_score(new_s)
                if new_score > current_score + 1e-12:
                    s = new_s
                    current_score = new_score
                    improved = True
                    no_improve = 0
            
            # Try replacing a random element
            if len(s) > 3:
                idx = random.choice(range(1, len(s)))
                old_val = s[idx]
                new_val = random.randint(1, max(s) + 8)
                if new_val not in s:
                    new_s = s[:idx] + [new_val] + s[idx+1:]
                    new_score = compute_score(new_s)
                    if new_score > current_score + 1e-12:
                        s = sorted(set(new_s))
                        current_score = new_score
                        improved = True
                        no_improve = 0
            
            # Try perturbing a random element
            if len(s) > 3:
                idx = random.choice(range(1, len(s)))
                old_val = s[idx]
                new_val = old_val + random.randint(-4, 4)
                if new_val >= 0 and new_val not in s:
                    new_s = s[:idx] + [new_val] + s[idx+1:]
                    new_s = sorted(new_s)
                    new_score = compute_score(new_s)
                    if new_score > current_score + 1e-12:
                        s = new_s
                        current_score = new_score
                        improved = True
                        no_improve = 0
            
            if not improved:
                no_improve += 1
                if no_improve > 30:
                    # Restart with perturbation
                    idx = random.choice(range(1, len(s)))
                    s[idx] = max(1, s[idx] + random.randint(-5, 5))
                    s = sorted(set(x for x in s if x >= 0))
                    if 0 not in s:
                        s.insert(0, 0)
                    no_improve = 0
        
        return s, current_score
    
    # Run local search from the best set found
    if time.time() < deadline - 20.0:
        improved_set, improved_score = local_search(best_set, max_iter=300)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Run local search from a few other good candidates
    # Re-run exhaustive to get top candidates
    candidates = []
    for M in range(4, 14):
        if time.time() > deadline - 15.0:
            break
        universe = list(range(0, M + 1))
        for size in range(3, min(8, M + 1)):
            if time.time() > deadline - 15.0:
                break
            for subset in combinations(universe, size):
                if 0 not in subset:
                    continue
                s = list(subset)
                score = compute_score(s)
                candidates.append((score, s))
    
    candidates.sort(key=lambda x: -x[0])
    for score, s in candidates[:5]:
        if time.time() > deadline - 10.0:
            break
        if score <= best_score:
            break
        improved_set, improved_score = local_search(s, max_iter=200)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
