"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Use sets for sum and diff
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Exhaustive search for small sets
    # We extend the range slightly to 25 to ensure we don't miss a better small configuration
    max_u_limit_small = 25
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 10:
            break
        elements = list(range(1, max_u + 1))
        # Size 3 to 7 is reasonable for small max_u
        for size in range(3, 8):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # Phase 2: Structured sets with larger max_u
    # These are less likely to be optimal due to the log denominator, 
    # but we check a few patterns just in case.
    for max_u in range(30, 100, 5):
        if time.time() > deadline - 10:
            break
        
        # Two blocks
        for a in range(1, 5): # size of first block
            for b in range(a + 1, max_u - a + 1):
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # Geometric with small ratios
        for r in [2, 3]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
            
        # Arithmetic progressions
        for d in range(1, 5):
            ap = []
            v = 0
            while v <= max_u:
                ap.append(v)
                v += d
            try_set(ap)
    
    # Phase 3: Local search with multiple restarts
    # We perform local search starting from the current best set
    # and also from some random sets to escape local optima.
    
    def local_search(start_set, max_iter=100):
        current = sorted(set(start_set))
        if 0 not in current:
            current.insert(0, 0)
        current_score = compute_score(current)
        no_improve = 0
        max_no_improve = 20
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
                
            improved = False
            max_cur = max(current)
            
            # 1. Try adding an element
            if len(current) < 12:
                # Candidates: near existing elements, or extending the range
                candidates = set()
                for x in current:
                    for dx in [-2, -1, 1, 2]:
                        if x + dx > 0:
                            candidates.add(x + dx)
                # Extend range slightly
                candidates.update(range(max_cur + 1, max_cur + 5))
                
                for new_el in sorted(candidates):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score:
                        current = cand
                        current_score = s
                        improved = True
                        no_improve = 0
                        break
            
            # 2. Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    orig = current[i]
                    # Try replacing with nearby values
                    for new_el in range(max(1, orig - 3), orig + 4):
                        if new_el in current or new_el == 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > current_score:
                            current = cand
                            current_score = s
                            improved = True
                            no_improve = 0
                            break
                    if improved:
                        break
            
            # 3. Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > current_score:
                        current = cand
                        current_score = s
                        improved = True
                        no_improve = 0
                        break
            
            if not improved:
                no_improve += 1
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_set = current
        return current

    # Run local search from best set
    local_search(best_set)
    
    # Run local search from a few random sets if time permits
    if time.time() < deadline - 20:
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Generate a random set
            size = random.randint(3, 8)
            max_u = random.randint(5, 30)
            inner = random.sample(range(1, max_u + 1), min(size - 1, max_u))
            cand = sorted([0] + inner)
            local_search(cand, max_iter=50)
            
    return list(best_set)
# EVOLVE-BLOCK-END
