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
        # Compute sumset
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set
        diffset = set()
        for a in U:
            for b in U:
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
    
    # Phase 1: Try structured candidates for various sizes and max values
    for max_u in range(1, 50):
        if time.time() > deadline - 5:
            break
        
        # Arithmetic progression
        for n in range(3, 12):
            if (max_u) % (n - 1) == 0:
                step = max_u // (n - 1)
                if step >= 1:
                    ap = [i * step for i in range(n)]
                    try_set(ap)
        
        # Powers of 2
        if max_u >= 8:
            geo = []
            v = 0
            while v <= max_u:
                geo.append(v)
                if len(geo) > 1:
                    v = geo[-1] * 2
                else:
                    v = 1
            if geo and max(geo) <= max_u:
                try_set(geo)
        
        # Fibonacci-like
        fib = [0, 1]
        while fib[-1] <= max_u:
            fib.append(fib[-1] + fib[-2])
        for n in range(3, len(fib)):
            if fib[n-1] <= max_u:
                try_set(fib[:n])
        
        # Blocks: {0,1, 2,3, 4,5, ...} pattern
        for block_size in [2, 3, 4]:
            blocks = []
            for start in range(0, max_u + 1, block_size):
                for i in range(block_size):
                    if start + i <= max_u:
                        blocks.append(start + i)
            if blocks and 0 in blocks:
                try_set(blocks)
        
        # Random structured sets
        if max_u <= 30:
            random.seed(42 + max_u)
            for n in range(3, min(10, max_u + 1)):
                for _ in range(500):
                    if time.time() > deadline - 5:
                        break
                    inner = list(range(1, max_u))
                    if len(inner) < n - 2:
                        continue
                    selected = random.sample(inner, n - 2)
                    cand = sorted([0] + selected + [max_u])
                    try_set(cand)
    
    # Phase 2: Local search around best set
    if time.time() < deadline - 15:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 15:
                max_cur = max(current)
                for new_el in range(1, max_cur + 5):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
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
                    for new_el in range(1, max(current) + 5):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element (if size > 3)
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
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
    
    # Phase 3: Try larger structured sets if time remains
    if time.time() < deadline - 20:
        # Try sets with specific patterns for larger max_u
        for max_u in range(50, 200, 10):
            if time.time() > deadline - 20:
                break
            
            # Geometric with ratio 3
            geo3 = [0, 1]
            while geo3[-1] * 3 <= max_u:
                geo3.append(geo3[-1] * 3)
            if max(geo3) <= max_u:
                try_set(geo3)
            
            # Mixed: small AP + large spread
            for n_ap in [3, 4, 5]:
                for step in [1, 2, 3]:
                    ap_part = [i * step for i in range(n_ap)]
                    remaining = max_u - ap_part[-1]
                    if remaining > 0:
                        # Add a few more spread out
                        extra = []
                        v = ap_part[-1] + step
                        while v <= max_u and len(extra) < 3:
                            extra.append(v)
                            v += remaining // 4 + 1
                        if extra:
                            cand = sorted(ap_part + extra + [max_u])
                            try_set(cand)
        
        # Random search with larger max_u
        random.seed(123)
        for _ in range(500):
            if time.time() > deadline - 20:
                break
            max_u = random.randint(50, 300)
            n = random.randint(4, 10)
            inner = list(range(1, max_u))
            if len(inner) < n - 2:
                continue
            selected = random.sample(inner, n - 2)
            cand = sorted([0] + selected + [max_u])
            try_set(cand)
    
    # Phase 4: Final polish - try to improve best_set with small perturbations
    if time.time() < deadline - 5:
        current = list(best_set)
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            if random.random() < 0.5:
                # Add
                if len(current) < 20:
                    max_cur = max(current)
                    new_el = random.randint(1, max_cur + 10)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
            else:
                # Replace
                if len(current) >= 3:
                    idx = random.randint(1, len(current) - 1)  # avoid 0
                    new_el = random.randint(1, max(current) + 10)
                    if new_el not in current:
                        cand = list(current)
                        cand[idx] = new_el
                        cand = sorted(set(cand))
                        if 0 in cand:
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
    
    return list(best_set)
# EVOLVE-BLOCK-END
