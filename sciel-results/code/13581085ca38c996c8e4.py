import time
from math import log, floor
import random

def score_set(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {0, floor(b^k) for k=1..m}"""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def modular_greedy_construct(p, n_elements, seed):
    """
    Greedy construction restricted to multiples of prime p.
    U = {k * p | k in V} where V is chosen greedily.
    This eliminates low-order sum collisions since all elements are ≡ 0 mod p.
    """
    random.seed(seed)
    
    # Start with V = {0}
    V = [0]
    best_score = score_set([0, p])
    V = [0, 1]
    
    for _ in range(n_elements - 2):
        # Generate candidate multipliers
        # Use a mix: some near geometric progression, some random
        max_existing = max(V) if V else 0
        
        candidates = set()
        # Geometric-ish expansion
        for base in [1.5, 2, 2.5, 3]:
            for k in range(1, 8):
                val = floor(max_existing * (base ** k)) if max_existing > 0 else k
                if val > 0:
                    candidates.add(val)
        
        # Random candidates
        for _ in range(50):
            val = random.randint(1, max_existing * 3 + 10)
            if val > 0:
                candidates.add(val)
        
        # Small integers
        for val in range(1, min(50, max_existing + 1)):
            candidates.add(val)
        
        candidates = [c for c in candidates if c not in V]
        
        best_candidate = None
        best_local_score = best_score
        
        # Try a subset of candidates (to save time)
        random.shuffle(candidates)
        for c in candidates[:30]:
            new_V = V + [c]
            new_V.sort()
            # Convert to actual set U
            U = [v * p for v in new_V]
            s = score_set(U)
            if s > best_local_score:
                best_local_score = s
                best_candidate = c
        
        if best_candidate is not None:
            V.append(best_candidate)
            V.sort()
            best_score = best_local_score
        else:
            # No improvement, try a random small addition
            if random.random() < 0.5:
                V.append(1 if 1 not in V else 2)
                V.sort()
            break
    
    # Final polish: convert V back to U and run local search
    U = [v * p for v in V]
    return U, best_score

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    def generate_local_candidates(current_U, limit):
        U_set = set(current_U)
        candidates = set()
        max_val = max(current_U)
        for a in current_U:
            for b in current_U:
                s = a + b
                d = abs(a - b)
                if 0 < s <= limit and s not in U_set:
                    candidates.add(s)
                if 0 < d <= limit and d not in U_set:
                    candidates.add(d)
        # Add geometric neighbors
        if max_val > 0:
            for mult in [1.2, 1.5, 2, 2.5, 3]:
                val = floor(max_val * mult)
                if 0 < val <= limit and val not in U_set:
                    candidates.add(val)
        # Add small values
        for val in range(1, min(30, limit + 1)):
            if val not in U_set:
                candidates.add(val)
        return list(candidates)
    
    candidates = generate_local_candidates(U, max_val_limit)
    
    while time.time() < deadline - 3:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 2.5))
            candidates = generate_local_candidates(U, new_limit)
        
        if move_type < 0.55:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_local_candidates(U, max(max_val_limit, int(max(U) * 2.5)))
                        break
        elif move_type < 0.75:
            # Try to REMOVE an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_local_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_local_candidates(U, max(max_val_limit, int(max(U) * 2.5)))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    random.seed(42)
    
    best_set = [0, 1]
    best_score = score_set(best_set)
    
    # 1. Quick structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 2, 5, 10, 20, 40, 80],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Geometric grid search
    b_candidates = [b_int / 10.0 for b_int in range(13, 60)]
    for b in b_candidates:
        if time.time() > deadline - 60:
            break
        for m in range(3, 18):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 3. Modular greedy constructions with various primes
    primes_to_try = [7, 11, 13, 17, 19, 23, 31]
    for p in primes_to_try:
        if time.time() > deadline - 50:
            break
        for n_elem in range(6, 22):
            if time.time() > deadline - 45:
                break
            for seed in [1, 7, 42]:
                if time.time() > deadline - 43:
                    break
                U, s = modular_greedy_construct(p, n_elem, seed)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
                # Quick local polish
                if s > best_score - 0.05:  # If competitive, polish
                    if time.time() > deadline - 20:
                        break
                    polished_score, polished_set = local_search_dynamic(U, deadline, int(max(U) * 3 + 100), 35)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_set = polished_set
    
    # 4. Dynamic local search from multiple seeds
    seeds = [best_set]
    for cand in structured_candidates[:2]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 800
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 25:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 6:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # 5. Final fine-grained polish
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 2 and iteration < 800:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old_val = current[i]
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score + 1e-12:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and n < 25 and time.time() < deadline - 2:
                max_val = max(current)
                for new_val in range(max_val - 15, max_val + 20):
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score + 1e-12:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            if not improved and n > 3 and time.time() < deadline - 2:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score + 1e-12:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    # Ensure valid output
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
