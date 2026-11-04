import time
import random
from math import log, log2
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def polynomial_score(U):
    """
    Score using polynomial interpretation:
    P(x) = sum_{u in U} x^u
    Sumset = support of P(x)^2
    Difference set = support of P(x) * P(x^{-1})
    
    This is equivalent to fast_score but framed differently.
    """
    return fast_score(U)

def base_expansion_set(base, max_digits, seed_val=None):
    """
    Generate a set using base-expansion: u_k = sum_{j=0}^{k} b_j * base^j, b_j in {0,1}
    This structurally limits sumset collisions to local digit-wise interactions
    while difference sets expand with number of active digits.
    """
    if seed_val is None:
        seed_val = random.randint(1, max_digits * base)
    
    rng = random.Random(seed_val)
    digits = []
    for j in range(max_digits):
        if j == 0:
            digits.append(1)
        else:
            p = 0.4 + 0.2 * (j % 2)
            if rng.random() < p:
                digits.append(1)
            else:
                digits.append(0)
    
    U = [0]
    for j, b in enumerate(digits):
        if b == 1:
            val = base ** j
            if val not in U:
                U.append(val)
    
    active_digits = [j for j, b in enumerate(digits) if b == 1]
    for i in range(len(active_digits)):
        for j in range(i+1, len(active_digits)):
            val = base ** active_digits[i] + base ** active_digits[j]
            if val not in U and val < base ** max_digits:
                U.append(val)
    
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    return U

def geometric_set(base, ratio, n):
    """Generate a geometric-like set starting from base with given ratio."""
    if n <= 1:
        return [0]
    U = [0]
    val = base
    for _ in range(n - 1):
        U.append(val)
        val = int(val * ratio + 0.5)
    return U

def base_expansion_seeds():
    """Generate multiple base-expansion seeds with varying parameters."""
    seeds = []
    for base in [3, 4, 5, 7, 10]:
        for max_digits in [4, 5, 6, 7, 8]:
            for seed_val in range(5):
                G = base_expansion_set(base, max_digits, seed_val)
                if len(G) >= 3:
                    seeds.append(G)
    return seeds

def polynomial_greedy_construct(max_val, n_target, deadline):
    """
    Greedy construction guided by polynomial structure.
    At each step, choose the element that maximizes the increase in |U-U|/|U+U| ratio.
    
    The key insight: when adding a new element u to U, the change in sumset and difference set
    can be computed incrementally using the polynomial interpretation.
    """
    U = [0]
    current_score = 0.0
    
    # Pre-compute candidate values
    candidates = list(range(1, min(max_val + 1, 200)))
    
    while len(U) < n_target and time.time() < deadline:
        best_candidate = None
        best_score = current_score
        
        # Try each candidate
        for cand in candidates:
            if cand in U:
                continue
            
            new_U = U + [cand]
            s = fast_score(new_U)
            if s > best_score:
                best_score = s
                best_candidate = cand
        
        if best_candidate is None:
            break
        
        U.append(best_candidate)
        U.sort()
        current_score = best_score
        
        # Update candidates: remove used, add new possibilities
        candidates = [c for c in candidates if c not in U]
        # Add some new candidates near existing elements
        new_cands = set()
        for u in U:
            for delta in [1, 2, 3, 5, 8, 13, 21]:
                nc = u + delta
                if 1 <= nc <= max_val and nc not in U:
                    new_cands.add(nc)
        candidates.extend(list(new_cands))
        candidates = list(set(candidates))
    
    return U

def local_search_dynamic(U, deadline, max_iter=20000):
    """
    Dynamic local search with adaptive step sizes and simulated annealing.
    Supports replace, add, and remove operations.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score_local = current_score
    
    temp = 0.3
    cooling = 0.999
    step_scale = 1.0
    no_improve_count = 0
    max_no_improve = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n < 2:
            break
        
        max_val = max(current)
        
        step_size = max(1, int(max_val * 0.05 * step_scale))
        
        num_candidates = 6 + random.randint(0, 4)
        candidates = []
        
        current_set = set(current)
        
        for _ in range(num_candidates):
            op = random.random()
            
            if op < 0.6:
                i = random.randint(1, n - 1)
                old_val = current[i]
                
                r = random.random()
                if r < 0.5:
                    delta = random.randint(-step_size, step_size)
                    new_val = old_val + delta
                elif r < 0.8:
                    new_val = random.randint(1, max_val)
                else:
                    new_val = random.randint(1, max_val * 2)
                
                if new_val <= 0 or new_val in current_set:
                    continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != n:
                    continue
                
                candidates.append(new_set)
                
            elif op < 0.85:
                if n >= 12:
                    continue
                r = random.random()
                if r < 0.5:
                    new_val = random.randint(1, max_val * 2)
                else:
                    i = random.randint(0, n - 1)
                    new_val = current[i] + random.randint(1, step_size)
                
                if new_val <= 0 or new_val in current_set:
                    continue
                
                new_set = current[:] + [new_val]
                new_set = sorted(set(new_set))
                
                if len(new_set) != n + 1:
                    continue
                
                candidates.append(new_set)
                
            else:
                if n <= 3:
                    continue
                i = random.randint(1, n - 1)
                new_set = current[:i] + current[i+1:]
                candidates.append(new_set)
        
        if not candidates:
            continue
        
        best_candidate = None
        best_candidate_score = -1
        
        for new_set in candidates:
            s = fast_score(new_set)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_set
        
        if best_candidate is None:
            continue
        
        if best_candidate_score > current_score:
            current = best_candidate
            current_score = best_candidate_score
            step_scale = min(1.5, step_scale * 1.05)
            no_improve_count = 0
            if current_score > best_score_local:
                best_score_local = current_score
                best = current[:]
        else:
            delta_s = current_score - best_candidate_score
            if delta_s < temp:
                accept_prob = 1.0 / (1.0 + delta_s / max(0.01, temp))
                if random.random() < accept_prob:
                    current = best_candidate
                    current_score = best_candidate_score
                    step_scale = max(0.3, step_scale * 0.95)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                step_scale = min(2.0, step_scale * 1.5)
                no_improve_count = 0
        
        temp *= cooling
    
    return best, best_score_local

def final_polish(U, deadline, max_iter=5000):
    """
    High-precision polish pass with very small steps.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score_local = current_score
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n < 2:
            break
        
        max_val = max(current)
        current_set = set(current)
        improved = False
        
        for i in range(1, n):
            if time.time() > deadline:
                break
            
            old_val = current[i]
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_set:
                    continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != n:
                    continue
                
                s = fast_score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    current_set = set(current)
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best = current[:]
                    improved = True
                    break
        
        if not improved and n < 12:
            if time.time() > deadline:
                break
            for i in range(n):
                if time.time() > deadline:
                    break
                for delta in [1, 2, 3, 5, 8, 13, 21]:
                    new_val = current[i] + delta
                    if new_val <= 0 or new_val in current_set:
                        continue
                    new_set = current[:] + [new_val]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n + 1:
                        continue
                    s = fast_score(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        current_set = set(current)
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best = current[:]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved and n > 3:
            if time.time() > deadline:
                break
            for i in range(1, n):
                if time.time() > deadline:
                    break
                new_set = current[:i] + current[i+1:]
                s = fast_score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    current_set = set(current)
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best = current[:]
                    improved = True
                    break
    
    return best, best_score_local

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    random.seed(12345)
    
    seeds = []
    
    # Basic seeds
    seeds.append([0, 1, 4, 9, 11])
    seeds.append([0, 1, 2, 4, 8, 16])
    seeds.append([0, 1, 3, 7, 15, 31])
    
    # Geometric seeds
    for base in [1, 2, 3, 5]:
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            for n in [5, 6, 7, 8, 9]:
                G = geometric_set(base, ratio, n)
                if 0 in G and len(set(G)) == len(G):
                    seeds.append(G)
    
    # Base-expansion seeds
    be_seeds = base_expansion_seeds()
    seeds.extend(be_seeds)
    
    # Random seeds
    for _ in range(30):
        n = random.randint(4, 10)
        max_v = random.randint(50, 500)
        R = [0] + sorted(random.sample(range(1, max_v), n - 1))
        seeds.append(R)
    
    # Fibonacci-like seeds
    fibs = [0, 1]
    for _ in range(10):
        fibs.append(fibs[-1] + fibs[-2])
    for start in range(min(5, len(fibs) - 2)):
        for length in range(4, min(8, len(fibs) - start)):
            seeds.append(fibs[start:start+length])
    
    # Polynomial greedy construction seeds
    if time.time() < deadline - 30:
        pg1 = polynomial_greedy_construct(100, 8, deadline - 25)
        if len(pg1) >= 3:
            seeds.append(pg1)
        pg2 = polynomial_greedy_construct(200, 10, deadline - 20)
        if len(pg2) >= 3:
            seeds.append(pg2)
    
    best_set = seeds[0]
    best_score = fast_score(best_set)
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = fast_score(seed)
        if s > best_score:
            best_score = s
            best_set = seed
    
    scored_seeds = []
    for seed in seeds:
        s = fast_score(seed)
        scored_seeds.append((s, seed))
    scored_seeds.sort(key=lambda x: -x[0])
    
    # Local search from best seed
    if time.time() < deadline - 20:
        refined, refined_score = local_search_dynamic(best_set, deadline - 15, max_iter=15000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Local search from other promising seeds
    for score, seed in scored_seeds[:8]:
        if time.time() > deadline - 10:
            break
        if seed == best_set:
            continue
        if score > best_score * 0.9:
            refined, refined_score = local_search_dynamic(seed, deadline - 5, max_iter=5000)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    # Final polish
    if time.time() < deadline - 3:
        refined, refined_score = final_polish(best_set, deadline - 1, max_iter=5000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # One more local search with remaining time
    if time.time() < deadline - 2:
        remaining_iter = max(1000, int((deadline - time.time()) * 500))
        refined, refined_score = local_search_dynamic(best_set, deadline - 0.5, max_iter=remaining_iter)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Final polish again
    if time.time() < deadline - 0.5:
        refined, refined_score = final_polish(best_set, deadline - 0.1, max_iter=2000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
