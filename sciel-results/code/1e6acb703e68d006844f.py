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
    
    U_arr = np.array(U)
    
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

def generate_candidates(U, max_val, n_cands=100):
    """Generate candidate values for adding to set U."""
    candidates = set()
    max_u = max(U) if U else 0
    
    # Random candidates
    for _ in range(n_cands):
        c = random.randint(1, max_val)
        candidates.add(c)
    
    # Candidates near existing elements
    for u in U:
        if u == 0:
            continue
        for delta in [-2, -1, 1, 2, 3, 5, 8, 13]:
            c = u + delta
            if 1 <= c <= max_val and c not in U:
                candidates.add(c)
    
    # Candidates that are sums/differences of existing elements
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            s = U[i] + U[j]
            d = abs(U[i] - U[j])
            if 1 <= s <= max_val and s not in U:
                candidates.add(s)
            if 1 <= d <= max_val and d not in U:
                candidates.add(d)
    
    return list(candidates)

def local_search_dynamic(U, deadline, max_iter=20000):
    """
    Dynamic local search with adaptive step sizes and simulated annealing.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score_local = current_score
    
    # Adaptive parameters
    temp = 0.5
    cooling = 0.9995
    step_scale = 1.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n < 2:
            break
        
        max_val = max(current)
        
        # Adaptive step size
        step_size = max(1, int(max_val * 0.1 * step_scale))
        
        # Generate multiple candidates per iteration
        num_candidates = 3 + random.randint(0, 2)
        candidates = []
        
        for _ in range(num_candidates):
            i = random.randint(1, n - 1)  # Don't remove 0
            old_val = current[i]
            
            if random.random() < 0.6:
                # Local perturbation
                delta = random.randint(-step_size, step_size)
                new_val = old_val + delta
            else:
                # Larger jump
                new_val = random.randint(1, max_val * 3)
            
            if new_val <= 0 or new_val in current:
                continue
            
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != n:
                continue
            
            candidates.append((new_set, new_val, i))
        
        if not candidates:
            continue
        
        # Evaluate candidates
        best_candidate = None
        best_candidate_score = -1
        
        for new_set, new_val, i in candidates:
            s = fast_score(new_set)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_set
        
        if best_candidate is None:
            continue
        
        # Accept or reject
        if best_candidate_score > current_score:
            current = best_candidate
            current_score = best_candidate_score
            step_scale = min(2.0, step_scale * 1.1)  # Expand search
            if current_score > best_score_local:
                best_score_local = current_score
                best = current[:]
        else:
            # Metropolis acceptance
            delta_s = current_score - best_candidate_score
            if delta_s < temp:
                accept_prob = 1.0 / (1.0 + delta_s / max(0.01, temp))
                if random.random() < accept_prob:
                    current = best_candidate
                    current_score = best_candidate_score
                    step_scale = max(0.5, step_scale * 0.9)  # Contract search
        
        temp *= cooling
    
    return best, best_score_local

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s
    
    random.seed(12345)
    
    # Multiple seed strategies
    seeds = []
    
    # Basic seeds
    seeds.append([0, 1, 4, 9, 11])
    seeds.append([0, 1, 2, 4, 8, 16])
    seeds.append([0, 1, 3, 7, 15, 31])
    
    # Geometric seeds with various parameters
    for base in [1, 2, 3, 5]:
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            for n in [5, 6, 7, 8]:
                G = geometric_set(base, ratio, n)
                if 0 in G and len(set(G)) == len(G):
                    seeds.append(G)
    
    # Random seeds
    for _ in range(20):
        n = random.randint(4, 8)
        max_v = random.randint(50, 300)
        R = [0] + sorted(random.sample(range(1, max_v), n - 1))
        seeds.append(R)
    
    best_set = seeds[0]
    best_score = fast_score(best_set)
    
    # Evaluate all seeds and keep the best
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = fast_score(seed)
        if s > best_score:
            best_score = s
            best_set = seed
    
    # Local search from best seed
    if time.time() < deadline - 15:
        refined, refined_score = local_search_dynamic(best_set, deadline - 10, max_iter=15000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Local search from other promising seeds
    for seed in seeds[:10]:
        if time.time() > deadline - 10:
            break
        if seed == best_set:
            continue
        s = fast_score(seed)
        if s > best_score * 0.95:  # Only refine seeds that are close to best
            refined, refined_score = local_search_dynamic(seed, deadline - 5, max_iter=5000)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    # Final polish: more iterations on the best set
    if time.time() < deadline - 3:
        refined, refined_score = local_search_dynamic(best_set, deadline - 1, max_iter=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
