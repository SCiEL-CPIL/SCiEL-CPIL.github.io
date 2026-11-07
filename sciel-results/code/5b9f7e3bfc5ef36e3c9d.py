"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import numpy as np

NMAX = 256

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of ints.
    Condition: For all distinct a,b,c in S U {x}, a does not divide b*c.
    """
    n = len(S)
    # Case 1: x divides b*c
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                return False
            # Case 2: b divides x*c
            if (x * c) % b == 0:
                return False
            # Case 3: c divides x*b
            if (x * b) % c == 0:
                return False
    return True

def _greedy(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _build_conflict_matrix():
    """
    Build a weighted conflict matrix where entry (i,j) represents how many
    violating triples involve both i and j. This helps identify which numbers
    are 'structurally isolated' (low conflict) vs. highly connected.
    """
    n = NMAX - 1  # numbers 2..256
    conflict = np.zeros((n, n), dtype=np.float64)
    
    # For each triple (a,b,c) with a|b*c, add weight to pairs (a,b), (a,c), (b,c)
    # But we need to be careful: the constraint is on distinct a,b,c
    # We'll count for each pair how many third elements create a violation
    
    # Precompute: for each a, which pairs (b,c) have a|b*c
    # Then for each such triple, increment conflict[b][c], conflict[a][b], conflict[a][c]
    
    # This is O(n^3) which is 255^3 ~ 16.5M, feasible with numpy
    nums = list(range(2, NMAX + 1))
    idx = {x: i - 2 for i, x in enumerate(nums)}  # x -> index in 0..254
    
    for i, a in enumerate(nums):
        for j in range(i + 1, len(nums)):
            b = nums[j]
            for k in range(j + 1, len(nums)):
                c = nums[k]
                if (b * c) % a == 0:
                    # Triple (a,b,c) violates
                    # Add to conflict matrix for all three pairs
                    conflict[i, j] += 1
                    conflict[j, i] += 1
                    conflict[i, k] += 1
                    conflict[k, i] += 1
                    conflict[j, k] += 1
                    conflict[k, j] += 1
    
    return conflict

def _spectral_order(conflict_matrix, deadline, seed=42):
    """
    Use eigenvalue centrality to order numbers by how 'isolated' they are.
    Lower eigenvector centrality = fewer conflicts = better to include early.
    """
    if time.time() > deadline:
        return list(range(2, NMAX + 1))
    
    # Compute eigenvector centrality
    # For sparse-ish matrices, use power iteration
    n = conflict_matrix.shape[0]
    
    # Normalize
    row_sums = np.sum(conflict_matrix, axis=1)
    row_sums[row_sums == 0] = 1
    normalized = conflict_matrix / row_sums[:, np.newaxis]
    
    # Power iteration for principal eigenvector
    rng = np.random.RandomState(seed)
    v = np.ones(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        v_new = normalized @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            break
        v = v_new
    
    # Lower centrality = more isolated = better to include first
    # Sort by ascending centrality
    order_indices = np.argsort(v)
    nums = list(range(2, NMAX + 1))
    return [nums[i] for i in order_indices]

def _local_search_improve(S, deadline, max_iters=500):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(42)
    
    # Pre-compute candidates not in S
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        progress = False
        # 1. Try to add any candidate
        if candidates:
            trial_candidates = rng.sample(candidates, min(10, len(candidates)))
            for x in trial_candidates:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break
            
        if progress:
            continue
            
        # 2. Try to swap
        if not S:
            break
            
        idx_remove = rng.randrange(len(S))
        removed = S[idx_remove]
        
        S_temp = S[:]
        S_temp.pop(idx_remove)
        S_temp_set = S_set - {removed}
        
        if candidates:
            trial_candidates = rng.sample(candidates, min(20, len(candidates)))
            swapped = False
            for x in trial_candidates:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            
            if not swapped:
                pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin

    best_S = None
    best_size = 0
    
    # Phase 0: Build conflict matrix (one-time cost)
    conflict_matrix = _build_conflict_matrix()
    
    # Phase 1: Diverse Greedy Initializations
    
    # 1. Spectral order (lowest centrality first)
    if time.time() < deadline - 30:
        spectral_order = _spectral_order(conflict_matrix, deadline - 25, seed=42)
        S_spec = _greedy(spectral_order, deadline)
        if S_spec and len(S_spec) > best_size:
            best_S = S_spec
            best_size = len(S_spec)
    
    # 2. Largest First
    if time.time() < deadline - 25:
        order_largest = list(range(NMAX, 1, -1))
        S1 = _greedy(order_largest, deadline)
        if S1 and len(S1) > best_size:
            best_S = S1
            best_size = len(S1)
        
    # 3. Smallest First
    if time.time() < deadline - 20:
        order_smallest = list(range(2, NMAX + 1))
        S2 = _greedy(order_smallest, deadline)
        if S2 and len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
        
    # 4. Randomized Orders
    for seed in [1, 7, 42, 100, 200]:
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy(order_rand, deadline)
        if Sr and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)
            
    # 5. Spectral with different seeds
    for seed in [1, 7, 100]:
        if time.time() > deadline - 10:
            break
        spectral_order = _spectral_order(conflict_matrix, deadline - 5, seed=seed)
        S_spec = _greedy(spectral_order, deadline)
        if S_spec and len(S_spec) > best_size:
            best_S = S_spec
            best_size = len(S_spec)
            
    if not best_S:
        best_S = [2, 3] 
        best_size = 2

    # Phase 2: Local Search Improvement
    if time.time() < deadline - 10:
        S_improved = _local_search_improve(best_S, deadline - 10, max_iters=2000)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    if time.time() < deadline - 5:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
                
        S_improved2 = _local_search_improve(S_perturbed, deadline - 5, max_iters=1000)
        if S_improved2 and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)
            
    if time.time() < deadline - 2:
        rng_perturb2 = random.Random(456)
        S_perturbed2 = best_S[:]
        for _ in range(rng_perturb2.randint(1, 2)):
            if S_perturbed2:
                idx = rng_perturb2.randrange(len(S_perturbed2))
                S_perturbed2.pop(idx)
                
        S_improved3 = _local_search_improve(S_perturbed2, deadline - 2, max_iters=500)
        if S_improved3 and len(S_improved3) > best_size:
            best_S = S_improved3
            best_size = len(S_improved3)

    return [int(x) for x in best_S]
