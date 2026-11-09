import time
import random
import math
import numpy as np

N_VERT = 12
M = 40
TIME_LIMIT = 75  # seconds, leave 15s margin for 90s wall-clock limit

def _count_pairs(masks):
    """Count comparable pairs in a list of integer masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _count_pairs_np(masks_np):
    """Count comparable pairs using NumPy vectorization."""
    n = len(masks_np)
    a = masks_np[:, np.newaxis]
    b = masks_np[np.newaxis, :]
    ab = a & b
    comparable = (ab == a) | (ab == b)
    # Exclude diagonal
    np.fill_diagonal(comparable, False)
    return int(np.sum(comparable) // 2)

def _best_replacement_np(masks_np, i, all_masks_np, current_set):
    """Find the best replacement for position i using vectorized evaluation."""
    old = masks_np[i]
    others = np.delete(masks_np, i)
    
    # Precompute old contributions
    ab_old = others & old
    comp_old = (ab_old == others) | (ab_old == old)
    old_contrib = int(np.sum(comp_old))
    
    best_delta = 0
    best_new = None
    
    # Filter out masks already in current set
    available = []
    for mask in all_masks_np:
        if mask not in current_set:
            available.append(mask)
    
    if not available:
        return 0, None
    
    available_np = np.array(available, dtype=np.int64)
    
    # Vectorize: for each candidate, compute comparable count with others
    # Shape: (len(available), len(others))
    ab = available_np[:, np.newaxis] & others[np.newaxis, :]
    comp = (ab == available_np[:, np.newaxis]) | (ab == others[np.newaxis, :])
    new_contribs = comp.sum(axis=1)
    
    deltas = new_contribs - old_contrib
    max_idx = np.argmax(deltas)
    max_delta = int(deltas[max_idx])
    
    if max_delta > 0:
        return max_delta, int(available_np[max_idx])
    return 0, None

def _greedy_init(seed=42):
    """Greedy construction: build a dense comparable structure by adding masks that maximize marginal gain."""
    all_masks = list(range(1 << N_VERT))
    random.seed(seed)
    
    # Start with a few well-chosen seeds to create a base structure
    # Use masks of varying sizes to create comparability
    selected = []
    
    # Add empty set and full set as anchors
    selected.append(0)
    selected.append((1 << N_VERT) - 1)
    
    # Greedily add masks that maximize comparable pairs with existing set
    for _ in range(M - 2):
        best_mask = None
        best_gain = -1
        
        # Sample a subset of candidates for speed (not all 4096)
        candidates = random.sample(all_masks, min(500, len(all_masks)))
        
        for mask in candidates:
            if mask in selected:
                continue
            # Count how many existing sets are comparable with this mask
            gain = 0
            for existing in selected:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
    
    return selected

def _simulated_annealing(masks, deadline, start_temp=10.0, cool_rate=0.95):
    """Simulated Annealing with 1-opt and 2-opt moves."""
    current = list(masks)
    current_np = np.array(current, dtype=np.int64)
    current_score = _count_pairs_np(current_np)
    best = list(current)
    best_score = current_score
    
    all_masks_np = np.arange(1 << N_VERT, dtype=np.int64)
    current_set = set(current)
    
    temp = start_temp
    
    while time.time() < deadline and temp > 0.01:
        temp *= cool_rate
        
        # 1-opt move
        i = random.randint(0, M - 1)
        old_i = current[i]
        # Pick a random candidate
        available = [m for m in all_masks_np if m not in current_set]
        if not available:
            continue
        new_i = random.choice(available)
        
        # Calculate delta for 1-opt
        others = np.delete(current_np, i)
        ab_old = others & old_i
        comp_old = (ab_old == others) | (ab_old == old_i)
        old_contrib = int(np.sum(comp_old))
        
        ab_new = others & new_i
        comp_new = (ab_new == others) | (ab_new == new_i)
        new_contrib = int(np.sum(comp_new))
        
        delta = new_contrib - old_contrib
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            current[i] = new_i
            current_np[i] = new_i
            current_set.remove(old_i)
            current_set.add(new_i)
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        # 2-opt move (replace two elements)
        if random.random() < 0.5: # Probability of attempting 2-opt
            i = random.randint(0, M - 1)
            j = random.randint(0, M - 1)
            if i == j:
                continue
            
            old_i = current[i]
            old_j = current[j]
            
            # Pick two random candidates
            available = [m for m in all_masks_np if m not in current_set]
            if len(available) < 2:
                continue
            
            new_i, new_j = random.sample(available, 2)
            
            # Calculate delta for 2-opt
            # Remove i and j from consideration for "others"
            others_idx = [k for k in range(M) if k != i and k != j]
            others = current_np[others_idx]
            
            # Old contributions of i and j with others and each other
            ab_i_old = others & old_i
            comp_i_old = (ab_i_old == others) | (ab_i_old == old_i)
            contrib_i_old = int(np.sum(comp_i_old))
            
            ab_j_old = others & old_j
            comp_j_old = (ab_j_old == others) | (ab_j_old == old_j)
            contrib_j_old = int(np.sum(comp_j_old))
            
            # Comparable between i and j
            if (old_i & old_j) == old_i or (old_i & old_j) == old_j:
                comp_ij_old = 1
            else:
                comp_ij_old = 0
            
            total_old = contrib_i_old + contrib_j_old + comp_ij_old
            
            # New contributions of new_i and new_j with others and each other
            ab_i_new = others & new_i
            comp_i_new = (ab_i_new == others) | (ab_i_new == new_i)
            contrib_i_new = int(np.sum(comp_i_new))
            
            ab_j_new = others & new_j
            comp_j_new = (ab_j_new == others) | (ab_j_new == new_j)
            contrib_j_new = int(np.sum(comp_j_new))
            
            # Comparable between new_i and new_j
            if (new_i & new_j) == new_i or (new_i & new_j) == new_j:
                comp_ij_new = 1
            else:
                comp_ij_new = 0
            
            total_new = contrib_i_new + contrib_j_new + comp_ij_new
            
            delta = total_new - total_old
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                current[i] = new_i
                current[j] = new_j
                current_np[i] = new_i
                current_np[j] = new_j
                current_set.remove(old_i)
                current_set.remove(old_j)
                current_set.add(new_i)
                current_set.add(new_j)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
        
        # Occasional local search polish (1-opt greedy)
        if random.random() < 0.05:
            for _ in range(10):
                improved = False
                for k in range(M):
                    bd, bn = _best_replacement_np(current_np, k, all_masks_np, current_set)
                    if bn is not None and bd > 0:
                        old_k = current[k]
                        current[k] = bn
                        current_np[k] = bn
                        current_set.remove(old_k)
                        current_set.add(bn)
                        current_score += bd
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best = list(current)
                if not improved:
                    break

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_masks = None
    best_score = -1
    
    # Try multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    
    for seed in seeds:
        if time.time() >= deadline - 5:
            break
        
        # Initialize with structured greedy
        masks = _greedy_init(seed)
        
        # Run Simulated Annealing
        result_masks, result_score = _simulated_annealing(masks, deadline - 2)
        
        if result_score > best_score:
            best_score = result_score
            best_masks = result_masks
    
    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_VERT) if (mask >> b) & 1])
    
    return result
