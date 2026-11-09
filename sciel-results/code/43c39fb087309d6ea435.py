import time
import random
import math

N_ORIG = 12
M = 40
N_OPT = 6  # We hypothesize the optimal set lies in B_6
FULL_MASK_OPT = (1 << N_OPT) - 1

def _count_pairs_opt(masks):
    """Count comparable pairs in a list of masks (size N_OPT)."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _optimize_subset():
    """Find best 40 masks from B_6 using Simulated Annealing with incremental scoring."""
    all_masks = list(range(1 << N_OPT))
    num_masks = len(all_masks)
    
    best_score = -1
    best_mask_set = None
    
    # Run multiple restarts
    # We have 90s total. Let's allocate 80s for search.
    start_time = time.time()
    deadline = start_time + 80
    
    num_restarts = 20 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(1000 + restart)
        
        # Initialization: Greedy or Random?
        # Greedy: Start with empty, add best mask, repeat.
        # Or Random: Sample M masks.
        # Let's try Random first, then SA.
        current_set = set(random.sample(all_masks, M))
        current_list = list(current_set)
        
        # Calculate initial degrees
        # degree[i] is count of j != i such that current_list[i] and current_list[j] are comparable
        degrees = [0] * M
        current_score = 0
        for i in range(M):
            a = current_list[i]
            for j in range(M):
                if i == j: continue
                b = current_list[j]
                if (a & b) == a or (a & b) == b:
                    degrees[i] += 1
                    if i < j:
                        current_score += 1
        
        if current_score > best_score:
            best_score = current_score
            best_mask_set = set(current_list)
        
        T = 10.0
        min_T = 0.1
        cooling = 0.999
        no_improve = 0
        max_iters = 50000 # High limit, controlled by time
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > 5000:
                break
            
            # Pick index to replace
            i = random.randint(0, M - 1)
            old_mask = current_list[i]
            
            # Pick new mask not in current_set
            # Efficient rejection sampling
            while True:
                new_mask = random.randint(0, FULL_MASK_OPT)
                if new_mask not in current_set:
                    break
            
            # Calculate delta
            # Remove old_mask: loses comparabilities with all j != i
            # Add new_mask: gains comparabilities with all j != i
            
            # Note: degrees[i] changes, but we only need the delta for the total score.
            # Total score = sum(degrees) / 2.
            # Delta = (sum of new comparabilities for new_mask) - (sum of old comparabilities for old_mask)
            # But we must exclude the pair (i, i) which is not counted.
            # Comparabilities with j != i are symmetric.
            
            delta = 0
            for j in range(M):
                if j == i: continue
                m_j = current_list[j]
                
                old_comp = 1 if ((old_mask & m_j) == old_mask or (old_mask & m_j) == m_j) else 0
                new_comp = 1 if ((new_mask & m_j) == new_mask or (new_mask & m_j) == m_j) else 0
                
                delta += new_comp - old_comp
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                # Update state
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_list[i] = new_mask
                current_score += delta
                no_improve = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask_set = set(current_list)
            else:
                no_improve += 1
                
            T *= cooling
            if T < min_T:
                T = min_T

    return list(best_mask_set), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin
    
    # Run the optimizer
    masks, score = _optimize_subset()
        
    # Convert to list of lists of ints
    result = []
    for mask in masks:
        result.append([b for b in range(N_OPT) if (mask >> b) & 1])
    
    return result
