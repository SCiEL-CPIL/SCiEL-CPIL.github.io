import time
import random
import math
import numpy as np

N = 12
M = 40
FULL_MASK = (1 << N) - 1
NUM_MASKS = 1 << N

# Precompute comparability matrix
_comp_cache = None

def _get_comp_matrix():
    global _comp_cache
    if _comp_cache is not None:
        return _comp_cache
    
    comp = [None] * NUM_MASKS
    for i in range(NUM_MASKS):
        all_masks = np.arange(NUM_MASKS, dtype=np.int64)
        inter = i & all_masks
        is_sub = (inter == i)
        is_super = (inter == all_masks)
        comp[i] = (is_sub | is_super).astype(np.int8)
    _comp_cache = comp
    return _comp_cache

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _init_family():
    """Initialize family with a mix of chains and greedy fill."""
    random.seed(42)
    family = []
    used = set()
    
    # Generate some maximal chains
    for _ in range(4):
        chain = [0]
        cur = 0
        while cur != FULL_MASK:
            available_bits = [b for b in range(N) if not (cur >> b) & 1]
            if not available_bits:
                break
            b = random.choice(available_bits)
            cur = cur | (1 << b)
            if cur not in used:
                family.append(cur)
                used.add(cur)
            else:
                pass
        if 0 not in used:
            family.append(0)
            used.add(0)
        if FULL_MASK not in used:
            family.append(FULL_MASK)
            used.add(FULL_MASK)
            
    # Greedy fill to reach M
    all_masks = list(range(NUM_MASKS))
    random.shuffle(all_masks)
    
    while len(family) < M:
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in used:
                continue
            score = 0
            for c in family:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            break
        family.append(best_mask)
        used.add(best_mask)
        
    if len(family) > M:
        family = family[:M]
        used = set(family)
        
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 58
    
    comp = _get_comp_matrix()
    
    # Initialize
    family = _init_family()
    current_score = _count_pairs(family)
    best_score = current_score
    best_family = list(family)
    
    # Maintain compat_count: compat_count[k] = sum_{j in family} comp[k][j]
    compat_count = np.zeros(NUM_MASKS, dtype=np.int32)
    for mask in family:
        compat_count += comp[mask]
    
    # Local Search Loop with simulated annealing
    T = 5.0
    cooling = 0.99995
    min_T = 0.05
    
    # Track no-improvement counter
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        
        # Pick a random index to replace
        i = random.randint(0, M - 1)
        old_mask = family[i]
        
        # Calculate scores for all potential replacements
        scores = compat_count - comp[old_mask]
        
        # Mask out sets already in family (except old_mask which is being replaced)
        # We need to exclude all current family members from consideration
        for m in family:
            if m != old_mask:
                scores[m] = -999999
        
        # Find best candidate
        best_candidate = int(np.argmax(scores))
        best_candidate_score = scores[best_candidate]
        
        if best_candidate_score < -999999:
            # All candidates are in family, pick random
            best_candidate = random.randint(0, FULL_MASK)
            while best_candidate in family and best_candidate != old_mask:
                best_candidate = random.randint(0, FULL_MASK)
            best_candidate_score = scores[best_candidate]
        
        new_mask = best_candidate
        
        # Calculate delta
        old_contrib = compat_count[old_mask] - 1
        new_contrib = best_candidate_score
        
        delta = new_contrib - old_contrib
        
        # Acceptance
        accept = False
        if delta > 0:
            accept = True
        else:
            # Simulated annealing acceptance
            if T > min_T:
                accept_prob = math.exp(delta / T)
                accept = random.random() < accept_prob
        
        if accept:
            # Accept move
            family[i] = new_mask
            compat_count += comp[new_mask]
            compat_count -= comp[old_mask]
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Full-scan local search every N steps or when stuck
        if no_improve >= max_no_improve or (no_improve > 0 and no_improve % 500 == 0):
            # Try full scan for each position
            improved = False
            for pos in range(M):
                old_m = family[pos]
                sc = compat_count - comp[old_m]
                for m in family:
                    if m != old_m:
                        sc[m] = -999999
                
                best_c = int(np.argmax(sc))
                best_sc = sc[best_c]
                
                if best_sc < -999999:
                    continue
                
                old_c = compat_count[old_m] - 1
                d = best_sc - old_c
                
                if d > 0:
                    family[pos] = best_c
                    compat_count += comp[best_c]
                    compat_count -= comp[old_m]
                    current_score += d
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)
            
            if improved:
                no_improve = 0
            else:
                # Reset temperature to escape local minimum
                T = max(T, 2.0)
                no_improve = 0
        
        # Check time budget
        if time.time() > deadline - 2:
            break
    
    # Final polish: try all replacements one more time
    if time.time() < deadline - 1:
        for pos in range(M):
            old_m = family[pos]
            sc = compat_count - comp[old_m]
            for m in family:
                if m != old_m:
                    sc[m] = -999999
            
            best_c = int(np.argmax(sc))
            best_sc = sc[best_c]
            
            if best_sc < -999999:
                continue
            
            old_c = compat_count[old_m] - 1
            d = best_sc - old_c
            
            if d > 0:
                family[pos] = best_c
                compat_count += comp[best_c]
                compat_count -= comp[old_m]
                current_score += d
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
    
    # Convert best_family to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result

if __name__ == "__main__":
    res = construct()
    assert len(res) == 40
    assert all(isinstance(s, list) for s in res)
    assert all(isinstance(x, int) for s in res for x in s)
    masks = [sum(1 << b for b in s) for s in res]
    print("Actual Score:", _count_pairs(masks))
