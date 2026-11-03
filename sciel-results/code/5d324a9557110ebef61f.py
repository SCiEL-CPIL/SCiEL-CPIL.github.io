"""Structural approach: hierarchical blocks with geometric gap growth."""
import time
import math
import random

def _score_set(U):
    """Compute the lower bound score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_U = max(U)
    if max_U == 0:
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
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_base = math.log(2 * max_U + 1)
    if log_base <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / log_base
    return score


def _build_hierarchical(k_blocks, block_size, gap_ratio, start_gap=1):
    """
    Build U as k_blocks blocks of block_size elements each.
    Block i starts at position: start_gap * gap_ratio^i
    Within each block, elements are consecutive integers.
    """
    U = set()
    U.add(0)
    
    pos = start_gap
    for i in range(k_blocks):
        if i > 0:
            pos = int(start_gap * (gap_ratio ** i))
        for j in range(block_size):
            U.add(pos + j)
        if pos > 200000:
            break
    
    return sorted(U)


def _build_hierarchical_ap(k_blocks, block_size, block_step, gap_ratio, start_gap=1):
    """
    Build U as k_blocks blocks, each an arithmetic progression.
    Block i: elements at pos_i + j*block_step for j in range(block_size)
    pos_{i+1} = pos_i * gap_ratio (geometric gap growth)
    """
    U = set()
    U.add(0)
    
    pos = start_gap
    for i in range(k_blocks):
        if i > 0:
            pos = int(start_gap * (gap_ratio ** i))
        for j in range(block_size):
            val = pos + j * block_step
            if val > 200000:
                break
            U.add(val)
        if pos > 200000:
            break
    
    return sorted(U)


def _build_multi_scale(n_scales, scale_base, points_per_scale):
    """
    Multi-scale construction: at each scale s, place points_per_scale 
    points spread over a range of size scale_base^s.
    """
    U = set()
    U.add(0)
    
    for s in range(n_scales):
        scale = scale_base ** s
        if scale > 200000:
            break
        # Place points at regular intervals within this scale
        for j in range(points_per_scale):
            pos = int(j * scale / max(points_per_scale - 1, 1))
            U.add(pos)
    
    return sorted(U)


def _local_search_hierarchical(U, deadline, max_val_limit=200000):
    """
    Local search on the hierarchical representation.
    Try modifying block positions, sizes, and gap ratios.
    """
    U_set = set(U)
    best_score = _score_set(list(U_set))
    best_U = list(U_set)
    
    start_time = time.time()
    iteration = 0
    
    while time.time() < deadline - 5:
        iteration += 1
        improved = False
        
        # Generate candidates from current structure
        candidates = set()
        
        # Sums and differences of existing elements
        for a in list(U_set)[:30]:
            for b in list(U_set)[:30]:
                candidates.add(a + b)
                candidates.add(abs(a - b))
        
        # Try adding elements
        valid_candidates = [c for c in candidates if c not in U_set and 0 < c <= max_val_limit]
        random.shuffle(valid_candidates)
        
        for c in valid_candidates[:30]:
            new_U = U_set | {c}
            new_score = _score_set(list(new_U))
            if new_score > best_score:
                U_set = new_U
                best_score = new_score
                best_U = list(U_set)
                improved = True
                break
        
        # Try removing elements (if set is large enough)
        if len(U_set) > 5 and not improved:
            elements = [e for e in U_set if e != 0]
            random.shuffle(elements)
            for c in elements[:15]:
                new_U = U_set - {c}
                if len(new_U) < 2:
                    continue
                new_score = _score_set(list(new_U))
                if new_score > best_score:
                    U_set = new_U
                    best_score = new_score
                    best_U = list(U_set)
                    improved = True
                    break
        
        # Try swapping
        if len(U_set) > 5 and not improved:
            elements = [e for e in U_set if e != 0]
            if len(elements) >= 2:
                elem_to_remove = random.choice(elements)
                swap_candidates = [c for c in candidates if c not in U_set and 0 < c <= max_val_limit and c != elem_to_remove]
                random.shuffle(swap_candidates)
                for c in swap_candidates[:10]:
                    new_U = (U_set - {elem_to_remove}) | {c}
                    new_score = _score_set(list(new_U))
                    if new_score > best_score:
                        U_set = new_U
                        best_score = new_score
                        best_U = list(U_set)
                        improved = True
                        break
        
        if not improved and iteration > 50:
            break
    
    return best_U, best_score


def _ternary_search_param(lo, hi, func, deadline, tol=1e-6):
    """Ternary search for a single parameter."""
    if time.time() > deadline:
        return (lo + hi) / 2
    
    while hi - lo > tol and time.time() < deadline:
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        f1 = func(m1)
        f2 = func(m2)
        if f1 < f2:
            lo = m1
        else:
            hi = m2
    
    return (lo + hi) / 2


def construct():
    """Main construction function using hierarchical block structure."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    random.seed(12345)
    
    best_set = [0, 1, 2]
    best_score = _score_set(best_set)
    
    # Phase 1: Explore hierarchical constructions with geometric gaps
    if time.time() < deadline - 60:
        for k_blocks in [3, 4, 5, 6, 7, 8, 10, 12]:
            for block_size in [2, 3, 4, 5, 6, 8]:
                for gap_ratio in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
                    if time.time() > deadline - 30:
                        break
                    U = _build_hierarchical(k_blocks, block_size, gap_ratio)
                    if len(U) >= 2:
                        score = _score_set(U)
                        if score > best_score:
                            best_score = score
                            best_set = U
    
    # Phase 2: Arithmetic progression blocks
    if time.time() < deadline - 40:
        for k_blocks in [3, 4, 5, 6, 7]:
            for block_size in [2, 3, 4, 5]:
                for block_step in [1, 2, 3, 5]:
                    for gap_ratio in [2.0, 3.0, 4.0, 5.0, 6.0]:
                        if time.time() > deadline - 20:
                            break
                        U = _build_hierarchical_ap(k_blocks, block_size, block_step, gap_ratio)
                        if len(U) >= 2:
                            score = _score_set(U)
                            if score > best_score:
                                best_score = score
                                best_set = U
    
    # Phase 3: Multi-scale constructions
    if time.time() < deadline - 30:
        for n_scales in [3, 4, 5, 6, 7]:
            for scale_base in [2, 3, 4, 5]:
                for points_per_scale in [2, 3, 4, 5]:
                    if time.time() > deadline - 15:
                        break
                    U = _build_multi_scale(n_scales, scale_base, points_per_scale)
                    if len(U) >= 2:
                        score = _score_set(U)
                        if score > best_score:
                            best_score = score
                            best_set = U
    
    # Phase 4: Local search to refine the best set found
    if time.time() < deadline - 10 and len(best_set) >= 2:
        refined, refined_score = _local_search_hierarchical(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Phase 5: Final random perturbation and local search
    if time.time() < deadline - 5:
        # Try a few random perturbations of the best set
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = set(best_set)
            # Add a random element
            max_val = max(best_set) * 2
            for _ in range(3):
                c = random.randint(1, max_val)
                perturbed.add(c)
            # Maybe remove one
            if len(perturbed) > 3:
                removable = [e for e in perturbed if e != 0]
                if removable:
                    perturbed.discard(random.choice(removable))
            
            if len(perturbed) >= 2:
                score = _score_set(list(perturbed))
                if score > best_score:
                    best_score = score
                    best_set = list(perturbed)
        
        # Final local search
        if len(best_set) >= 2:
            refined, refined_score = _local_search_hierarchical(best_set, deadline)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    return best_set
