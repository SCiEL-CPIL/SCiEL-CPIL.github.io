"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
MAX_VAL = 128
_TIME_LIMIT = 85.0  # Evaluator kills at 90s; use 85s for safety

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        for j in range(i + 1, N):
            if not (mask >> j) & 1:
                continue
            # a = i+1, b = j+1
            # a+b = i+j+2, bit index = i+j+1
            if i + j + 1 >= N:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            # candidates: k in S, k+i+1 in S, k+j+1 in S
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            # Exclude c == a (k==i) and c == b (k==j)
            if candidates & ~((1 << i) | (1 << j)):
                return False
    return True

def _construct_baseline():
    """
    Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
    """
    mask = 0
    # Add 65..128
    for v in range(65, 129):
        mask |= (1 << (v - 1))
    
    # Add specific small numbers
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        mask |= (1 << (v - 1))
        
    return mask

def _greedy_fill(mask, order=None):
    """
    Try to add elements from 'order' to mask if valid.
    """
    if order is None:
        order = list(range(1, N + 1))
        
    for v in order:
        bit = 1 << (v - 1)
        if mask & bit:
            continue
        trial = mask | bit
        if _is_valid_mask(trial):
            mask = trial
    return mask

def _local_search(mask, deadline):
    """
    Local search to improve mask size.
    Strategy: 
    1. Try adding any single missing element.
    2. Try 'remove one, add many' moves.
    3. Try 'remove two, add many' moves (2-exchange).
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    stuck_rounds = 0
    max_stuck = 3
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                stuck_rounds = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try 'remove one, add many'
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
        
        for v in elems:
            if time.time() >= deadline:
                break
            bit = 1 << (v - 1)
            reduced_mask = current_mask & ~bit
            
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << (add_v - 1)
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_mask(trial):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
        
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
            stuck_rounds = 0
            continue
        
        # Phase 3: Try 'remove two, add many' (2-exchange)
        found_better_2 = False
        best_candidate_2 = current_mask
        best_candidate_2_size = current_size
        
        if len(elems) >= 2:
            # Limit the number of pairs to try for speed
            max_pairs = min(len(elems) * (len(elems) - 1) // 2, 200)
            pair_count = 0
            for idx_i in range(len(elems)):
                if time.time() >= deadline or pair_count >= max_pairs:
                    break
                for idx_j in range(idx_i + 1, len(elems)):
                    if time.time() >= deadline or pair_count >= max_pairs:
                        break
                    pair_count += 1
                    a = elems[idx_i]
                    b = elems[idx_j]
                    reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    
                    candidate = reduced
                    for add_v in range(1, N + 1):
                        add_bit = 1 << (add_v - 1)
                        if candidate & add_bit:
                            continue
                        trial = candidate | add_bit
                        if _is_valid_mask(trial):
                            candidate = trial
                    
                    if candidate.bit_count() > best_candidate_2_size:
                        best_candidate_2 = candidate
                        best_candidate_2_size = candidate.bit_count()
                        found_better_2 = True
        
        if found_better_2:
            current_mask = best_candidate_2
            current_size = best_candidate_2_size
            stuck_rounds = 0
            continue
        
        # No improvement in any phase
        stuck_rounds += 1
        if stuck_rounds >= max_stuck:
            break
        
    return current_mask

def _branch_and_bound(mask, depth=0, deadline=0.0):
    """
    Branch and bound to find maximum independent set.
    This is a heuristic implementation.
    """
    if time.time() > deadline:
        return mask
    
    current_size = mask.bit_count()
    
    # Try to add elements greedily
    # Order by "constraint tightness" or just random?
    # Let's try adding elements that are most "compatible".
    
    # For simplicity, we'll just do a few random restarts of greedy inside B&B logic
    # Or rather, use B&B to explore removing/adding specific conflicting nodes.
    
    # Given the complexity, a full B&B might be too slow or complex to implement correctly
    # in the remaining time. The local search "remove one, add many" is effectively
    # a 1-exchange move which is a form of local B&B.
    
    # Instead, let's implement a "2-exchange" move: remove two, add many.
    # This allows escaping deeper local minima.
    
    best_mask = mask
    best_size = current_size
    
    # Get list of elements in mask
    elems = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    # If size is small enough, try all pairs
    if len(elems) <= 30:
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if time.time() > deadline:
                    return best_mask
                
                a = elems[i]
                b = elems[j]
                reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                
                candidate = _greedy_fill(reduced)
                if candidate.bit_count() > best_size:
                    best_size = candidate.bit_count()
                    best_mask = candidate
    else:
        # Random sampling of pairs for larger sets
        num_samples = 100
        for _ in range(num_samples):
            if time.time() > deadline:
                return best_mask
            
            if len(elems) < 2:
                break
            a, b = random.sample(elems, 2)
            reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
            
            candidate = _greedy_fill(reduced)
            if candidate.bit_count() > best_size:
                best_size = candidate.bit_count()
                best_mask = candidate

    return best_mask

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety
    deadline = start_time + 80.0
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill
    if time.time() < deadline - 10:
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 5:
        searched_mask = _local_search(best_mask, deadline - 5)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = best_mask.bit_count()
            
    # 4. Branch and Bound / 2-exchange search
    if time.time() < deadline - 2:
        bb_mask = _branch_and_bound(best_mask, deadline=deadline - 2)
        if bb_mask.bit_count() > best_size:
            best_mask = bb_mask
            best_size = best_mask.bit_count()
            
    # 5. Random restarts if time remains
    # Try different initial orders for greedy
    if time.time() < deadline - 2:
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            random.seed(restart)
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            
            # Perturbation: remove 5 random elements, then local search.
            if best_size > 0:
                elems = [i + 1 for i in range(N) if (best_mask >> i) & 1]
                if len(elems) >= 5:
                    to_remove = random.sample(elems, 5)
                    perturbed = best_mask
                    for r in to_remove:
                        perturbed &= ~(1 << (r - 1))
                    
                    # Greedy fill on perturbed
                    filled_p = _greedy_fill(perturbed, perm)
                    
                    # Local search on filled_p
                    searched_p = _local_search(filled_p, deadline - 2)
                    
                    if searched_p.bit_count() > best_size:
                        best_mask = searched_p
                        best_size = best_mask.bit_count()
                        
                    # Try B&B on the perturbed result too?
                    # Only if we have time
                    if time.time() < deadline - 1:
                        bb_p = _branch_and_bound(searched_p, deadline=deadline - 1)
                        if bb_p.bit_count() > best_size:
                            best_mask = bb_p
                            best_size = best_mask.bit_count()

    # Final validity check
    if not _is_valid_mask(best_mask):
        # Fallback to baseline
        best_mask = _construct_baseline()
        
    # Convert mask to list of ints
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
