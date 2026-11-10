import time
import random

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _init_bitmasks():
    """Precompute bitmask representations for fast 4-AP checking."""
    # For each position x and difference d, precompute the bitmask of the 4-AP
    # that would be formed if x is at a specific position in the 4-AP.
    # We'll compute on-the-fly but cache the key operations.
    pass


def _has_4ap_bitmask(mask, x):
    """Check if adding element x (1-indexed) to set represented by mask creates a 4-AP.
    Returns True if a 4-AP would be created."""
    bit = 1 << (x - 1)
    d_max = (N - 1) // 3
    
    # x as first term: x, x+d, x+2d, x+3d
    for d in range(1, d_max + 1):
        b1 = x + d
        b2 = x + 2 * d
        b3 = x + 3 * d
        if b3 > N:
            break
        if (mask >> (b1 - 1)) & 1 and (mask >> (b2 - 1)) & 1 and (mask >> (b3 - 1)) & 1:
            return True
    
    # x as second term: x-d, x, x+d, x+2d
    for d in range(1, d_max + 1):
        b0 = x - d
        b2 = x + d
        b3 = x + 2 * d
        if b0 < 1 or b3 > N:
            if b3 > N:
                break
            continue
        if (mask >> (b0 - 1)) & 1 and (mask >> (b2 - 1)) & 1 and (mask >> (b3 - 1)) & 1:
            return True
    
    # x as third term: x-2d, x-d, x, x+d
    for d in range(1, d_max + 1):
        b0 = x - 2 * d
        b1 = x - d
        b3 = x + d
        if b0 < 1 or b3 > N:
            if b0 < 1:
                break
            continue
        if (mask >> (b0 - 1)) & 1 and (mask >> (b1 - 1)) & 1 and (mask >> (b3 - 1)) & 1:
            return True
    
    # x as fourth term: x-3d, x-2d, x-d, x
    for d in range(1, d_max + 1):
        b0 = x - 3 * d
        b1 = x - 2 * d
        b2 = x - d
        if b0 < 1:
            break
        if (mask >> (b0 - 1)) & 1 and (mask >> (b1 - 1)) & 1 and (mask >> (b2 - 1)) & 1:
            return True
    
    return False


def _count_3ap_bitmask(mask):
    """Count 3-APs in set represented by bitmask."""
    count = 0
    for a in range(1, N + 1):
        if not ((mask >> (a - 1)) & 1):
            continue
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            if ((mask >> (b - 1)) & 1) and ((mask >> (c - 1)) & 1):
                count += 1
    return count


def _count_3ap_fast(mask):
    """Faster 3-AP count using precomputed pairs."""
    count = 0
    for a in range(1, N + 1):
        if not ((mask >> (a - 1)) & 1):
            continue
        # For each a in set, count pairs (b,c) with b=(a+c)/2, b,c in set
        # Equivalently: for each d, check if a+d and a+2d are in set
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            if ((mask >> (b - 1)) & 1) and ((mask >> (c - 1)) & 1):
                count += 1
    return count


def _grasp_construct(alpha, start_time, rng):
    """GRASP greedy construction with randomization.
    At each step, choose from the candidate list (elements that don't create 4-AP)
    using a restricted candidate list of size alpha * |candidates|, picking randomly."""
    mask = 0
    selected = set()
    
    # Initial: pick a random starting element
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    while candidates:
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # Find all candidates that can be added without creating a 4-AP
        feasible = []
        for x in candidates:
            if not _has_4ap_bitmask(mask, x):
                feasible.append(x)
        
        if not feasible:
            break
        
        # Restricted candidate list
        rcl_size = max(1, int(alpha * len(feasible)))
        rcl = rng.sample(feasible, min(rcl_size, len(feasible)))
        x = rng.choice(rcl)
        
        mask |= (1 << (x - 1))
        selected.add(x)
        candidates.remove(x)
    
    return selected, mask


def _tabu_search(selected, mask, start_time, max_iters, tabu_len, rng):
    """Tabu search with add/remove/swap moves."""
    best_score = _count_3ap_fast(mask)
    best_mask = mask
    best_set = set(selected)
    
    current_mask = mask
    current_set = set(selected)
    current_score = best_score
    
    # Tabu lists: store (element, iteration) for recently moved elements
    tabu_add = {}   # element -> iteration when it was removed (tabu to add)
    tabu_remove = {} # element -> iteration when it was added (tabu to remove)
    tabu_swap_out = {}
    tabu_swap_in = {}
    
    iteration = 0
    no_improve = 0
    max_no_improve = 50
    
    while iteration < max_iters:
        if time.time() - start_time > DEADLINE - 3:
            break
        
        best_move = None
        best_move_score = current_score
        best_move_type = None
        best_move_params = None
        
        # Generate candidate moves
        # 1. Add moves: add an element not in current set
        for x in range(1, N + 1):
            if x in current_set:
                continue
            if tabu_add.get(x, -1) > iteration - tabu_len:
                continue
            if _has_4ap_bitmask(current_mask, x):
                continue
            new_mask = current_mask | (1 << (x - 1))
            new_score = _count_3ap_fast(new_mask)
            if new_score > best_move_score:
                best_move_score = new_score
                best_move_type = 'add'
                best_move_params = x
        
        # 2. Remove moves: remove an element from current set
        for x in current_set:
            if tabu_remove.get(x, -1) > iteration - tabu_len:
                continue
            new_mask = current_mask & ~(1 << (x - 1))
            new_score = _count_3ap_fast(new_mask)
            if new_score > best_move_score:
                best_move_score = new_score
                best_move_type = 'remove'
                best_move_params = x
        
        # 3. Swap moves: remove one, add another
        # Only try a subset for speed
        if len(current_set) > 0:
            swap_candidates = list(current_set)
            if len(swap_candidates) > 30:
                swap_candidates = rng.sample(swap_candidates, 30)
            for x in swap_candidates:
                if tabu_swap_out.get(x, -1) > iteration - tabu_len:
                    continue
                base_mask = current_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if y in current_set or y == x:
                        continue
                    if tabu_swap_in.get(y, -1) > iteration - tabu_len:
                        continue
                    if _has_4ap_bitmask(base_mask, y):
                        continue
                    new_mask = base_mask | (1 << (y - 1))
                    new_score = _count_3ap_fast(new_mask)
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move_type = 'swap'
                        best_move_params = (x, y)
        
        # Apply best move if found
        if best_move_type is not None:
            if best_move_type == 'add':
                x = best_move_params
                current_mask |= (1 << (x - 1))
                current_set.add(x)
                tabu_add[x] = iteration  # was not here, now is -> tabu to remove
                tabu_remove[x] = iteration
            elif best_move_type == 'remove':
                x = best_move_params
                current_mask &= ~(1 << (x - 1))
                current_set.discard(x)
                tabu_remove[x] = iteration
                tabu_add[x] = iteration
            elif best_move_type == 'swap':
                x, y = best_move_params
                current_mask &= ~(1 << (x - 1))
                current_mask |= (1 << (y - 1))
                current_set.discard(x)
                current_set.add(y)
                tabu_swap_out[x] = iteration
                tabu_swap_in[y] = iteration
                tabu_remove[x] = iteration
                tabu_add[x] = iteration
                tabu_remove[y] = iteration
                tabu_add[y] = iteration
            
            current_score = best_move_score
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                best_set = set(current_set)
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        iteration += 1
    
    return best_set, best_mask, best_score


def construct():
    start_time = time.time()
    rng = random.Random(12345)
    
    best_score = 0
    best_set = set()
    best_mask = 0
    
    # Multiple GRASP + Tabu restarts with different alpha values
    alphas = [0.3, 0.5, 0.7, 1.0, 0.4, 0.6, 0.8]
    tabu_lens = [7, 9, 11, 13]
    
    for alpha in alphas:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for tabu_len in tabu_lens:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # GRASP construction
            selected, mask = _grasp_construct(alpha, start_time, rng)
            if not selected:
                continue
            
            score = _count_3ap_fast(mask)
            
            # Tabu search
            if time.time() - start_time > DEADLINE - 5:
                # Just keep the GRASP result if it's better
                if score > best_score:
                    best_score = score
                    best_set = selected
                    best_mask = mask
                continue
            
            max_tabu_iters = 2000
            new_set, new_mask, new_score = _tabu_search(
                selected, mask, start_time, max_tabu_iters, tabu_len, rng
            )
            
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
                best_mask = new_mask
    
    # Final: try to add any remaining elements
    if time.time() - start_time < DEADLINE - 2:
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if not _has_4ap_bitmask(best_mask, x):
                new_mask = best_mask | (1 << (x - 1))
                new_score = _count_3ap_fast(new_mask)
                if new_score > best_score:
                    best_score = new_score
                    best_set.add(x)
                    best_mask = new_mask
    
    # One more pass of local improvement
    if time.time() - start_time < DEADLINE - 2:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            # Try swaps
            for x in list(best_set):
                if time.time() - start_time > DEADLINE - 2:
                    break
                base_mask = best_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    if y in best_set or y == x:
                        continue
                    if _has_4ap_bitmask(base_mask, y):
                        continue
                    new_mask = base_mask | (1 << (y - 1))
                    new_score = _count_3ap_fast(new_mask)
                    if new_score > best_score:
                        best_set.discard(x)
                        best_set.add(y)
                        best_mask = new_mask
                        best_score = new_score
                        improved = True
                        break
    
    return sorted(best_set)
