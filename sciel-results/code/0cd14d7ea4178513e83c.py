import time
import random
import sys
import numpy as np

N = 256

def _precompute():
    """Precompute all necessary structures."""
    # For 4-AP checking: for each x, precompute the set of "forbidden triples"
    # When adding x, check if any forbidden triple is fully in the mask
    # 4-AP: (a, a+d, a+2d, a+3d)
    # If x is position 0: need a+d, a+2d, a+3d all present
    # If x is position 1: need a, a+2d, a+3d all present (where a=x-d)
    # If x is position 2: need a, a+d, a+3d all present (where a=x-2d)
    # If x is position 3: need a, a+d, a+2d all present (where a=x-3d)
    
    # For each x, store list of bitmasks (each is a 3-element bitmask)
    forbidden_4ap_bitmasks = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                # x=a: need b,c,e
                forbidden_4ap_bitmasks[a].append((1 << b) | (1 << c) | (1 << e))
                # x=b: need a,c,e
                forbidden_4ap_bitmasks[b].append((1 << a) | (1 << c) | (1 << e))
                # x=c: need a,b,e
                forbidden_4ap_bitmasks[c].append((1 << a) | (1 << b) | (1 << e))
                # x=e: need a,b,c
                forbidden_4ap_bitmasks[e].append((1 << a) | (1 << b) | (1 << c))
    
    # For 3-AP counting: for each middle b, list of (a, c) pairs
    three_ap_pairs = [[] for _ in range(N + 1)]
    for b in range(1, N + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c > b and c <= N:
                three_ap_pairs[b].append((a, c))
    
    # For delta_add: precompute for each x the bitmasks of pairs
    # x as middle: pairs (a,c) where a+c=2x
    # x as first: pairs (b,c) where x,b,c form AP -> b,c with 2b=x+c, b>x, c>b
    # x as last: pairs (a,b) where a,b,x form AP -> a+b=2b... a+2b=x? No: a,b,x with b-a=x-b -> 2b=a+x
    x_as_first_bitmasks = [[] for _ in range(N + 1)]
    x_as_last_bitmasks = [[] for _ in range(N + 1)]
    x_as_middle_bitmasks = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as middle
        for a, c in three_ap_pairs[x]:
            x_as_middle_bitmasks[x].append((1 << a) | (1 << c))
        # x as first: (x, b, c) where b > x, c = 2b - x > b, c <= N
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                x_as_first_bitmasks[x].append((1 << b) | (1 << c))
        # x as last: (a, b, x) where a < b < x, a + x = 2b
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if a < b < x:
                    x_as_last_bitmasks[x].append((1 << a) | (1 << b))
    
    return forbidden_4ap_bitmasks, three_ap_pairs, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks


def _creates_4ap(mask, x, forbidden_4ap_bitmasks):
    """Check if adding x to mask creates a 4-AP."""
    for bm in forbidden_4ap_bitmasks[x]:
        if (mask & bm) == bm:
            return True
    return False


def _delta_add(mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks):
    """Count new 3-APs created by adding x."""
    count = 0
    for bm in x_as_middle_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    for bm in x_as_first_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    for bm in x_as_last_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    return count


def _delta_remove(mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks):
    """Count 3-APs lost by removing x from mask."""
    count = 0
    for bm in x_as_middle_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    for bm in x_as_first_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    for bm in x_as_last_bitmasks[x]:
        if (mask & bm) == bm:
            count += 1
    return count


def _count_3aps(mask, three_ap_pairs):
    """Count total 3-APs in set represented by mask."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        for a, c in three_ap_pairs[b]:
            if (mask & (1 << a)) and (mask & (1 << c)):
                count += 1
    return count


def _greedy_construct(forbidden_4ap_bitmasks, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, order):
    """Greedy construction with scoring: add element that gives max 3-APs without creating 4-AP."""
    mask = 0
    remaining = list(order)
    
    while remaining:
        best_x = None
        best_delta = 0
        for x in remaining:
            if _creates_4ap(mask, x, forbidden_4ap_bitmasks):
                continue
            d = _delta_add(mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
            if d > best_delta:
                best_delta = d
                best_x = x
        if best_x is not None:
            mask |= (1 << best_x)
            remaining.remove(best_x)
        else:
            break
    return mask


def _local_search(mask, forbidden_4ap_bitmasks, three_ap_pairs, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, deadline):
    """Hill climbing local search with add/remove/swap moves."""
    current_mask = mask
    current_score = _count_3aps(current_mask, three_ap_pairs)
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
        
        # Try all additions
        for y in unset_elements[:]:
            if _creates_4ap(current_mask, y, forbidden_4ap_bitmasks):
                continue
            delta = _delta_add(current_mask, y, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
            if delta > 0:
                current_mask |= (1 << y)
                current_score += delta
                improved = True
                unset_elements.remove(y)
                set_elements.append(y)
        
        # Try all removals (if removing x doesn't lose 3-APs, or frees up for better additions)
        if time.time() < deadline:
            for x in set_elements[:]:
                delta_remove = _delta_remove(current_mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
                # Only remove if it doesn't hurt (we'll try to add back later)
                # Actually, let's try swaps instead
                pass
        
        # Try all swaps
        if time.time() < deadline:
            for x in set_elements[:]:
                if time.time() > deadline:
                    break
                delta_x = _delta_remove(current_mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
                temp_mask = current_mask & ~(1 << x)
                for y in unset_elements:
                    if _creates_4ap(temp_mask, y, forbidden_4ap_bitmasks):
                        continue
                    delta_y = _delta_add(temp_mask, y, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
                    if delta_y - delta_x > 0:
                        current_mask = temp_mask | (1 << y)
                        current_score += (delta_y - delta_x)
                        improved = True
                        set_elements.remove(x)
                        set_elements.append(y)
                        unset_elements.remove(y)
                        unset_elements.append(x)
                        break
    
    return current_mask, current_score


def _simulated_annealing(mask, forbidden_4ap_bitmasks, three_ap_pairs, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, deadline, seed=42):
    """Simulated annealing."""
    random.seed(seed)
    current_mask = mask
    current_score = _count_3aps(current_mask, three_ap_pairs)
    best_mask = current_mask
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
    unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
    
    while time.time() < deadline and T > T_min:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3 and unset_elements:
            # Add a random element
            y = random.choice(unset_elements)
            if _creates_4ap(current_mask, y, forbidden_4ap_bitmasks):
                T *= cooling
                continue
            delta = _delta_add(current_mask, y, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
            new_mask = current_mask | (1 << y)
            new_score = current_score + delta
            if delta >= 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                unset_elements.remove(y)
                set_elements.append(y)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        elif move_type < 0.6 and set_elements:
            # Remove a random element
            x = random.choice(set_elements)
            delta = -_delta_remove(current_mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
            new_mask = current_mask & ~(1 << x)
            new_score = current_score + delta
            if delta >= 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                set_elements.remove(x)
                unset_elements.append(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Swap
            if set_elements and unset_elements:
                x = random.choice(set_elements)
                y = random.choice(unset_elements)
                temp_mask = current_mask & ~(1 << x)
                if _creates_4ap(temp_mask, y, forbidden_4ap_bitmasks):
                    T *= cooling
                    continue
                delta_x = _delta_remove(current_mask, x, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
                delta_y = _delta_add(temp_mask, y, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks)
                delta = delta_y - delta_x
                new_mask = temp_mask | (1 << y)
                new_score = current_score + delta
                if delta >= 0 or random.random() < np.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    set_elements.remove(x)
                    set_elements.append(y)
                    unset_elements.remove(y)
                    unset_elements.append(x)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
        
        T *= cooling
    
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    # Precompute
    forbidden_4ap_bitmasks, three_ap_pairs, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks = _precompute()
    
    best_mask = 0
    best_score = 0
    
    # Phase 1: Multiple greedy constructions
    random.seed(42)
    
    # Natural order greedy
    mask1 = _greedy_construct(forbidden_4ap_bitmasks, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, list(range(1, N + 1)))
    score1 = _count_3aps(mask1, three_ap_pairs)
    if score1 > best_score:
        best_mask = mask1
        best_score = score1
    
    # Reverse order greedy
    mask2 = _greedy_construct(forbidden_4ap_bitmasks, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, list(range(N, 0, -1)))
    score2 = _count_3aps(mask2, three_ap_pairs)
    if score2 > best_score:
        best_mask = mask2
        best_score = score2
    
    # Randomized greedy
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask_t = _greedy_construct(forbidden_4ap_bitmasks, x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks, order)
        score_t = _count_3aps(mask_t, three_ap_pairs)
        if score_t > best_score:
            best_mask = mask_t
            best_score = score_t
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 15:
        sa_mask, sa_score = _simulated_annealing(
            best_mask, forbidden_4ap_bitmasks, three_ap_pairs,
            x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks,
            deadline - 15, seed=123
        )
        if sa_score > best_score:
            best_mask = sa_mask
            best_score = sa_score
    
    # Phase 3: Local search on best
    if time.time() < deadline - 3:
        best_mask, best_score = _local_search(
            best_mask, forbidden_4ap_bitmasks, three_ap_pairs,
            x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks,
            deadline - 3
        )
    
    # Phase 4: Another SA round if time permits
    if time.time() < deadline - 2:
        sa_mask2, sa_score2 = _simulated_annealing(
            best_mask, forbidden_4ap_bitmasks, three_ap_pairs,
            x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks,
            deadline - 2, seed=777
        )
        if sa_score2 > best_score:
            best_mask = sa_mask2
            best_score = sa_score2
    
    # Final local search
    if time.time() < deadline - 0.5:
        best_mask, best_score = _local_search(
            best_mask, forbidden_4ap_bitmasks, three_ap_pairs,
            x_as_first_bitmasks, x_as_last_bitmasks, x_as_middle_bitmasks,
            deadline - 0.5
        )
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    return result
