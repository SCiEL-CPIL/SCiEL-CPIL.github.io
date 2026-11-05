import time
import random

N = 256
T_LIMIT = 90.0

# ---------- Precomputation ----------
# For each element x, precompute:
#   four_conflict_masks[x]: list of bitmasks; adding x creates a 4-AP iff (A & m) == m for some m
#   three_pair_masks[x]:    list of 2-element bitmasks; adding x creates a 3-AP iff (A & m) == m
four_conflict_masks = [[] for _ in range(N + 1)]
three_pair_masks = [[] for _ in range(N + 1)]

# 4-APs
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_conflict_masks[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (e-1)))
        four_conflict_masks[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (e-1)))
        four_conflict_masks[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (e-1)))
        four_conflict_masks[e].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))

# 3-APs
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_pair_masks[a].append((1 << (b-1)) | (1 << (c-1)))
        three_pair_masks[b].append((1 << (a-1)) | (1 << (c-1)))
        three_pair_masks[c].append((1 << (a-1)) | (1 << (b-1)))

def elem_bit(x):
    return 1 << (x - 1)

def is_4ap_free_add(A, x):
    for m in four_conflict_masks[x]:
        if (A & m) == m:
            return False
    return True

def delta_3aps_add(A, x):
    c = 0
    for m in three_pair_masks[x]:
        if (A & m) == m:
            c += 1
    return c

def count_3aps(A):
    # Each 3-AP counted 3 times (once per element)
    total = 0
    bits = A
    while bits:
        lsb = bits & (-bits)
        x = lsb.bit_length()  # 1-indexed
        total += delta_3aps_add(A, x)
        bits ^= lsb
    return total // 3

def greedy_initial(seed_val=0):
    rng = random.Random(seed_val)
    A = 0
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        rng.shuffle(candidates)
    for x in candidates:
        if is_4ap_free_add(A, x):
            A |= elem_bit(x)
    return A

def local_search_1swap(A, deadline):
    """Fast 1-swap local search to polish."""
    cur_score = count_3aps(A)
    while time.time() < deadline:
        improved = False
        # Best addition
        best_add = -1
        best_add_d = 0
        for x in range(1, N + 1):
            if (A >> (x-1)) & 1:
                continue
            if is_4ap_free_add(A, x):
                d = delta_3aps_add(A, x)
                if d > best_add_d:
                    best_add_d = d
                    best_add = x
        if best_add != -1:
            A |= elem_bit(best_add)
            cur_score += best_add_d
            improved = True
            continue
        # Best swap
        best_swap = None
        best_swap_score = cur_score
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((A >> (x-1)) & 1):
                continue
            loss = delta_3aps_add(A, x)
            A_temp = A & ~elem_bit(x)
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if y == x or ((A_temp >> (y-1)) & 1):
                    continue
                if not is_4ap_free_add(A_temp, y):
                    continue
                gain = delta_3aps_add(A_temp, y)
                ns = cur_score - loss + gain
                if ns > best_swap_score:
                    best_swap_score = ns
                    best_swap = (x, y)
        if best_swap:
            x, y = best_swap
            A = (A & ~elem_bit(x)) | elem_bit(y)
            cur_score = best_swap_score
            improved = True
            continue
        if not improved:
            break
    return A

def simulated_annealing(A_init, deadline, rng):
    """
    SA with 1-move (add/remove/swap) and 2-opt (remove 2, add 2) neighborhoods.
    """
    A = A_init
    cur_score = count_3aps(A)
    best_A = A
    best_score = cur_score
    
    T_start = 50.0
    T_end = 0.1
    # Total steps we hope to do; we'll run until deadline
    step = 0
    # We'll use a geometric cooling schedule based on elapsed fraction
    # But simpler: run until deadline, cool T linearly in time
    t0 = time.time()
    total_time = max(deadline - t0, 1.0)
    
    # Precompute list of elements for faster iteration
    elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        # Cooling
        frac = (time.time() - t0) / total_time
        frac = min(frac, 1.0)
        T = T_start * (T_end / T_start) ** frac
        
        # Choose neighborhood: 70% 1-move, 30% 2-opt
        if rng.random() < 0.7:
            # 1-move: add, remove, or swap
            r = rng.random()
            if r < 0.33:
                # Add
                x = rng.choice(elements)
                if (A >> (x-1)) & 1:
                    continue
                if not is_4ap_free_add(A, x):
                    continue
                new_A = A | elem_bit(x)
                delta = delta_3aps_add(A, x)
                new_score = cur_score + delta
            elif r < 0.66:
                # Remove
                # Pick a random element in A
                bits = A
                if bits == 0:
                    continue
                lsb = bits & (-bits)
                # Random selection: pick random bit position
                # Simpler: pick random x, check if in A
                x = rng.choice(elements)
                if not ((A >> (x-1)) & 1):
                    continue
                new_A = A & ~elem_bit(x)
                # Delta is negative: loss of 3-APs containing x
                # But we must account for 3-APs that had x and another element
                # Actually delta_3aps_add(A, x) counts 3-APs formed by adding x to A\x
                # So removing x loses exactly that many
                delta = -delta_3aps_add(new_A, x)
                new_score = cur_score + delta
            else:
                # Swap: remove x, add y
                x = rng.choice(elements)
                if not ((A >> (x-1)) & 1):
                    continue
                y = rng.choice(elements)
                if y == x or ((A >> (y-1)) & 1):
                    continue
                A_temp = A & ~elem_bit(x)
                if not is_4ap_free_add(A_temp, y):
                    continue
                new_A = A_temp | elem_bit(y)
                loss = delta_3aps_add(A, x)
                gain = delta_3aps_add(A_temp, y)
                delta = -loss + gain
                new_score = cur_score + delta
        else:
            # 2-opt: remove 2, add 2
            # Pick 2 distinct elements in A to remove
            x1 = rng.choice(elements)
            x2 = rng.choice(elements)
            if x1 == x2 or not ((A >> (x1-1)) & 1) or not ((A >> (x2-1)) & 1):
                continue
            A_temp = A & ~elem_bit(x1) & ~elem_bit(x2)
            # Pick 2 distinct elements not in A_temp to add
            y1 = rng.choice(elements)
            y2 = rng.choice(elements)
            if y1 == y2 or ((A_temp >> (y1-1)) & 1) or ((A_temp >> (y2-1)) & 1):
                continue
            if not is_4ap_free_add(A_temp, y1):
                continue
            A_temp2 = A_temp | elem_bit(y1)
            if not is_4ap_free_add(A_temp2, y2):
                continue
            new_A = A_temp2 | elem_bit(y2)
            # Compute delta
            # Loss from removing x1, x2 from A
            loss1 = delta_3aps_add(A, x1)
            # After removing x1, loss from removing x2
            A_after_x1 = A & ~elem_bit(x1)
            loss2 = delta_3aps_add(A_after_x1, x2)
            # Gain from adding y1, y2 to A_temp
            gain1 = delta_3aps_add(A_temp, y1)
            A_after_y1 = A_temp | elem_bit(y1)
            gain2 = delta_3aps_add(A_after_y1, y2)
            delta = -loss1 - loss2 + gain1 + gain2
            new_score = cur_score + delta
        
        # Metropolis acceptance
        if new_score >= cur_score or rng.random() < pow(2.718281828, (new_score - cur_score) / T):
            A = new_A
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_A = A
        
        step += 1
        # Periodic polish every 5000 steps to escape plateaus
        if step % 5000 == 0 and time.time() < deadline - 5:
            polished = local_search_1swap(A, deadline - 2)
            p_score = count_3aps(polished)
            if p_score > best_score:
                best_score = p_score
                best_A = polished
            A = polished
            cur_score = p_score
    
    return best_A

def mask_to_list(A_mask):
    return [i for i in range(1, N + 1) if (A_mask >> (i-1)) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + T_LIMIT - 5.0  # 5s safety margin
    
    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1
    
    # Generate multiple diverse initial candidates
    num_inits = 8
    for i in range(num_inits):
        if time.time() > deadline - 10:
            break
        init_mask = greedy_initial(i)
        # Quick polish
        polished = local_search_1swap(init_mask, deadline - 5)
        s = count_3aps(polished)
        if s > best_score:
            best_score = s
            best_mask = polished
    
    # Run SA from the best initial, and from a couple others
    sa_budget = deadline - time.time()
    if sa_budget > 10:
        # SA from best
        sa_result = simulated_annealing(best_mask, deadline - 2, rng)
        sa_score = count_3aps(sa_result)
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_result
    
    # If time remains, try SA from a different seed
    if time.time() < deadline - 15:
        alt_init = greedy_initial(99)
        alt_polished = local_search_1swap(alt_init, deadline - 10)
        alt_sa = simulated_annealing(alt_polished, deadline - 2, rng)
        alt_score = count_3aps(alt_sa)
        if alt_score > best_score:
            best_score = alt_score
            best_mask = alt_sa
    
    if best_mask == 0:
        best_mask = greedy_initial(0)
    
    return mask_to_list(best_mask)
