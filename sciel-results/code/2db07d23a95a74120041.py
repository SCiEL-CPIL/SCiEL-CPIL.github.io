"""Maximize 3-APs in subset of [1,256] with no 4-APs, using bitmask + fast local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 256

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Precompute 4-APs: for each x, bitmask of 4-APs it's in
    # We need to check if adding x creates a 4-AP: all 3 other members present.
    # Precompute for each x: list of (other_mask, ap_id) where other_mask is the
    # bitmask of the other 3 elements in that 4-AP.
    # To check: does (selected & other_mask) == other_mask?
    four_ap_info = [[] for _ in range(N)]  # index 0..255 for elements 1..256
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # 4-AP: a, b, c, e
            # For element a: others are b, c, e
            four_ap_info[a - 1].append(((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))))
            four_ap_info[b - 1].append(((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))))
            four_ap_info[c - 1].append(((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))))
            four_ap_info[e - 1].append(((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))))

    # Precompute 3-AP contributions for fast delta counting
    # For each element x, precompute:
    #   first_aps[x]: list of (y, z) where x, y, z form 3-AP (x is first)
    #   mid_aps[x]: list of (y, z) where y, x, z form 3-AP (x is middle)  
    #   last_aps[x]: list of (y, z) where y, z, x form 3-AP (x is last)
    # When adding x to selected, new 3-APs = count of pairs where both others in selected
    
    # For fast delta: when adding x, count how many 3-APs involving x are completed
    # We'll precompute for each x a list of bitmasks (pairs of other elements)
    # Actually, let's precompute for each x:
    #   pair_masks[x]: list of (mask1, mask2) where mask1|mask2 represents the two other elements
    # New 3-APs when adding x = sum over pairs of (selected & mask1 & mask2) != 0
    # Better: for each x, list of (a_mask, b_mask) for the two other positions
    
    # Even better: precompute for each x the list of "partner pairs" as bitmasks
    # When adding x: new_3aps = sum(1 for m in partner_masks[x] if (selected & m) == m)
    # where m is the bitmask of the 2 other elements needed

    partner_masks = [[] for _ in range(N)]
    for x in range(1, N + 1):
        # x is first: x, x+d, x+2d
        for d in range(1, N - x + 1):
            y = x + d
            z = x + 2 * d
            if z <= N:
                partner_masks[x - 1].append((1 << (y - 1)) | (1 << (z - 1)))
        # x is middle: x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            y = x - d
            z = x + d
            partner_masks[x - 1].append((1 << (y - 1)) | (1 << (z - 1)))
        # x is last: x-2d, x-d, x
        for d in range(1, (x + 1) // 2):
            y = x - 2 * d
            z = x - d
            if y >= 1:
                partner_masks[x - 1].append((1 << (y - 1)) | (1 << (z - 1)))

    def would_create_4ap(x, selected_mask):
        """Check if adding element x (1-indexed) to selected_mask creates a 4-AP."""
        idx = x - 1
        for others in four_ap_info[idx]:
            if (selected_mask & others) == others:
                return True
        return False

    def delta_3aps_add(x, selected_mask):
        """Count new 3-APs created by adding element x."""
        idx = x - 1
        count = 0
        for pm in partner_masks[idx]:
            if (selected_mask & pm) == pm:
                count += 1
        return count

    def total_3aps(selected_mask):
        """Count all 3-APs in the set represented by selected_mask."""
        count = 0
        # For each element in the set, count 3-APs where it's the first element
        # and the other two are also in the set
        # To avoid double counting, only count where x is the first (smallest) element
        for x in range(1, N + 1):
            if not (selected_mask >> (x - 1)) & 1:
                continue
            for d in range(1, N - x + 1):
                y = x + d
                z = x + 2 * d
                if z > N:
                    break
                if (selected_mask >> (y - 1)) & 1 and (selected_mask >> (z - 1)) & 1:
                    count += 1
        return count

    def greedy_random(rng, strategy="random"):
        """Greedy construction with given strategy."""
        selected_mask = 0
        if strategy == "random":
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif strategy == "gain":
            order = list(range(1, N + 1))
        elif strategy == "natural":
            order = list(range(1, N + 1))
        elif strategy == "reverse":
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)

        if strategy == "gain":
            remaining = set(order)
            while remaining and time_left() > 2.0:
                best_x = None
                best_gain = -1
                for x in remaining:
                    if would_create_4ap(x, selected_mask):
                        continue
                    gain = delta_3aps_add(x, selected_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                selected_mask |= (1 << (best_x - 1))
                remaining.discard(best_x)
        else:
            for x in order:
                if time_left() < 2.0:
                    break
                if not would_create_4ap(x, selected_mask):
                    selected_mask |= (1 << (x - 1))
        return selected_mask

    def local_search(selected_mask, max_iters=200):
        """Local search with add, remove, and swap moves."""
        current_score = total_3aps(selected_mask)
        best_score = current_score
        best_mask = selected_mask
        no_improve_count = 0

        for it in range(max_iters):
            if time_left() < 3.0:
                break

            improved = False

            # Try adding elements (single add)
            for x in range(1, N + 1):
                if (selected_mask >> (x - 1)) & 1:
                    continue
                if would_create_4ap(x, selected_mask):
                    continue
                gain = delta_3aps_add(x, selected_mask)
                if gain > 0:
                    new_mask = selected_mask | (1 << (x - 1))
                    new_score = current_score + gain
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                    selected_mask = new_mask
                    current_score = new_score
                    improved = True

            # Try swaps: remove x, add y
            if not improved:
                for x in range(1, N + 1):
                    if time_left() < 3.0:
                        break
                    if not (selected_mask >> (x - 1)) & 1:
                        continue
                    temp_mask = selected_mask & ~(1 << (x - 1))
                    # Score change from removing x: negative of 3-APs involving x
                    # We need to compute how many 3-APs x is part of
                    # For simplicity, compute new score directly for candidates
                    for y in range(1, N + 1):
                        if time_left() < 3.0:
                            break
                        if (temp_mask >> (y - 1)) & 1:
                            continue
                        if would_create_4ap(y, temp_mask):
                            continue
                        new_mask = temp_mask | (1 << (y - 1))
                        new_score = total_3aps(new_mask)
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask
                        if new_score > current_score:
                            selected_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break

            # Try remove + add two (2-for-1 or 1-for-2 moves)
            if not improved:
                # Try removing one element and adding two
                for x in range(1, N + 1):
                    if time_left() < 3.0:
                        break
                    if not (selected_mask >> (x - 1)) & 1:
                        continue
                    temp_mask = selected_mask & ~(1 << (x - 1))
                    for y in range(1, N + 1):
                        if time_left() < 3.0:
                            break
                        if (temp_mask >> (y - 1)) & 1:
                            continue
                        if would_create_4ap(y, temp_mask):
                            continue
                        temp2 = temp_mask | (1 << (y - 1))
                        for z in range(1, N + 1):
                            if time_left() < 3.0:
                                break
                            if (temp2 >> (z - 1)) & 1:
                                continue
                            if would_create_4ap(z, temp2):
                                continue
                            new_mask = temp2 | (1 << (z - 1))
                            new_score = total_3aps(new_mask)
                            if new_score > best_score:
                                best_score = new_score
                                best_mask = new_mask
                            if new_score > current_score:
                                selected_mask = new_mask
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            else:
                no_improve_count = 0

        return best_mask, best_score

    def simulated_annealing(selected_mask, iterations=5000):
        """Simulated annealing with swap and add/remove moves."""
        current_mask = selected_mask
        current_score = total_3aps(current_mask)
        best_mask = current_mask
        best_score = current_score
        rng = random.Random(int(time.time() * 1000) % 100000)
        T = 10.0
        cooling = 0.995

        for it in range(iterations):
            if time_left() < 3.0:
                break

            # Generate a random move
            move_type = rng.random()
            if move_type < 0.4:
                # Swap: remove random x, add random y
                x = rng.randint(1, N)
                if not (current_mask >> (x - 1)) & 1:
                    continue
                y = rng.randint(1, N)
                if (current_mask >> (y - 1)) & 1:
                    continue
                temp = current_mask & ~(1 << (x - 1))
                if would_create_4ap(y, temp):
                    continue
                new_mask = temp | (1 << (y - 1))
            elif move_type < 0.7:
                # Add
                y = rng.randint(1, N)
                if (current_mask >> (y - 1)) & 1:
                    continue
                if would_create_4ap(y, current_mask):
                    continue
                new_mask = current_mask | (1 << (y - 1))
            else:
                # Remove
                x = rng.randint(1, N)
                if not (current_mask >> (x - 1)) & 1:
                    continue
                new_mask = current_mask & ~(1 << (x - 1))

            new_score = total_3aps(new_mask)
            delta = new_score - current_score
            if delta >= 0 or rng.random() < np.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
            T *= cooling

        return best_mask, best_score

    # Main: run multiple strategies and keep best
    best_mask = 0
    best_score = -1

    # Run various greedy strategies
    strategies = [
        ("natural", 0),
        ("reverse", 0),
        ("random", 1),
        ("random", 2),
        ("random", 3),
        ("gain", 0),
    ]

    for strat, seed in strategies:
        if time_left() < 5.0:
            break
        rng = random.Random(seed + 12345)
        mask = greedy_random(rng, strat)
        score = total_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Local search on best
    if time_left() > 5.0:
        best_mask, best_score = local_search(best_mask, max_iters=100)

    # Try local search on a few other starting points
    for seed in [10, 20, 30]:
        if time_left() < 10.0:
            break
        rng = random.Random(seed)
        mask = greedy_random(rng, "random")
        score = total_3aps(mask)
        if score >= best_score - 20:  # only if competitive
            mask, score = local_search(mask, max_iters=80)
            if score > best_score:
                best_score = score
                best_mask = mask

    # Simulated annealing on best
    if time_left() > 10.0:
        best_mask, best_score = simulated_annealing(best_mask, iterations=3000)

    # Final local search polish
    if time_left() > 5.0:
        best_mask, best_score = local_search(best_mask, max_iters=50)

    # Convert mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    if not result:
        result = [1]
    return result
