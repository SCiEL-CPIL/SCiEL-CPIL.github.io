import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 256

    # Precompute 4-APs and 3-APs as bit masks
    masks_4ap = []
    elem_to_4ap_masks = {i: [] for i in range(N + 1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d)) | (1 << (a + 3 * d))
            masks_4ap.append(mask)
            for x in (a, a + d, a + 2 * d, a + 3 * d):
                elem_to_4ap_masks[x].append(mask)

    masks_3ap = []
    elem_to_3ap_masks = {i: [] for i in range(N + 1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d))
            masks_3ap.append(mask)
            for x in (a, a + d, a + 2 * d):
                elem_to_3ap_masks[x].append(mask)

    def is_valid_add(A, x):
        if A & (1 << x):
            return False
        for mask in elem_to_4ap_masks[x]:
            if (A & mask).bit_count() == 3:
                return False
        return True

    def is_valid_swap(A, x, y):
        for mask in elem_to_4ap_masks[y]:
            if mask & (1 << x):
                continue
            other_bits = mask ^ (1 << y)
            if (A & other_bits) == other_bits:
                return False
        return True

    def count_3aps(A):
        count = 0
        for mask in masks_3ap:
            if (A & mask) == mask:
                count += 1
        return count

    def gain_add(A, x):
        gain = 0
        for mask in elem_to_3ap_masks[x]:
            other_bits = mask ^ (1 << x)
            if (A & other_bits) == other_bits:
                gain += 1
        return gain

    def loss_remove(A, x):
        loss = 0
        for mask in elem_to_3ap_masks[x]:
            other_bits = mask ^ (1 << x)
            if (A & other_bits) == other_bits:
                loss += 1
        return loss

    def delta_swap(A, x, y):
        loss = 0
        for mask in elem_to_3ap_masks[x]:
            other_bits = mask ^ (1 << x)
            if (A & other_bits) == other_bits:
                loss += 1
        gain = 0
        for mask in elem_to_3ap_masks[y]:
            other_bits = mask ^ (1 << y)
            if (A & other_bits) == other_bits:
                if not (other_bits & (1 << x)):
                    gain += 1
        return gain - loss

    # === Phase 1: Weighted Greedy Construction ===
    # Score each element by potential 3-AP contribution, pick highest-scoring valid element iteratively
    A = 0
    current_score = 0
    selected = set()
    
    # Compute initial "potential" for each element: number of 3-APs it participates in
    elem_potential = {i: len(elem_to_3ap_masks[i]) for i in range(1, N + 1)}
    
    candidates = list(range(1, N + 1))
    
    while candidates and time.time() < deadline - 10.0:
        best_elem = -1
        best_score_val = -1
        for x in candidates:
            if A & (1 << x):
                continue
            if not is_valid_add(A, x):
                continue
            g = gain_add(A, x)
            # Dynamic score: immediate gain + a fraction of remaining potential
            # This balances short-term gain with long-term density
            score_val = g + 0.1 * elem_potential[x]
            if score_val > best_score_val or (score_val == best_score_val and x > best_elem):
                best_score_val = score_val
                best_elem = x
        
        if best_elem == -1:
            break
        
        A |= (1 << best_elem)
        selected.add(best_elem)
        current_score += gain_add(A ^ (1 << best_elem), best_elem)  # gain before adding
        candidates.remove(best_elem)
    
    best_A = A
    best_score = count_3aps(A)
    
    # === Phase 2: Focused Local Search with proper data structures ===
    def local_search(A, score, time_limit):
        # Maintain sorted lists for efficiency
        active = sorted([i for i in range(1, N + 1) if A & (1 << i)])
        inactive = sorted([i for i in range(1, N + 1) if not (A & (1 << i))])
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adds first (cheap, can only help)
            for x in list(inactive):
                if is_valid_add(A, x):
                    g = gain_add(A, x)
                    if g > 0:
                        A |= (1 << x)
                        score += g
                        active.append(x)
                        active.sort()
                        inactive.remove(x)
                        improved = True
                        break
            if improved:
                continue
            
            # Try swaps: use deterministic scan with early termination
            # Only try swaps where y has high potential
            swap_improved = False
            for y in inactive:
                if not inactive:
                    break
                y_potential = len(elem_to_3ap_masks[y])
                if y_potential < 2:
                    continue
                for x in active:
                    if x > N:
                        continue
                    if not is_valid_swap(A, x, y):
                        continue
                    d = delta_swap(A, x, y)
                    if d > 0:
                        A ^= (1 << x)
                        A ^= (1 << y)
                        score += d
                        active.remove(x)
                        active.append(y)
                        active.sort()
                        inactive.remove(y)
                        inactive.append(x)
                        inactive.sort()
                        swap_improved = True
                        improved = True
                        break
                if swap_improved:
                    break
            
            if not improved and time.time() < time_limit - 1.0:
                # Randomized swaps for escape
                no_improve = 0
                max_tries = 200
                while no_improve < max_tries and time.time() < time_limit:
                    if not active or not inactive:
                        break
                    x = random.choice(active)
                    y = random.choice(inactive)
                    if is_valid_swap(A, x, y):
                        d = delta_swap(A, x, y)
                        if d > 0:
                            A ^= (1 << x)
                            A ^= (1 << y)
                            score += d
                            active.remove(x)
                            active.append(y)
                            active.sort()
                            inactive.remove(y)
                            inactive.append(x)
                            inactive.sort()
                            no_improve = 0
                            improved = True
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
        
        return A, score

    # Run local search
    A_ls, score_ls = local_search(A, best_score, deadline - 15.0)
    if score_ls > best_score:
        best_score = score_ls
        best_A = A_ls
    
    # === Phase 3: Multiple restarts with perturbation ===
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Perturb the best solution
        A_pert = best_A
        k = min(8, num_restarts * 2)  # Increasing perturbation
        for _ in range(k):
            if time.time() > deadline - 5.0:
                break
            active = [i for i in range(1, N + 1) if A_pert & (1 << i)]
            inactive = [i for i in range(1, N + 1) if not (A_pert & (1 << i))]
            if not active or not inactive:
                break
            r = random.random()
            if r < 0.6:
                x = random.choice(active)
                y = random.choice(inactive)
                if is_valid_swap(A_pert, x, y):
                    A_pert ^= (1 << x)
                    A_pert ^= (1 << y)
            elif r < 0.8:
                y = random.choice(inactive)
                if is_valid_add(A_pert, y):
                    A_pert |= (1 << y)
            else:
                x = random.choice(active)
                A_pert ^= (1 << x)
        
        score_pert = count_3aps(A_pert)
        
        # Local search on perturbed
        A_ls2, score_ls2 = local_search(A_pert, score_pert, deadline - 5.0)
        
        if score_ls2 > best_score:
            best_score = score_ls2
            best_A = A_ls2
    
    # Final polish
    A_final, score_final = local_search(best_A, best_score, deadline)
    if score_final > best_score:
        best_A = A_final
        best_score = score_final

    res = [i for i in range(1, N + 1) if best_A & (1 << i)]
    
    if not res:
        return [1]
        
    return res

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
