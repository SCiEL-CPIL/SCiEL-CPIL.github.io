"""Simulated Annealing with WalkSAT moves for 4-AP-free set construction."""
import time
import random
import math

N = 256
TIME_LIMIT = 88.0  # Safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Precompute all 3-APs and 4-APs (0-indexed)
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element, store the OTHER elements in each 3-AP and 4-AP it participates in
    # This allows fast incremental updates
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # Weights for penalty
    P3 = 1.0
    P4 = 1000.0 # Large penalty for 4-APs

    def count_3aps_bitmask(S):
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def count_4aps_bitmask(S):
        cnt = 0
        for a, b, c, d in four_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def get_energy(S):
        """Returns (neg_3ap_count, 4ap_count). We want to minimize this tuple."""
        n3 = count_3aps_bitmask(S)
        n4 = count_4aps_bitmask(S)
        return -n3, n4

    def delta_energy_add(S, x):
        """Calculate change in (neg_3, n4) when adding x to S."""
        d_neg3 = 0
        d_n4 = 0
        # Check 3-APs
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                d_neg3 -= 1 # Energy decreases (more 3-APs is better, so neg count goes down)
        
        # Check 4-APs
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                d_n4 += 1
        return d_neg3, d_n4

    def delta_energy_remove(S, x):
        """Calculate change in (neg_3, n4) when removing x from S."""
        d_neg3 = 0
        d_n4 = 0
        # Check 3-APs
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                d_neg3 += 1 # Energy increases (fewer 3-APs)
        
        # Check 4-APs
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                d_n4 -= 1
        return d_neg3, d_n4

    def local_search_1swap(S, max_iters=2000):
        """Fast 1-swap local search to polish the result."""
        current_S = S
        cur_n3 = count_3aps_bitmask(current_S)
        cur_n4 = count_4aps_bitmask(current_S)
        
        if cur_n4 > 0:
            # Try to remove elements to kill 4-APs
            # Simple heuristic: remove elements in 4-APs with low 3-AP contribution
            # This is a rough cleanup, SA should have minimized n4 already
            pass

        for _ in range(max_iters):
            improved = False
            # Try adding
            for x in range(N):
                if not ((current_S >> x) & 1):
                    d_neg3, d_n4 = delta_energy_add(current_S, x)
                    if d_n4 == 0 and d_neg3 < 0:
                        current_S |= (1 << x)
                        cur_n3 += -d_neg3
                        improved = True
                        break
            if improved:
                continue
            
            # Try removing (only if it helps 3-APs? No, removing always hurts 3-APs unless we add something else)
            # So 1-swap is just add if safe and beneficial.
            # If no add improves, try swaps?
            # Let's do a quick swap search
            best_swap_gain = 0
            best_swap = None
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
            
            # Sample to keep it fast
            if len(members) > 50:
                sample_m = rng.sample(members, 50)
            else:
                sample_m = members
            if len(non_members) > 50:
                sample_nm = rng.sample(non_members, 50)
            else:
                sample_nm = non_members
                
            for x_rm in sample_m:
                S_tmp = current_S & ~(1 << x_rm)
                # Calculate loss of removing x_rm
                d_neg3_rm, d_n4_rm = delta_energy_remove(current_S, x_rm)
                for x_add in sample_nm:
                    if not ((S_tmp >> x_add) & 1):
                        d_neg3_add, d_n4_add = delta_energy_add(S_tmp, x_add)
                        total_d_neg3 = d_neg3_rm + d_neg3_add
                        total_d_n4 = d_n4_rm + d_n4_add
                        if total_d_n4 == 0 and total_d_neg3 < 0:
                            gain = -total_d_neg3
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap = (x_rm, x_add)
                                
            if best_swap:
                x_rm, x_add = best_swap
                current_S = (current_S & ~(1 << x_rm)) | (1 << x_add)
                cur_n3 += best_swap_gain
                improved = True
                continue
                
            if not improved:
                break
                
        return current_S, count_3aps_bitmask(current_S)

    # Initialize with a random set of size ~100
    # Or better, a greedy 4-AP-free set
    def greedy_construct():
        S = 0
        # Order by 3-AP weight descending
        weights = [len(elem_to_3aps[x]) for x in range(N)]
        order = sorted(range(N), key=lambda x: -weights[x])
        for x in order:
            # Check if adding x creates a 4-AP
            safe = True
            for others in elem_to_4aps[x]:
                ok = True
                for y in others:
                    if not ((S >> y) & 1):
                        ok = False
                        break
                if ok:
                    safe = False
                    break
            if safe:
                S |= (1 << x)
        return S

    S_init = greedy_construct()
    cur_S = S_init
    cur_n3, cur_n4 = get_energy(cur_S)
    best_S = cur_S
    best_n3 = cur_n3
    
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995
    T = T_start
    steps = 0
    max_steps = 500000 # Will be limited by time
    
    while time.time() - start_time < TIME_LIMIT - 2:
        # Generate move
        if rng.random() < 0.2 and cur_n4 > 0:
            # WalkSAT: pick a random 4-AP and flip a random element
            # We need to find a 4-AP in current S
            # This is expensive if done from scratch every time.
            # Approximation: just pick a random element and flip it, but bias towards elements in 4-APs?
            # For simplicity and speed, let's stick to random flips but maybe increase probability of flipping elements that are in many 4-APs?
            # Actually, let's just do random flip for now, SA is powerful enough.
            # Or, find one 4-AP quickly.
            # Let's just do random flip.
            x = rng.randint(0, N - 1)
        else:
            x = rng.randint(0, N - 1)
            
        if (cur_S >> x) & 1:
            # Remove x
            d_neg3, d_n4 = delta_energy_remove(cur_S, x)
            new_S = cur_S & ~(1 << x)
        else:
            # Add x
            d_neg3, d_n4 = delta_energy_add(cur_S, x)
            new_S = cur_S | (1 << x)
            
        # Calculate energy change
        # Energy = -n3 + P4 * n4
        # Delta E = d(-n3) + P4 * d(n4)
        dE = d_neg3 + P4 * d_n4
        
        if dE < 0 or (T > 0 and rng.random() < math.exp(-dE / T)):
            cur_S = new_S
            cur_n3 += -d_neg3
            cur_n4 += d_n4
            
            if cur_n4 == 0 and cur_n3 > best_n3:
                best_n3 = cur_n3
                best_S = cur_S
        
        T *= alpha
        steps += 1
        
        # Periodic local search
        if steps % 1000 == 0 and time.time() - start_time < TIME_LIMIT - 5:
            S_ls, score_ls = local_search_1swap(cur_S, max_iters=100)
            if score_ls > best_n3:
                best_n3 = score_ls
                best_S = S_ls
            # Reset current state to the improved one if it's better or similar
            if score_ls >= cur_n3 - 5:
                cur_S = S_ls
                cur_n3 = score_ls
                cur_n4 = count_4aps_bitmask(cur_S)

    # Final polish
    if time.time() - start_time < TIME_LIMIT - 1:
        S_final, score_final = local_search_1swap(best_S, max_iters=500)
        if score_final > best_n3:
            best_n3 = score_final
            best_S = S_final

    # Verify no 4-APs
    final_n4 = count_4aps_bitmask(best_S)
    if final_n4 > 0:
        # Force remove elements to fix 4-APs
        # This is a repair step
        for _ in range(100):
            if count_4aps_bitmask(best_S) == 0:
                break
            # Find a 4-AP
            found = False
            for a, b, c, d in four_aps:
                if (best_S >> a) & 1 and (best_S >> b) & 1 and (best_S >> c) & 1 and (best_S >> d) & 1:
                    # Remove one with lowest 3-AP weight
                    # Approximate weight: number of 3-APs it's in
                    # Just remove 'a' for simplicity in this emergency case
                    best_S &= ~(1 << a)
                    found = True
                    break
            if not found:
                break

    # Convert to list
    result = [x + 1 for x in range(N) if (best_S >> x) & 1]
    
    if not result:
        # Fallback
        result = [1, 2, 4, 8, 16, 32, 64, 128] # Powers of 2 have no 3-APs? No, 1,2,4 is not 3-AP. 1,3,5 is.
        # Just return a valid small set
        S_fb = 0
        for x in range(N):
             # Greedy safe add
             safe = True
             for others in elem_to_4aps[x]:
                 ok = True
                 for y in others:
                     if not ((S_fb >> y) & 1):
                         ok = False
                         break
                 if ok:
                     safe = False
                     break
             if safe:
                 S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result
