"""Construct a linear triple system on 16 points minimizing independence number."""
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    rng = random.Random(12345)
    
    # Precompute all C(16,3) = 560 triples
    all_triples = []
    triple_masks = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    n_triples = len(all_triples)  # 560
    
    # Precompute compatibility: for each triple i, bitmask of compatible triples
    # Compatible means intersection size <= 1
    # 560 triples, so we need 560-bit masks -> use Python integers
    compat = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            inter = mi & mj
            cnt = inter.bit_count()
            if cnt <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Precompute all C(16,k) subset masks for each k
    subsets_masks_by_k = {}
    for k in range(1, 17):
        masks = []
        for S in combinations(range(16), k):
            m = 0
            for p in S:
                m |= (1 << p)
            masks.append(m)
        subsets_masks_by_k[k] = masks
    
    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def compute_alpha_v2(selected_indices):
        """Compute independence number: largest point set with no triple fully inside."""
        if not selected_indices:
            return 16
        
        tms = [triple_masks[i] for i in selected_indices]
        n_t = len(tms)
        
        for k in range(16, 0, -1):
            for S_mask in subsets_masks_by_k[k]:
                triple_free = True
                for tm in tms:
                    if (tm & S_mask) == tm:
                        triple_free = False
                        break
                if triple_free:
                    return k
        return 0
    
    # Simulated annealing
    # State: frozenset of triple indices
    # Energy: alpha (want to minimize)
    
    # Initial solution: greedy construction
    def greedy_initial():
        """Build a maximal linear system greedily."""
        available = (1 << n_triples) - 1
        selected = set()
        
        while available:
            bits = available.bit_count()
            if bits == 0:
                break
            idx_in_bits = rng.randrange(bits)
            tmp = available
            for _ in range(idx_in_bits):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            i = bit.bit_length() - 1
            
            selected.add(i)
            available &= compat[i]
            available &= ~(1 << i)
        
        return selected
    
    # Try multiple greedy starts and pick best
    best_initial = None
    best_initial_alpha = 16
    
    for _ in range(50):
        if time.time() > deadline - 40:
            break
        sol = greedy_initial()
        alpha = compute_alpha_v2(list(sol))
        if alpha < best_initial_alpha:
            best_initial_alpha = alpha
            best_initial = sol
    
    if best_initial is None:
        best_initial = greedy_initial()
        best_initial_alpha = compute_alpha_v2(list(best_initial))
    
    # Simulated annealing
    current = best_initial
    current_alpha = best_initial_alpha
    best = set(current)
    best_alpha = current_alpha
    
    T_start = 5.0
    T_end = 0.1
    total_iters = 50000
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(total_iters, 1))
    
    # Build current availability mask
    def build_available_mask(selected):
        """Bitmask of triples compatible with all in selected."""
        if not selected:
            return (1 << n_triples) - 1
        avail = (1 << n_triples) - 1
        for i in selected:
            avail &= compat[i]
        for i in selected:
            avail &= ~(1 << i)
        return avail
    
    # Fast move: add a triple
    def try_add(avail_mask):
        """Try to add a random triple from available."""
        bits = avail_mask.bit_count()
        if bits == 0:
            return None
        idx_in_bits = rng.randrange(bits)
        tmp = avail_mask
        for _ in range(idx_in_bits):
            tmp &= tmp - 1
        bit = tmp & (-tmp)
        return bit.bit_length() - 1
    
    # Run SA
    iteration = 0
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 5:
        iteration += 1
        
        if iteration < total_iters:
            T = T_start * (cooling ** iteration)
        else:
            T = T_end
        
        move_type = rng.random()
        
        if move_type < 0.3 and current:
            # Remove a triple
            i = rng.choice(tuple(current))
            new_state = current - {i}
            new_alpha = compute_alpha_v2(list(new_state))
            
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.7:
            # Add a triple
            avail = build_available_mask(current)
            i = try_add(avail)
            if i is None:
                no_improve += 1
                continue
            new_state = current | {i}
            new_alpha = compute_alpha_v2(list(new_state))
            
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Swap: remove one, add one
            if not current:
                no_improve += 1
                continue
            i_remove = rng.choice(tuple(current))
            remaining = current - {i_remove}
            avail = build_available_mask(remaining)
            i_add = try_add(avail)
            if i_add is None:
                no_improve += 1
                continue
            new_state = remaining | {i_add}
            new_alpha = compute_alpha_v2(list(new_state))
            
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            sol = greedy_initial()
            alpha = compute_alpha_v2(list(sol))
            if alpha < best_alpha:
                best_alpha = alpha
                best = set(sol)
            current = sol
            current_alpha = alpha
            no_improve = 0
            T = T_start * 0.5
    
    # Final polish: try to add more triples to best
    if time.time() < deadline - 3:
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            avail = build_available_mask(best)
            i = try_add(avail)
            if i is None:
                break
            candidate = best | {i}
            alpha = compute_alpha_v2(list(candidate))
            if alpha <= best_alpha:
                best = candidate
                best_alpha = alpha
    
    return [list(all_triples[i]) for i in sorted(best)]

# EVOLVE-BLOCK-END
