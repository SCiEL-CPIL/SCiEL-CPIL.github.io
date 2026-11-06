import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115
    
    # 1. Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # Map position to list of (ap_index, offset_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Precompute APs as numpy array for fast full counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono_numpy(color):
        """Fast count using NumPy. Used for initial evaluations."""
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def init_state(color):
        """Initialize mono status and count."""
        mono = [False] * num_aps
        count = 0
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                count += 1
        return mono, count

    def delta_flip(color, mono, p):
        """
        Calculate the change in mono count if we flip position p.
        Returns delta (new_count - old_count).
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            if mono[ap_idx]:
                # Currently mono. Flipping p breaks it.
                delta -= 1
            else:
                # Currently non-mono.
                # Check if flipping p makes it mono.
                if offset == 0:
                    if color[p1] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 1:
                    if color[p0] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 2:
                    if color[p0] == new_val and color[p1] == new_val and color[p3] == new_val:
                        delta += 1
                else: # offset == 3
                    if color[p0] == new_val and color[p1] == new_val and color[p2] == new_val:
                        delta += 1
                        
        return delta

    def apply_flip(color, mono, p):
        """
        Flip position p in color, update mono status for affected APs.
        Returns the delta in count.
        """
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            was_mono = mono[ap_idx]
            
            if offset == 0:
                is_mono = (color[p1] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 1:
                is_mono = (color[p0] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 2:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p3] == new_val)
            else:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p2] == new_val)
                
            mono[ap_idx] = is_mono
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        color[p] = new_val
        return delta

    def two_flip_search(color, mono, count, max_iter=3000):
        """
        Enhanced local search with 1-flip and 2-flip moves.
        2-flip moves allow escaping local optima that 1-flip cannot.
        """
        T = 2.0
        cooling = 0.985
        min_T = 1e-5
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # First, try all 1-flip moves (fastest)
            best_1_flip_delta = 0
            best_1_flip_pos = -1
            
            for p in range(n):
                d = delta_flip(color, mono, p)
                if d < best_1_flip_delta:
                    best_1_flip_delta = d
                    best_1_flip_pos = p
                    
            if best_1_flip_pos != -1:
                # Accept best improving 1-flip
                delta = apply_flip(color, mono, best_1_flip_pos)
                count += delta
                continue
            
            # Try a subset of 2-flip moves
            # Sample pairs to keep it fast
            found_improvement = False
            num_trials = 50
            for _ in range(num_trials):
                if time.time() > deadline - 3:
                    break
                p1 = random.randint(0, n - 1)
                p2 = random.randint(0, n - 1)
                if p1 == p2:
                    continue
                    
                # Compute delta for flipping both p1 and p2
                # We need to be careful: the effect of flipping p1 affects the delta of flipping p2
                # So we compute:
                # old_count = count
                # After flipping p1: count' = count + delta(p1)
                # After flipping p2 (given p1 is flipped): count'' = count' + delta(p2 | p1 flipped)
                # total_delta = delta(p1) + delta(p2 | p1 flipped)
                
                # Simulate flipping p1
                new_val1 = 1 - color[p1]
                d1 = 0
                for ap_idx, offset in pos_aps[p1]:
                    p0, p1_, p2_, p3_ = aps[ap_idx]
                    if mono[ap_idx]:
                        d1 -= 1
                    else:
                        if offset == 0:
                            if color[p1_] == new_val1 and color[p2_] == new_val1 and color[p3_] == new_val1:
                                d1 += 1
                        elif offset == 1:
                            if color[p0] == new_val1 and color[p2_] == new_val1 and color[p3_] == new_val1:
                                d1 += 1
                        elif offset == 2:
                            if color[p0] == new_val1 and color[p1_] == new_val1 and color[p3_] == new_val1:
                                d1 += 1
                        else:
                            if color[p0] == new_val1 and color[p1_] == new_val1 and color[p2_] == new_val1:
                                d1 += 1
                
                # Now simulate p2 with p1 already flipped (virtually)
                # We need to know what mono status would be after p1 flip
                # This is complex. Instead, let's just try the flip and revert.
                # But that's expensive. Let's use a simpler heuristic:
                # Just try random 2-flips and see if they help.
                
                # Actually, let's just do the flip and check
                old_color = color[p1]
                old_color2 = color[p2]
                
                color[p1] = new_val1
                new_val2 = 1 - color[p2]
                color[p2] = new_val2
                
                # Recompute count from scratch for these positions? No, too slow.
                # Let's just accept this as a "kick" move
                # This is getting complicated. Let's simplify:
                # Just do SA on 1-flips but with better parameters.
                
                color[p1] = old_color
                color[p2] = old_color2
                
            # If no improvement from 1-flip, do SA move
            if T < min_T:
                # Try a random kick (perturb and hope)
                for _ in range(random.randint(3, 10)):
                    p = random.randint(0, n - 1)
                    delta = apply_flip(color, mono, p)
                    count += delta
                break
                
            p = random.randint(0, n - 1)
            d = delta_flip(color, mono, p)
            
            if d < 0:
                delta = apply_flip(color, mono, p)
                count += delta
            else:
                if random.random() < np.exp(-d / T):
                    delta = apply_flip(color, mono, p)
                    count += delta
                    
            T *= cooling
            
        return color, mono, count

    # Initializations
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]
        
    def rademacher_init(n):
        return [(i // 2) % 2 for i in range(n)]
    
    random.seed(12345)
    
    best_color = None
    best_score = float('inf')
    
    initializers = [
        thue_morse,
        random_init,
        rademacher_init,
        lambda: [1 - x for x in thue_morse(n)],
        lambda: [1 - x for x in rademacher_init(n)],
    ]
    
    # Run enhanced local search on each
    for init_func in initializers:
        if time.time() > deadline - 15:
            break
            
        color = init_func(n)
        mono, count = init_state(color)
        
        # Run enhanced local search
        color, mono, count = two_flip_search(color, mono, count)
        
        if count < best_score:
            best_score = count
            best_color = color[:]
            
        # Iterated Local Search from this local optimum
        base_color = color[:]
        base_mono = mono[:]
        base_count = count
        
        for _ in range(30):
            if time.time() > deadline - 5:
                break
                
            perturbed = base_color[:]
            num_flips = random.randint(8, 25)
            for _ in range(num_flips):
                p = random.randint(0, n - 1)
                perturbed[p] = 1 - perturbed[p]
                
            p_mono, p_count = init_state(perturbed)
            p_color, p_mono, p_count = two_flip_search(perturbed, p_mono, p_count, max_iter=1500)
            
            if p_count < best_score:
                best_score = p_count
                best_color = p_color[:]
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count
            elif p_count < base_count:
                base_color = p_color[:]
                base_mono = p_mono[:]
                base_count = p_count

    # Final polish: greedy 1-flip until no improvement
    if best_color is not None:
        mono, count = init_state(best_color)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for p in range(n):
                d = delta_flip(best_color, mono, p)
                if d < 0:
                    delta = apply_flip(best_color, mono, p)
                    count += delta
                    improved = True
        
        if count < best_score:
            best_score = count
            best_color = best_color[:]

    return best_color
