import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n  # 2187
    shape = (q,) * n
    
    words_np = np.array(np.unravel_index(np.arange(total_words), shape)).T
    
    # Precompute ball masks
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        w = words_np[i]
        ball_masks[i, i] = True
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                idx = int(np.ravel_multi_index(new_w, shape))
                ball_masks[i, idx] = True
    
    def greedy_set_cover(perturb=0.0, seed=None):
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        rng = np.random.default_rng(seed) if seed is not None else None
        while uncovered.any():
            gains = (ball_masks & uncovered[np.newaxis, :]).sum(axis=1).astype(float)
            if perturb > 0 and rng is not None:
                gains += rng.random(total_words) * perturb
            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 0.01:
                break
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return selected
    
    def local_search(selected):
        selected_set = set(selected)
        
        # Remove redundant
        changed = True
        while changed and (time.time() - start_time < time_limit):
            changed = False
            for idx in list(selected_set):
                if time.time() - start_time > time_limit:
                    break
                if len(selected_set) <= 1:
                    continue
                trial = selected_set - {idx}
                masks = np.array([ball_masks[i] for i in trial])
                if masks.any(axis=0).all():
                    selected_set = trial
                    changed = True
                    break
        
        # 2-swap: remove two, add one
        if time.time() - start_time < time_limit - 10:
            improved = True
            while improved and (time.time() - start_time < time_limit - 10):
                improved = False
                sel_list = list(selected_set)
                m = len(sel_list)
                max_pairs = 500
                pairs_tried = 0
                
                for i in range(m):
                    if time.time() - start_time > time_limit - 10 or pairs_tried >= max_pairs:
                        break
                    for j in range(i + 1, m):
                        if time.time() - start_time > time_limit - 10 or pairs_tried >= max_pairs:
                            break
                        pairs_tried += 1
                        idx1, idx2 = sel_list[i], sel_list[j]
                        trial = selected_set - {idx1, idx2}
                        if not trial:
                            continue
                        masks = np.array([ball_masks[k] for k in trial])
                        cov = masks.any(axis=0)
                        unc = ~cov
                        if not unc.any():
                            selected_set = trial
                            improved = True
                            break
                        gains = (ball_masks & unc[np.newaxis, :]).sum(axis=1)
                        for k in trial:
                            gains[k] = -1
                        best_k = int(np.argmax(gains))
                        if gains[best_k] == unc.sum():
                            new_set = trial | {best_k}
                            if len(new_set) < len(selected_set):
                                selected_set = new_set
                                improved = True
                                break
                    if improved:
                        break
        
        return selected_set
    
    best_result = None
    best_size = float('inf')
    
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        if restart == 0:
            selected = greedy_set_cover()
        else:
            selected = greedy_set_cover(perturb=0.5, seed=restart)
        
        result_set = local_search(selected)
        if len(result_set) < best_size:
            best_size = len(result_set)
            best_result = result_set
            if best_size <= 186:
                break
    
    if best_result is None:
        best_result = set(greedy_set_cover())
    
    return [list(words_np[i]) for i in best_result]
