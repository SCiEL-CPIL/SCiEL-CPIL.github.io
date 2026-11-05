import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0

    space_size = q ** n

    # Precompute neighbors (ball of radius 1) for each word
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    neighbors = np.empty(space_size, dtype=object)
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n - 1 - pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n - 1 - pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)

    # cover_map[w] = list of codewords whose ball covers w
    cover_map = np.empty(space_size, dtype=object)
    for w in range(space_size):
        cover_map[w] = []
    for c in range(space_size):
        for w in neighbors[c]:
            cover_map[w].append(c)
    for w in range(space_size):
        cover_map[w] = np.array(cover_map[w], dtype=np.int32)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_cover(uncovered_mask):
        """Greedy: repeatedly pick codeword covering most uncovered words."""
        code = []
        while True:
            if time.time() - t_start > deadline - 10:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break

            score = np.zeros(space_size, dtype=np.int32)
            for w in uncovered_indices:
                for c in cover_map[w]:
                    score[c] += 1

            best_c = np.argmax(score)
            if score[best_c] == 0:
                break

            code.append(best_c)
            uncovered_mask[neighbors[best_c]] = False

        return code

    def local_search(code):
        """Try to remove redundant codewords."""
        code_set = set(code)
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code:
            cov[neighbors[c]] += 1

        improved = True
        while improved:
            if time.time() - t_start > deadline - 10:
                break
            improved = False
            random.shuffle(code)
            for i in range(len(code)):
                c = code[i]
                ball = neighbors[c]
                if np.any(cov[ball] == 1):
                    continue
                cov[ball] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break

        return code

    def lagrangian_heuristic():
        """
        Lagrangian relaxation:
        Minimize sum(x_c) subject to for all w: sum_{c in cover(w)} x_c >= 1
        Lagrangian: L(x, y) = sum_c x_c + sum_w y_w * (1 - sum_{c in cover(w)} x_c)
                   = sum_w y_w + sum_c x_c * (1 - sum_{w in neighbors(c)} y_w)
        
        For fixed y, the optimal x_c is 1 if (1 - sum_{w in neighbors(c)} y_w) < 0, i.e., sum_{w in neighbors(c)} y_w > 1.
        Then we check feasibility. If not feasible, we do a greedy repair.
        Subgradient update: y_w += alpha * (1 - sum_{c in S} x_c * [w in neighbors(c)])
        """
        # Initialize y
        y = np.ones(space_size, dtype=np.float64) * 0.5
        best_code = None
        best_size = float('inf')
        
        # Number of iterations based on time
        max_iter = 200
        for it in range(max_iter):
            if time.time() - t_start > deadline - 15:
                break
                
            # Compute sum_y_for_each_c: sum_{w in neighbors(c)} y_w
            sum_y = np.zeros(space_size, dtype=np.float64)
            for c in range(space_size):
                sum_y[c] = np.sum(y[neighbors[c]])
            
            # Select centers where sum_y > 1 (reduced cost negative)
            selected_mask = sum_y > 1.0
            selected = np.where(selected_mask)[0].tolist()
            
            if len(selected) == 0:
                # Fallback: pick all? Or greedy from scratch
                break
            
            # Check coverage
            cov = np.zeros(space_size, dtype=np.int32)
            for c in selected:
                cov[neighbors[c]] += 1
            
            if np.all(cov >= 1):
                # Feasible!
                code = selected[:]
                code = local_search(code)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                # Infeasible: repair with greedy
                uncovered_mask = (cov == 0)
                added = greedy_cover(uncovered_mask)
                code = selected + added
                # Remove duplicates
                code = list(set(code))
                code = local_search(code)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            
            # Subgradient update
            # Gradient: g_w = 1 - sum_{c in selected} [w in neighbors(c)] = 1 - cov[w]
            # But we should use the actual x from the Lagrangian solution, which is selected_mask
            # cov here is computed from selected.
            # Actually, the subgradient of L wrt y_w is (1 - sum_{c: w in N(c)} x_c)
            # If x_c = 1 for c in selected, then sum is cov[w].
            # So g_w = 1 - cov[w].
            
            # Compute norm for step size
            # Standard step size: alpha * (UB - L(y)) / ||g||^2
            # We need a valid UB. If we have best_code, UB = best_size.
            # L(y) = sum(y) + sum_c x_c * (1 - sum_y[c])
            # But x_c is 1 only if sum_y[c] > 1.
            # Let's compute L(y) properly.
            
            # Actually, simpler heuristic step size:
            # alpha = 0.5 * (best_size - len(selected) - something) / ...
            # Let's use a diminishing step size or a fixed small step.
            # Since we don't have a tight UB initially, let's use a heuristic.
            
            if best_code is None:
                # Initial UB: greedy cover
                unc = np.ones(space_size, dtype=bool)
                init_code = greedy_cover(unc)
                init_code = local_search(init_code)
                best_code = init_code[:]
                best_size = len(best_code)
            
            # Compute L(y)
            # L(y) = sum(y) + sum_{c: sum_y[c] > 1} (1 - sum_y[c])
            L_val = np.sum(y)
            if np.any(selected_mask):
                L_val += np.sum(1 - sum_y[selected_mask])
            
            # Gradient norm
            g = 1.0 - cov  # cov is int, g is float
            g_norm_sq = np.sum(g * g)
            
            if g_norm_sq > 0:
                # Step size: (UB - L_val) / g_norm_sq
                # Cap it to avoid large jumps
                step = (best_size - L_val) / g_norm_sq
                if step < 0:
                    step = 0.01
                # Diminishing factor
                alpha = 0.5
                y += alpha * step * g
                # Clip y to [0, 1] or [0, inf]? Usually y >= 0.
                y = np.maximum(y, 0.0)
            
            # Occasionally randomize y to escape local minima
            if it % 20 == 19:
                y += np.random.normal(0, 0.1, space_size)
                y = np.maximum(y, 0.0)
        
        return best_code

    # Run Lagrangian heuristic
    code = lagrangian_heuristic()
    
    if code is None or len(code) == 0:
        # Fallback to pure greedy
        unc = np.ones(space_size, dtype=bool)
        code = greedy_cover(unc)
        code = local_search(code)

    # Final verification
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in code:
        final_cov[neighbors[c]] += 1
    
    if np.any(final_cov == 0):
        # Fallback to greedy if invalid
        unc = np.ones(space_size, dtype=bool)
        code = greedy_cover(unc)
        code = local_search(code)
        # Force cover remaining
        if unc.any():
            remaining = np.where(unc)[0]
            for w in remaining:
                code.append(w)
                unc[neighbors[w]] = False

    return [idx_to_word(c) for c in code]
