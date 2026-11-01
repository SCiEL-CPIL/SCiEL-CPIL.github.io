import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    balls_arr = np.zeros((N, ball_size), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls.append(neighbors)
        balls_arr[w] = neighbors
    
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def prune(selected, cover_count):
        """Remove codewords that are not essential (all their ball members have cover_count > 1)."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                ball = balls_arr[c]
                if np.min(cover_count[ball]) > 1:
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
        return selected
    
    def greedy_construct(cover_count, uncovered):
        """Greedy set cover: repeatedly pick the word covering most uncovered positions."""
        selected = []
        while np.any(uncovered):
            gains = uncovered[balls_arr].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            ball = balls_arr[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
        return selected
    
    def local_search_swap(selected, cover_count, deadline):
        """Try random swaps to enable further pruning."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            if not selected:
                break
            # Try multiple random swaps
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(len(selected))
                u = selected[i]
                v = np.random.randint(N)
                if v == u:
                    continue
                # Check safety: words in B(u) \ B(v) must have cover_count >= 2
                ball_u = balls_arr[u]
                ball_v = balls_arr[v]
                # Find words in ball_u but not in ball_v
                # Use set difference for small sets
                ball_u_set = set(ball_u)
                ball_v_set = set(ball_v)
                risk = ball_u_set - ball_v_set
                if risk:
                    risk_arr = np.array(list(risk), dtype=np.int32)
                    if np.min(cover_count[risk_arr]) < 2:
                        continue
                # Safe swap
                cover_count[ball_u] -= 1
                cover_count[ball_v] += 1
                selected[i] = v
                # Prune
                new_len_before = len(selected)
                selected = prune(selected, cover_count)
                if len(selected) < new_len_before:
                    improved = True
                    break
                else:
                    # Revert if no improvement (optional: keep for exploration)
                    # For now, revert to avoid getting stuck
                    cover_count[ball_v] -= 1
                    cover_count[ball_u] += 1
                    selected[i] = u
        return selected
    
    def local_search_2for1(selected, cover_count, deadline):
        """Try to replace two codewords with one."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            K = len(selected)
            if K < 2:
                break
            # Try random pairs
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                c1 = selected[i1]
                c2 = selected[i2]
                # Find a word v that covers all words uniquely covered by c1 or c2
                # Words uniquely covered by c1: in B(c1) with cover_count == 1
                # Words uniquely covered by c2: in B(c2) with cover_count == 1
                ball1 = balls_arr[c1]
                ball2 = balls_arr[c2]
                unique1 = ball1[cover_count[ball1] == 1]
                unique2 = ball2[cover_count[ball2] == 1]
                all_unique = set(unique1) | set(unique2)
                if not all_unique:
                    # No unique words, so we can just remove one
                    # Remove the one with more redundancy
                    min1 = np.min(cover_count[ball1])
                    min2 = np.min(cover_count[ball2])
                    if min1 >= min2:
                        cover_count[ball1] -= 1
                        selected.pop(i1)
                    else:
                        cover_count[ball2] -= 1
                        selected.pop(i2)
                    selected = prune(selected, cover_count)
                    improved = True
                    break
                # Try to find v that covers all_unique
                # v must be within distance 1 of all words in all_unique
                # This means v is in the intersection of balls of all words in all_unique
                # Since |all_unique| can be up to 34, this is expensive
                # Instead, try random v and check
                found = False
                for _ in range(10):
                    v = np.random.randint(N)
                    ball_v = balls_arr[v]
                    ball_v_set = set(ball_v)
                    if all(x in ball_v_set for x in all_unique):
                        # Check safety for other words
                        # Words in (ball1 ∪ ball2) \ ball_v must have cover_count >= 2 after removal
                        combined = set(ball1) | set(ball2)
                        risk = combined - ball_v_set
                        if risk:
                            risk_arr = np.array(list(risk), dtype=np.int32)
                            # After removing c1 and c2, cover_count decreases by 1 for ball1 and ball2
                            # For words in risk, they are in ball1 or ball2 but not ball_v
                            # Their new cover_count = old - 1 (if in ball1) - 1 (if in ball2) + 0
                            # But we need to be careful: a word could be in both ball1 and ball2
                            # Let's compute the effective cover count
                            cc_check = cover_count.copy()
                            cc_check[ball1] -= 1
                            cc_check[ball2] -= 1
                            if np.min(cc_check[risk_arr]) < 2:
                                continue
                        # Perform 2-for-1
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[ball_v] += 1
                        selected.pop(max(i1, i2))
                        selected.pop(min(i1, i2))
                        selected.append(v)
                        selected = prune(selected, cover_count)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return selected
    
    best_size = N
    best_set = None
    
    # Main search loop
    while time.time() < deadline - 10.0:
        # 1. Greedy Construction
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        selected = greedy_construct(cover_count, uncovered)
        
        if np.any(uncovered):
            continue
        
        # 2. Initial Pruning
        selected = prune(selected, cover_count)
        
        # 3. Local Search: Swaps
        selected = local_search_swap(selected, cover_count, deadline)
        
        # 4. Local Search: 2-for-1
        selected = local_search_2for1(selected, cover_count, deadline)
        
        # 5. Final Pruning
        selected = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we're getting close to the deadline, do a final polish
        if time.time() > deadline - 20.0 and best_set is not None:
            # Try a few more targeted improvements on the best set
            cover_count_best = np.zeros(N, dtype=np.int32)
            for c in best_set:
                cover_count_best[balls_arr[c]] += 1
            best_set = prune(best_set, cover_count_best)
            best_set = local_search_swap(best_set, cover_count_best, deadline)
            best_set = local_search_2for1(best_set, cover_count_best, deadline)
            best_set = prune(best_set, cover_count_best)
            if len(best_set) < best_size:
                best_size = len(best_set)
                best_set = best_set.copy()
            break
    
    if best_set is None:
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
