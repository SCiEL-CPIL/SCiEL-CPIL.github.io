import time
import random

N = 256
TIME_LIMIT = 85.0

def construct():
    start_time = time.time()
    rng = random.Random(42)

    # 1. Precompute APs
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    # 2. Build adjacency structures
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)
            
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]

    def count_3aps(S):
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def is_safe_add(S, x):
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                return False
        return True

    def gain_3aps(S, x):
        gain = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        return gain

    def loss_3aps(S, x):
        loss = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                loss += 1
        return loss

    def repair_4aps(S):
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    elems = (a, b, c, d)
                    worst_idx = min(range(4), key=lambda i: elem_3ap_weight[elems[i]])
                    worst = elems[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def local_search(S_init, time_budget):
        current_S = S_init
        current_score = count_3aps(current_S)
        best_S = current_S
        best_score = current_score
        
        deadline = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < deadline:
            improved = False
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
            
            if not members or not non_members:
                break
                
            best_delta = 0
            best_move = None
            
            for x_remove in members:
                S_minus_x = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                
                for x_add in non_members:
                    if not is_safe_add(S_minus_x, x_add):
                        continue
                    gain = gain_3aps(S_minus_x, x_add)
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (x_remove, x_add)
                        
            if best_move is not None and best_delta > 0:
                x_remove, x_add = best_move
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_delta
                improved = True
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_S
            else:
                # Kick: Random perturbation
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb current_S
                    S_perturb = current_S
                    for _ in range(rng.randint(3, 10)):
                        x = rng.randint(0, N - 1)
                        if (S_perturb >> x) & 1:
                            S_perturb &= ~(1 << x)
                        else:
                            # Check if we can add
                            if is_safe_add(S_perturb, x):
                                S_perturb |= (1 << x)
                    S_perturb = repair_4aps(S_perturb)
                    score_perturb = count_3aps(S_perturb)
                    if score_perturb >= best_score * 0.9: # Accept if not too bad
                        current_S = S_perturb
                        current_score = score_perturb
                    no_improve_count = 0
                    max_no_improve = 10 # Tighten after kick
                else:
                    break

        return best_S, best_score

    # --- Multi-Start ---
    candidates = []
    
    # 1. Greedy by weight
    order1 = sorted(range(N), key=lambda x: -elem_3ap_weight[x])
    S1 = 0
    for x in order1:
        if is_safe_add(S1, x):
            S1 |= (1 << x)
    S1 = repair_4aps(S1)
    candidates.append(S1)
    
    # 2. Random
    S2 = 0
    for x in range(N):
        if rng.random() < 0.5:
            S2 |= (1 << x)
    S2 = repair_4aps(S2)
    candidates.append(S2)
    
    # 3. Residue class (e.g., x % 3 != 0)
    S3 = 0
    for x in range(N):
        if x % 3 != 0:
            S3 |= (1 << x)
    S3 = repair_4aps(S3)
    candidates.append(S3)

    best_S_overall = 0
    best_score_overall = 0
    
    time_left = TIME_LIMIT - (time.time() - start_time)
    time_per = time_left / len(candidates)
    
    for S_cand in candidates:
        if time.time() > TIME_LIMIT - 5:
            break
        S_opt, score_opt = local_search(S_cand, time_per)
        if score_opt > best_score_overall:
            best_score_overall = score_opt
            best_S_overall = S_opt
            
    # Final polish
    if time.time() < TIME_LIMIT - 2:
        S_final, score_final = local_search(best_S_overall, TIME_LIMIT - (time.time() - start_time) - 1)
        if score_final > best_score_overall:
            best_score_overall = score_final
            best_S_overall = S_final
            
    result = [x + 1 for x in range(N) if (best_S_overall >> x) & 1]
    if not result:
        return [1]
    return result
