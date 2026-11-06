import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    # Safety margin: 85s out of 90s
    deadline = start_time + 85.0 

    # ---------- Precomputation ----------
    # All 3-APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    # For each element, list of 3-AP indices it appears in
    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append(idx)
        ap_index_by_elem[b].append(idx)
        ap_index_by_elem[c].append(idx)

    # All 4-APs
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    # For each element, list of 4-AP indices it appears in
    four_ap_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for e in ap:
            four_ap_by_elem[e].append(idx)

    # For each 4-AP, the other 3 elements as a bitmask
    four_ap_others = []
    for ap in four_aps:
        for i in range(4):
            x = ap[i]
            others = 0
            for j in range(4):
                if j != i:
                    others |= (1 << (ap[j] - 1))
            four_ap_others.append((x, others))

    # mask4[x]: list of bitmasks of OTHER 3 elements for each 4-AP containing x
    mask4 = [[] for _ in range(n + 1)]
    for x, others in four_ap_others:
        mask4[x].append(others)

    # gain_pairs[x]: for each 3-AP containing x, the pair of other elements as bitmasks
    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for idx in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            others = [y for y in (a, b, c) if y != x]
            gain_pairs[x].append((1 << (others[0] - 1), 1 << (others[1] - 1)))

    # ---------- Core operations ----------
    def can_add(x, S):
        for m in mask4[x]:
            if (S & m) == m:
                return False
        return True

    def gain_add(x, S):
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S & bit1) and (S & bit2):
                g += 1
        return g

    def loss_remove(x, S):
        S_wo = S & ~(1 << (x - 1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo & bit1) and (S_wo & bit2):
                g += 1
        return g

    def score_full(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S >> (a - 1)) & 1 and (S >> (b - 1)) & 1 and (S >> (c - 1)) & 1:
                cnt += 1
        return cnt

    def is_valid(S):
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                return False
        return True

    # ---------- Steepest Ascent Local Search ----------
    def local_search_steepest(S, sc, time_budget=None):
        S = int(S)
        while True:
            if time_budget is not None and time.time() > time_budget:
                break
            
            improved = False
            best_delta = 0
            best_move = None

            # 1. Try all Add moves
            for x in range(1, n + 1):
                if (S >> (x - 1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                delta = gain_add(x, S)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', x, delta)

            # 2. Try all Remove moves (only if it helps? No, removal usually hurts score, 
            #    but might enable better adds later. However, for steepest ascent on score, 
            #    we usually only remove if it allows a swap that is better. 
            #    Pure removal reduces score. So we skip pure removals in steepest ascent 
            #    unless we are doing swaps.)
            
            # 3. Try all Swap moves (Remove y, Add x)
            #    To optimize, we can iterate y in S, then x not in S.
            #    But that's O(N^2). 
            #    Let's check swaps.
            for y in range(1, n + 1):
                if not ((S >> (y - 1)) & 1):
                    continue
                S_wo = S & ~(1 << (y - 1))
                loss_y = loss_remove(y, S)
                
                # If loss is 0, removing y is free. We can just try adding x to S_wo.
                # If loss > 0, we need gain_x > loss_y.
                
                for x in range(1, n + 1):
                    if (S_wo >> (x - 1)) & 1:
                        continue
                    if not can_add(x, S_wo):
                        continue
                    gain_x = gain_add(x, S_wo)
                    delta = gain_x - loss_y
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', x, y, delta)

            if best_move is None:
                break
            
            if best_move[0] == 'add':
                x = best_move[1]
                S |= (1 << (x - 1))
            elif best_move[0] == 'swap':
                x, y = best_move[1], best_move[2]
                S &= ~(1 << (y - 1))
                S |= (1 << (x - 1))
            
            sc += best_delta
            improved = True
            
            if not improved:
                break

        return S, sc

    # ---------- Perturbation for ILS ----------
    def perturb(S, k, time_budget=None):
        S = int(S)
        in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
        if len(in_s) < k:
            k = max(1, len(in_s) // 2)
        
        # Remove k random elements
        to_remove = random.sample(in_s, k)
        for x in to_remove:
            S &= ~(1 << (x - 1))
        
        # Optionally add some random valid elements to "reheat"
        # Let's add k random valid elements
        not_s = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1)]
        random.shuffle(not_s)
        added = 0
        for x in not_s:
            if added >= k:
                break
            if can_add(x, S):
                S |= (1 << (x - 1))
                added += 1
                
        return S

    # ---------- Greedy Build ----------
    def greedy_build(order, min_g=0):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g >= min_g:
                S |= (1 << (x - 1))
        return S

    # ---------- Prune from Full ----------
    def prune_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                best_loss = float('inf')
                best_elem = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_elem = e
                S &= ~(1 << (best_elem - 1))
        return S

    # ---------- Random Density + Repair ----------
    def random_density(p):
        S = 0
        for x in range(1, n + 1):
            if random.random() < p:
                S |= (1 << (x - 1))
        return S

    def repair(S):
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                best_loss = float('inf')
                best_e = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_e = e
                S &= ~(1 << (best_e - 1))
        return S

    # ---------- Main Search ----------
    random.seed(42)
    best_S = 0
    best_score = 0

    def update_best(S, sc):
        nonlocal best_S, best_score
        if sc > best_score:
            best_S = S
            best_score = sc

    # Strategy 1: Prune Full + ILS
    if time.time() < deadline - 20:
        S = prune_full()
        sc = score_full(S)
        S, sc = local_search_steepest(S, sc, time_budget=deadline - 15)
        update_best(S, sc)
        
        # ILS on this best
        current_S = S
        current_sc = sc
        for i in range(20):
            if time.time() > deadline - 10:
                break
            perturbed_S = perturb(current_S, 5)
            perturbed_sc = score_full(perturbed_S)
            perturbed_S, perturbed_sc = local_search_steepest(perturbed_S, perturbed_sc, time_budget=deadline - 5)
            if perturbed_sc > current_sc:
                current_S = perturbed_S
                current_sc = perturbed_sc
                update_best(current_S, current_sc)
            else:
                # Occasionally keep the perturbed one if it's close, to explore? 
                # No, stick to best for now.
                pass

    # Strategy 2: Greedy Random Restarts + ILS
    for r in range(30):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order)
        sc = score_full(S)
        S, sc = local_search_steepest(S, sc, time_budget=deadline - 5)
        update_best(S, sc)
        
        # ILS on this result
        current_S = S
        current_sc = sc
        for i in range(10):
            if time.time() > deadline - 5:
                break
            perturbed_S = perturb(current_S, 3)
            perturbed_sc = score_full(perturbed_S)
            perturbed_S, perturbed_sc = local_search_steepest(perturbed_S, perturbed_sc, time_budget=deadline - 2)
            if perturbed_sc > current_sc:
                current_S = perturbed_S
                current_sc = perturbed_sc
                update_best(current_S, current_sc)

    # Strategy 3: Random Density + Repair + ILS
    for p in [0.4, 0.5, 0.6]:
        if time.time() > deadline - 5:
            break
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = local_search_steepest(S, sc, time_budget=deadline - 2)
        update_best(S, sc)
        
        current_S = S
        current_sc = sc
        for i in range(10):
            if time.time() > deadline - 2:
                break
            perturbed_S = perturb(current_S, 5)
            perturbed_sc = score_full(perturbed_S)
            perturbed_S, perturbed_sc = local_search_steepest(perturbed_S, perturbed_sc, time_budget=deadline - 1)
            if perturbed_sc > current_sc:
                current_S = perturbed_S
                current_sc = perturbed_sc
                update_best(current_S, current_sc)

    # Final Polish on Global Best
    if time.time() < deadline - 1 and best_S:
        best_S, best_score = local_search_steepest(best_S, best_score, time_budget=deadline - 0.1)

    # Verify and return
    result = [x for x in range(1, n + 1) if (best_S >> (x - 1)) & 1]
    if not result:
        result = [1]

    if not is_valid(best_S):
        S = 0
        for x in range(1, n + 1):
            if can_add(x, S):
                S |= (1 << (x - 1))
        result = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]

    return result
