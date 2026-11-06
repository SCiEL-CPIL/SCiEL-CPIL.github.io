<<<<<<< SEARCH
def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    bits, score = greedy_construct()
    best_bits = bits
    best_score = score
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        return bits_to_list(best_bits)
    
    # Phase 2: Multiple rounds of SA + local search polish
    remaining = TIME_LIMIT - elapsed
    num_rounds = 5
    sa_budget = remaining * 0.6 / num_rounds
    polish_budget = remaining * 0.4 / num_rounds
    
    for r in range(num_rounds):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # SA from current best
        sa_bits, sa_score = simulated_annealing(best_bits, best_score, sa_budget)
        if sa_score > best_score:
            best_bits = sa_bits
            best_score = sa_score
        
        # Local search polish
        ls_bits, ls_score = local_search_polish(best_bits, best_score, polish_budget)
        if ls_score > best_score:
            best_bits = ls_bits
            best_score = ls_score
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 2:
        return bits_to_list(best_bits)
    
    # One more polish if time allows
    final_bits, final_score = local_search_polish(best_bits, best_score, max(0, TIME_LIMIT - (time.time() - start_time) - 1))
    if final_score > best_score:
        best_bits = final_bits
        best_score = final_score
    
    return bits_to_list(best_bits)
=======
def construct():
    start_time = time.time()
    
    # Phase 1: Multiple greedy constructions, keep best
    candidates = []
    
    # Standard greedy
    b1, s1 = greedy_construct()
    candidates.append((b1, s1))
    
    # Difference-targeting greedy
    b2, s2 = diff_targeting_greedy()
    candidates.append((b2, s2))
    
    # Randomized greedy (perturb choice)
    for _ in range(3):
        bits = 0
        current_score = 0
        used = set()
        for _ in range(N):
            best_gain = -1
            best_val = None
            for v in range(1, MAX_VAL + 1):
                if v in used:
                    continue
                new_bits = bits | (1 << v)
                new_score = score_bitset(new_bits)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_val = v
            if best_val is None:
                break
            # With some probability, pick a near-best instead of best
            if random.random() < 0.2:
                # Pick randomly from top-5 gains
                gains = []
                for v in range(1, MAX_VAL + 1):
                    if v in used:
                        continue
                    new_bits = bits | (1 << v)
                    new_score = score_bitset(new_bits)
                    gains.append((new_score - current_score, v))
                gains.sort(reverse=True)
                if gains:
                    top5 = gains[:5]
                    _, best_val = random.choice(top5)
            bits |= (1 << best_val)
            used.add(best_val)
            current_score = score_bitset(bits)
        candidates.append((bits, current_score))
    
    # Keep best initial
    best_bits, best_score = max(candidates, key=lambda x: x[1])
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        # Quick polish
        b, s = swap_polish(best_bits, best_score, max(0, TIME_LIMIT - elapsed - 2))
        if s > best_score:
            best_bits, best_score = b, s
        return bits_to_list(best_bits)
    
    # Phase 2: Multiple rounds of SA + local search + 2-swap polish
    remaining = TIME_LIMIT - elapsed
    num_rounds = 6
    sa_budget = remaining * 0.45 / num_rounds
    polish_budget = remaining * 0.35 / num_rounds
    two_swap_budget = remaining * 0.20 / num_rounds
    
    for r in range(num_rounds):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # SA from current best
        sa_bits, sa_score = simulated_annealing(best_bits, best_score, sa_budget)
        if sa_score > best_score:
            best_bits = sa_bits
            best_score = sa_score
        
        # Single-swap local search polish
        ls_bits, ls_score = swap_polish(best_bits, best_score, polish_budget)
        if ls_score > best_score:
            best_bits = ls_bits
            best_score = ls_score
        
        # 2-swap polish to escape local optima
        ts_bits, ts_score = two_swap_polish(best_bits, best_score, two_swap_budget)
        if ts_score > best_score:
            best_bits = ts_bits
            best_score = ts_score
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 3:
        return bits_to_list(best_bits)
    
    # Final polish if time allows
    remaining_final = TIME_LIMIT - (time.time() - start_time) - 1
    if remaining_final > 0.5:
        # Alternate between 1-swap and 2-swap polish
        half = remaining_final / 2
        b1, s1 = swap_polish(best_bits, best_score, half)
        if s1 > best_score:
            best_bits, best_score = b1, s1
        b2, s2 = two_swap_polish(best_bits, best_score, half)
        if s2 > best_score:
            best_bits, best_score = b2, s2
    
    return bits_to_list(best_bits)
>>>>>>> REPLACE
