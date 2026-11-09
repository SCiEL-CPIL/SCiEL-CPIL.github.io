import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 90s hard limit

    N = 17
    MIN_D = 6
    TOTAL = 1 << N

    # Precompute popcount lookup for all 17-bit values (fast vectorized index)
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)

    # ---- 1. Generate the known 256-word linear code [17,8,6] ----
    def generate_linear_code():
        basis = []
        current_code = [0]
        for g in range(1, TOTAL):
            if time.time() > deadline:
                break
            valid = True
            for c in current_code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_code = []
                for c in current_code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                current_code = new_code
                if len(basis) >= 8:
                    break
        return current_code

    best_code = generate_linear_code()
    best_size = len(best_code)
    best_set = set(best_code)

    # ---- Vectorized feasibility check ----
    def feasible(words, code_np):
        """True iff every word in `words` has Hamming distance >= MIN_D from all of code_np."""
        if code_np.size == 0:
            return True
        xors = words[:, None] ^ code_np[None, :]
        dists = popcount_lookup[xors]
        return bool(dists.min() >= MIN_D)

    def feasible_single(word, code_np):
        if code_np.size == 0:
            return True
        xors = code_np ^ word
        return bool(popcount_lookup[xors].min() >= MIN_D)

    # ---- Weight groups for biased orderings ----
    weight_groups = {}
    for w in range(TOTAL):
        weight_groups.setdefault(int(popcount_lookup[w]), []).append(w)

    all_words_list = list(range(TOTAL))

    # ---- Greedy pass over a given order ----
    def greedy_pass(order):
        code = []
        code_np = np.array([], dtype=np.uint32)
        for w in order:
            if feasible_single(w, code_np):
                code.append(w)
                code_np = np.append(code_np, w)
        return code

    # ---- Local improvement: extend a code with remaining candidates ----
    def try_extend(code):
        code_set = set(code)
        code_np = np.array(code, dtype=np.uint32)
        candidates = [w for w in range(TOTAL) if w not in code_set]
        random.shuffle(candidates)
        for w in candidates:
            if feasible_single(w, code_np):
                code.append(w)
                code_set.add(w)
                code_np = np.append(code_np, w)
        return code

    # ---- Swap-based local search: drop one word, greedily refill, keep if net gain ----
    def local_search_swap(code, max_rounds=20):
        code_np = np.array(code, dtype=np.uint32)
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 8:
                    break
                removed = code[i]
                temp_code = code[:i] + code[i+1:]
                temp_set = set(temp_code)
                temp_np = np.array(temp_code, dtype=np.uint32)
                added = 0
                for w in range(TOTAL):
                    if time.time() > deadline - 8:
                        break
                    if w in temp_set:
                        continue
                    if feasible_single(w, temp_np):
                        temp_code.append(w)
                        temp_set.add(w)
                        temp_np = np.append(temp_np, w)
                        added += 1
                        if added >= 2:  # net gain >= 1
                            break
                if len(temp_code) > len(code):
                    code = temp_code
                    code_np = temp_np
                    improved = True
                    break
            if not improved:
                break
        return code

    # ---- Main search loop ----
    pass_count = 0
    while time.time() < deadline - 10:
        pass_count += 1

        if pass_count % 3 == 0:
            # Weight-biased ordering (shuffle within weight classes, sometimes shuffle class order)
            order = []
            weight_order = sorted(weight_groups.keys())
            if random.random() < 0.5:
                random.shuffle(weight_order)
            for wt in weight_order:
                group = weight_groups[wt][:]
                random.shuffle(group)
                order.extend(group)
            current_code = greedy_pass(order)
        elif pass_count % 3 == 1:
            # Uniform random ordering
            random.shuffle(all_words_list)
            current_code = greedy_pass(all_words_list[:])
        else:
            # Extend the current best code, then local-search to polish
            current_code = list(best_code)
            current_code = try_extend(current_code)
            current_code = local_search_swap(current_code)

        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
            best_set = set(best_code)

    # Final safety polish on the best code found
    if time.time() < deadline - 3:
        polished = try_extend(list(best_code))
        if len(polished) > best_size:
            best_code = polished
            best_size = len(best_code)

    return best_code
