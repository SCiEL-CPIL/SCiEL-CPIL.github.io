import time
import numpy as np
import random
from collections import deque

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0

    total = q ** n
    ball_size = 1 + n * (q - 1)
    full_mask = (1 << total) - 1

    # Precompute place values
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball as bitmask (Python int) for each word
    ball_mask = [0] * total
    # Precompute ball as list for iteration
    ball_list = [None] * total
    for i in range(total):
        digits = []
        temp = i
        for j in range(n - 1, -1, -1):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        mask = 1 << i
        lst = [i]
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                mask |= (1 << new_idx)
                lst.append(new_idx)
        ball_mask[i] = mask
        ball_list[i] = lst

    def greedy_build(seed):
        rng = random.Random(seed)
        covered_mask = 0
        code = []
        candidates = list(range(total))
        rng.shuffle(candidates)
        for c in candidates:
            new_cov = bin(ball_mask[c] & ~covered_mask).count('1')
            if new_cov > 0:
                code.append(c)
                covered_mask |= ball_mask[c]
                if covered_mask == full_mask:
                    break
        return code

    # Multiple greedy restarts
    best_initial = None
    best_initial_size = float('inf')
    for seed in range(20):
        if time.time() > deadline - 60.0:
            break
        init_code = greedy_build(seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = init_code

    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)

    # Count coverage using array
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        for w in ball_list[c]:
            cnt[w] += 1

    def run_delete_phase(code_set, cnt_arr):
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_list[c]
                if min(cnt_arr[w] for w in ball) >= 2:
                    code_set.remove(c)
                    for w in ball:
                        cnt_arr[w] -= 1
                    removed += 1
                    changed = True
        return removed

    def try_swap(c_remove, c_add, code_set, cnt_arr):
        if c_add in code_set or c_remove == c_add:
            return False
        ball_rem = ball_list[c_remove]
        ball_add = ball_list[c_add]
        add_set = set(ball_add)
        for w in ball_rem:
            if w not in add_set and cnt_arr[w] < 2:
                return False
        code_set.remove(c_remove)
        code_set.add(c_add)
        for w in ball_rem:
            cnt_arr[w] -= 1
        for w in ball_add:
            cnt_arr[w] += 1
        return True

    def get_valid_swaps(c_remove, code_set, cnt_arr):
        ball_rem = ball_list[c_remove]
        unique_words = [w for w in ball_rem if cnt_arr[w] == 1]
        if not unique_words:
            return None
        valid_mask = ball_mask[unique_words[0]]
        for u in unique_words[1:]:
            valid_mask &= ball_mask[u]
            if valid_mask == 0:
                return []
        valid_list = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            if idx not in code_set:
                valid_list.append(idx)
            vm ^= lsb
        return valid_list

    def try_compress_pair(c1, c2, code_set, cnt_arr):
        """Try to replace two codewords with one if possible."""
        if c1 == c2 or c1 not in code_set or c2 not in code_set:
            return False
        
        # Compute coverage by all other codewords
        other_coverage = 0
        for c in code_set:
            if c != c1 and c != c2:
                other_coverage |= ball_mask[c]
        
        # Words only covered by c1 or c2
        uncovered_without_pair = full_mask ^ other_coverage
        # Words that need to be covered by the new codeword
        must_cover = uncovered_without_pair & (ball_mask[c1] | ball_mask[c2])
        
        if must_cover == 0:
            # Both codewords are redundant, just remove both
            code_set.remove(c1)
            code_set.remove(c2)
            for w in ball_list[c1]:
                cnt_arr[w] -= 1
            for w in ball_list[c2]:
                cnt_arr[w] -= 1
            return True
        
        # Find a single codeword that covers all words in must_cover
        # The new codeword's ball must be a superset of must_cover
        # i.e., for all w in must_cover, w must be in ball(new)
        # Equivalently, new must be in the intersection of balls of all words in must_cover
        
        # If must_cover has too many words, it's unlikely to find a single codeword
        must_cover_count = bin(must_cover).count('1')
        if must_cover_count > ball_size:
            return False
        
        # Find intersection of balls of all words in must_cover
        # Start with ball of first word
        mc = must_cover
        lsb = mc & (-mc)
        first_word = lsb.bit_length() - 1
        valid_mask = ball_mask[first_word]
        
        mc ^= lsb
        while mc:
            lsb = mc & (-mc)
            w = lsb.bit_length() - 1
            valid_mask &= ball_mask[w]
            if valid_mask == 0:
                return False
            mc ^= lsb
        
        # Find a codeword in valid_mask that is not in code_set
        # Or could be c1 or c2 if we're replacing
        vm = valid_mask
        found = None
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            if idx not in code_set or idx == c1 or idx == c2:
                found = idx
                break
            vm ^= lsb
        
        if found is None:
            return False
        
        # Perform the compression
        code_set.remove(c1)
        code_set.remove(c2)
        code_set.add(found)
        for w in ball_list[c1]:
            cnt_arr[w] -= 1
        for w in ball_list[c2]:
            cnt_arr[w] -= 1
        for w in ball_list[found]:
            cnt_arr[w] += 1
        return True

    # Local search with iterative compression
    no_improve = 0
    max_no_improve = 15
    rng = random.Random(123)

    while time.time() < deadline - 5.0:
        # Delete phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            no_improve = 0
        else:
            no_improve += 1

        # Swap phase
        swaps_done = 0
        max_swaps = 300
        while swaps_done < max_swaps and time.time() < deadline - 5.0:
            swaps_done += 1
            if not current_code:
                break
            c = rng.choice(list(current_code))
            valid = get_valid_swaps(c, current_code, cnt)
            if valid is None:
                c_prime = rng.randint(0, total - 1)
                attempts = 0
                while c_prime in current_code and attempts < 100:
                    c_prime = rng.randint(0, total - 1)
                    attempts += 1
                if c_prime in current_code:
                    continue
            elif len(valid) == 0:
                no_improve += 1
                continue
            else:
                c_prime = rng.choice(valid)

            if try_swap(c, c_prime, current_code, cnt):
                removed = run_delete_phase(current_code, cnt)
                if removed > 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # Iterative compression phase: try to replace pairs with single codewords
        if time.time() < deadline - 8.0 and len(current_code) >= 2:
            code_list = list(current_code)
            rng.shuffle(code_list)
            compression_attempts = 0
            max_compression_attempts = 30
            while compression_attempts < max_compression_attempts and time.time() < deadline - 8.0:
                compression_attempts += 1
                if len(current_code) < 2:
                    break
                pair = rng.sample(list(current_code), 2)
                c1, c2 = pair
                if try_compress_pair(c1, c2, current_code, cnt):
                    removed = run_delete_phase(current_code, cnt)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    no_improve = 0
                else:
                    no_improve += 1

        # Restart if stuck
        if no_improve > max_no_improve and time.time() < deadline - 20.0:
            seed = int(time.time() * 1000) % 100000
            init_code = greedy_build(seed)
            if len(init_code) < len(current_code) + 3:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    for w in ball_list[c]:
                        cnt[w] += 1
                no_improve = 0

        if best_size <= 160:
            break

    # Final polish
    if time.time() < deadline - 3.0:
        changed = True
        iters = 0
        while changed and time.time() < deadline - 2.0 and iters < 15:
            iters += 1
            changed = False
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                changed = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
            # Try all swaps
            for c in list(current_code):
                if time.time() > deadline - 2.0:
                    break
                valid = get_valid_swaps(c, current_code, cnt)
                if valid is None:
                    continue
                if not valid:
                    continue
                for cp in valid[:3]:
                    if time.time() > deadline - 2.0:
                        break
                    if try_swap(c, cp, current_code, cnt):
                        removed = run_delete_phase(current_code, cnt)
                        if removed > 0:
                            changed = True
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code = list(current_code)

    # Convert to output format
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
