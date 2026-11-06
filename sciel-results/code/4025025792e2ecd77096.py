import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    order = list(range(1, N + 1))
    
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]

    best_score = 0.0
    best_set_bits = 0
    
    def is_valid_addition(x, S_bits, S_list):
        for a in S_list:
            c = 2 * x - a
            if 1 <= c <= N and ((S_bits >> c) & 1):
                return False
            c = 2 * a - x
            if 1 <= c <= N and ((S_bits >> c) & 1):
                return False
            s = a + x
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N and ((S_bits >> mid) & 1):
                    return False
        return True

    def greedy_construct():
        S_bits = 0
        S_list = []
        current_sum = 0.0
        for x in range(1, N + 1):
            if is_valid_addition(x, S_bits, S_list):
                S_bits |= (1 << x)
                S_list.append(x)
                current_sum += weights[x]
        return S_bits, S_list, current_sum

    def steepest_ascent(S_bits, S_list, current_sum):
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_gain = 0.0
            best_x_out = -1
            best_x_in = -1
            
            for x_out in S_list:
                for x_in in range(1, x_out):
                    if (S_bits >> x_in) & 1:
                        continue
                    
                    new_S_bits = S_bits & ~(1 << x_out)
                    valid = True
                    for a in S_list:
                        if a == x_out:
                            continue
                        c = 2 * x_in - a
                        if 1 <= c <= N and ((new_S_bits >> c) & 1):
                            valid = False
                            break
                        c = 2 * a - x_in
                        if 1 <= c <= N and ((new_S_bits >> c) & 1):
                            valid = False
                            break
                        s = a + x_in
                        if s % 2 == 0:
                            mid = s // 2
                            if 1 <= mid <= N and ((new_S_bits >> mid) & 1):
                                valid = False
                                break
                    if valid:
                        gain = weights[x_in] - weights[x_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_x_out = x_out
                            best_x_in = x_in
                            
            if best_gain > 1e-9:
                S_bits &= ~(1 << best_x_out)
                S_bits |= (1 << best_x_in)
                S_list.remove(best_x_out)
                S_list.append(best_x_in)
                current_sum += best_gain
                improved = True
                
        return S_bits, S_list, current_sum

    # 1. Greedy + Local Search
    S_bits, S_list, score = greedy_construct()
    S_bits, S_list, score = steepest_ascent(S_bits, S_list, score)
    if score > best_score:
        best_score = score
        best_set_bits = S_bits

    # 2. Random Restarts
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20.0:
            break
        cur_bits = best_set_bits
        cur_list = [i for i in range(1, N+1) if (cur_bits >> i) & 1]
        cur_sum = sum(weights[i] for i in cur_list)
        
        if len(cur_list) > 3:
            to_rem = random.sample(cur_list, 3)
            for x in to_rem:
                cur_bits &= ~(1 << x)
                cur_list.remove(x)
                cur_sum -= weights[x]
        
        cur_bits, cur_list, cur_sum = steepest_ascent(cur_bits, cur_list, cur_sum)
        if cur_sum > best_score:
            best_score = cur_sum
            best_set_bits = cur_bits

    # 3. B&B
    def solve(idx, current_sum, S_bits, S_list):
        nonlocal best_score, best_set_bits
        if time.time() > deadline - 2.0:
            return
        if current_sum + suffix_sum[idx] <= best_score:
            return
        if idx == N:
            if current_sum > best_score:
                best_score = current_sum
                best_set_bits = S_bits
            return
        
        x = order[idx]
        
        # Include x
        if is_valid_addition(x, S_bits, S_list):
            S_bits |= (1 << x)
            S_list.append(x)
            solve(idx + 1, current_sum + weights[x], S_bits, S_list)
            S_list.pop()
            S_bits &= ~(1 << x)
            
        # Exclude x
        solve(idx + 1, current_sum, S_bits, S_list)

    solve(0, 0.0, 0, [])

    final_set = [i for i in range(1, N+1) if (best_set_bits >> i) & 1]
    return final_set
