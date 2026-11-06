import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    best_indices = [0] # Fallback: {2} -> sum 1/2 < 1
    best_sum = Fraction(1, 2)
    
    def local_search(initial_indices, initial_sum):
        indices = initial_indices.copy()
        curr_sum = initial_sum
        in_set = set(indices)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single element that fits
            best_add_i = -1
            best_add_sum = curr_sum
            for i in range(n):
                if i in in_set:
                    continue
                new_sum = curr_sum + reciprocals[i]
                if new_sum < 1 and new_sum > best_add_sum:
                    best_add_sum = new_sum
                    best_add_i = i
            
            if best_add_i != -1:
                indices.append(best_add_i)
                in_set.add(best_add_i)
                curr_sum = best_add_sum
                improved = True
                continue # Restart loop as set changed significantly
                
            # 2. Try to swap one out, one in
            best_swap_gain = Fraction(0)
            best_swap_out = -1
            best_swap_in = -1
            
            for out_i in indices:
                r_out = reciprocals[out_i]
                for in_i in range(n):
                    if in_i in in_set:
                        continue
                    r_in = reciprocals[in_i]
                    gain = r_in - r_out
                    if gain <= 0:
                        continue
                    new_sum = curr_sum + gain
                    if new_sum < 1 and gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_out = out_i
                        best_swap_in = in_i
                        
            if best_swap_gain > 0:
                indices.remove(best_swap_out)
                in_set.remove(best_swap_out)
                indices.append(best_swap_in)
                in_set.add(best_swap_in)
                curr_sum += best_swap_gain
                improved = True
                continue
                
            # 3. Try to remove one, add two
            best_r1_gain = Fraction(0)
            best_r1_out = -1
            best_r1_in1 = -1
            best_r1_in2 = -1
            
            for out_i in indices:
                r_out = reciprocals[out_i]
                remaining = [i for i in range(n) if i not in in_set and i != out_i]
                for a in range(len(remaining)):
                    in1 = remaining[a]
                    r_in1 = reciprocals[in1]
                    for b in range(a+1, len(remaining)):
                        in2 = remaining[b]
                        r_in2 = reciprocals[in2]
                        gain = r_in1 + r_in2 - r_out
                        if gain <= 0:
                            continue
                        new_sum = curr_sum + gain
                        if new_sum < 1 and gain > best_r1_gain:
                            best_r1_gain = gain
                            best_r1_out = out_i
                            best_r1_in1 = in1
                            best_r1_in2 = in2
            
            if best_r1_gain > 0:
                indices.remove(best_r1_out)
                in_set.remove(best_r1_out)
                indices.append(best_r1_in1)
                indices.append(best_r1_in2)
                in_set.add(best_r1_in1)
                in_set.add(best_r1_in2)
                curr_sum += best_r1_gain
                improved = True
                continue
                
            # 4. Try to remove two, add two
            if len(indices) >= 2:
                best_r2_gain = Fraction(0)
                best_r2_out1 = -1
                best_r2_out2 = -1
                best_r2_in1 = -1
                best_r2_in2 = -1
                
                for a in range(len(indices)):
                    out1 = indices[a]
                    r_out1 = reciprocals[out1]
                    for b in range(a+1, len(indices)):
                        out2 = indices[b]
                        r_out2 = reciprocals[out2]
                        r_out_sum = r_out1 + r_out2
                        remaining = [i for i in range(n) if i not in in_set and i != out1 and i != out2]
                        for c in range(len(remaining)):
                            in1 = remaining[c]
                            r_in1 = reciprocals[in1]
                            for d in range(c+1, len(remaining)):
                                in2 = remaining[d]
                                r_in2 = reciprocals[in2]
                                gain = r_in1 + r_in2 - r_out_sum
                                if gain <= 0:
                                    continue
                                new_sum = curr_sum + gain
                                if new_sum < 1 and gain > best_r2_gain:
                                    best_r2_gain = gain
                                    best_r2_out1 = out1
                                    best_r2_out2 = out2
                                    best_r2_in1 = in1
                                    best_r2_in2 = in2
                
                if best_r2_gain > 0:
                    indices.remove(best_r2_out1)
                    indices.remove(best_r2_out2)
                    in_set.remove(best_r2_out1)
                    in_set.remove(best_r2_out2)
                    indices.append(best_r2_in1)
                    indices.append(best_r2_in2)
                    in_set.add(best_r2_in1)
                    in_set.add(best_r2_in2)
                    curr_sum += best_r2_gain
                    improved = True
                    continue
                    
        return indices, curr_sum

    # Initial deterministic greedy
    order = list(range(n))
    init_indices = []
    init_sum = Fraction(0)
    for i in order:
        ns = init_sum + reciprocals[i]
        if ns < 1:
            init_indices.append(i)
            init_sum = ns
            
    idx, s = local_search(init_indices, init_sum)
    if s > best_sum:
        best_sum = s
        best_indices = idx
        
    # Random restarts
    while time.time() < deadline:
        random.shuffle(order)
        curr_indices = []
        curr_sum = Fraction(0)
        for i in order:
            ns = curr_sum + reciprocals[i]
            if ns < 1:
                curr_indices.append(i)
                curr_sum = ns
                
        idx, s = local_search(curr_indices, curr_sum)
        if s > best_sum:
            best_sum = s
            best_indices = idx
            
    return best_indices

if __name__ == "__main__":
    res = construct()
    denominators = list(range(2, 49))
    s = Fraction(0)
    for idx in res:
        s += Fraction(1, denominators[idx])
    print(f"Indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1 - s}")
