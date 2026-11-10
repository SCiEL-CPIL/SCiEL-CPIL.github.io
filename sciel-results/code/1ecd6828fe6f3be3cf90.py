import time

def construct():
    start_time = time.time()
    deadline = 90.0
    
    # The greedy construction for distinct subset sums
    # a_1 = 1
    # a_i = 1 + sum(a_1 ... a_{i-1})
    # This results in 1, 2, 4, 8, ...
    # Max element is 2^(n-1)
    
    n = 12
    res = []
    current_sum = 0
    
    for i in range(n):
        if time.time() > deadline - 0.1:
            break
        if i == 0:
            res.append(1)
            current_sum = 1
        else:
            next_val = current_sum + 1
            res.append(next_val)
            current_sum += next_val
            
    return res
