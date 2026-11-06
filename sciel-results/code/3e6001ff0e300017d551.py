import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 256
    
    # Check if a set is valid
    def is_valid(S):
        S_list = sorted(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for c in S_list[j:]:  # c >= b
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Greedy: try to build a large set
    # Strategy: start with upper half, then try to add lower elements
    # Better: greedy by trying each number from 256 down to 1
    
    best_set = set(range(192, 257))  # seed baseline: 65 elements
    
    # Try greedy construction: add elements from 256 down to 1
    # For each candidate, check if adding it preserves validity
    def can_add(S, x):
        S_list = sorted(S)
        # Check all pairs (a, b, c) involving x
        # Case 1: x is 'a' (smallest): x < b <= c, x | (b+c)
        for i in range(len(S_list)):
            b = S_list[i]
            if x < b:
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        # Case 2: x is 'b': a < x <= c, a | (x+c)
        for a in S_list:
            if a < x:
                for c in S_list:
                    if c >= x and (x + c) % a == 0:
                        return False
        # Case 3: x is 'c': a < b <= x, a | (b+x)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b <= x and a < b and (b + x) % a == 0:
                    return False
        return True
    
    # Greedy from 256 down to 1
    S = set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if can_add(S, x):
            S.add(x)
    
    if len(S) > len(best_set):
        best_set = S
    
    # Also try greedy from 1 up to 256
    S2 = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if can_add(S2, x):
            S2.add(x)
    
    if len(S2) > len(best_set):
        best_set = S2
    
    # Try local search: remove one, try to add two
    S3 = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S3_list = sorted(S3)
        for i in range(len(S3_list)):
            if time.time() > deadline:
                break
            a_remove = S3_list[i]
            S3_trial = S3 - {a_remove}
            # Try to add each missing number
            for x in range(1, 257):
                if x in S3_trial:
                    continue
                if can_add(S3_trial, x):
                    S3_trial.add(x)
                    break
            if len(S3_trial) > len(S3):
                S3 = S3_trial
                improved = True
                break
    
    if len(S3) > len(best_set):
        best_set = S3
    
    return sorted(best_set)
