import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_A = None
    best_max_mult = float('inf')
    
    # Try different sizes
    # k around 50-60 seems optimal
    for k in range(45, 75, 3):
        for _ in range(10):
            if time.time() > deadline:
                break
                
            A = set(random.sample(range(129), k))
            # Ensure range coverage potential
            if min(A) > 64:
                A.remove(max(A))
                A.add(random.randint(0, 64))
            if max(A) < 128:
                A.remove(min(A))
                A.add(128)
            
            # Repair
            A = repair_coverage(A)
            
            # Search
            A = local_search(A, deadline)
            
            # Evaluate
            max_mult = max_multiplicity(A)
            if max_mult < best_max_mult:
                best_max_mult = max_mult
                best_A = A.copy()
                
        if time.time() > deadline:
            break
            
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)

def max_multiplicity(A):
    A_set = set(A)
    max_n = 0
    for S in range(128, 257):
        c = 0
        for a in A:
            if S - a in A_set:
                c += 1
        diag = 1 if (S % 2 == 0 and S // 2 in A_set) else 0
        n = (c + diag) // 2
        if n > max_n:
            max_n = n
    return max_n

def repair_coverage(A):
    A = set(A)
    # Simple loop
    for _ in range(100):
        missing = []
        A_set = A
        for S in range(128, 257):
            c = 0
            for a in A:
                if S - a in A_set:
                    c += 1
            diag = 1 if (S % 2 == 0 and S // 2 in A_set) else 0
            n = (c + diag) // 2
            if n == 0:
                missing.append(S)
        if not missing:
            break
        S = missing[0]
        # Add pair
        a = random.randint(max(0, S-128), min(128, S))
        b = S - a
        A.add(a)
        A.add(b)
    return A

def local_search(A, deadline):
    A = set(A)
    A_list = list(A)
    
    def get_N(A):
        A_set = set(A)
        N = [0] * 129
        for S in range(128, 257):
            c = 0
            for a in A:
                if S - a in A_set:
                    c += 1
            diag = 1 if (S % 2 == 0 and S // 2 in A_set) else 0
            N[S-128] = (c + diag) // 2
        return N

    N_vec = get_N(A)
    current_max = max(N_vec)
    current_sum = sum(N_vec)
    
    while time.time() < deadline:
        if len(A) > 40:
            a = random.choice(A_list)
            b = random.randint(0, 128)
            if b in A: continue
            A.remove(a)
            A.add(b)
            new_N = get_N(A)
            new_max = max(new_N)
            new_sum = sum(new_N)
            if min(new_N) > 0 and (new_max < current_max or (new_max == current_max and new_sum < current_sum)):
                current_max = new_max
                current_sum = new_sum
                N_vec = new_N
                A_list = list(A)
            else:
                A.remove(b)
                A.add(a)
        else:
            b = random.randint(0, 128)
            if b in A: continue
            A.add(b)
            new_N = get_N(A)
            new_max = max(new_N)
            new_sum = sum(new_N)
            if min(new_N) > 0 and (new_max < current_max or (new_max == current_max and new_sum < current_sum)):
                current_max = new_max
                current_sum = new_sum
                N_vec = new_N
                A_list = list(A)
            else:
                A.remove(b)
                
    return A
