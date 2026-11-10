import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    U = set(range(65, n + 1))
    
    def is_valid_L(L_set):
        """
        Check if S = U U L_set is valid.
        """
        L_list = list(L_set)
        nL = len(L_list)
        
        # Case 3: One in U, two in L
        # For each pair b,c in L, check if exists a in U s.t. a+b<=128, a+c<=128, and b+c in S.
        # a exists iff min(128-b, 128-c) >= 65.
        # b+c in S iff b+c in L_set or b+c >= 65.
        for i in range(nL):
            b = L_list[i]
            for j in range(i + 1, nL):
                c = L_list[j]
                if min(128 - b, 128 - c) >= 65:
                    if b + c in L_set or b + c >= 65:
                        return False
        
        # Case 4: Three in L
        # For each triple a,b,c in L, check if sums are in S.
        # s in S iff s in L_set or s >= 65.
        for i in range(nL):
            a = L_list[i]
            for j in range(i + 1, nL):
                b = L_list[j]
                for k in range(j + 1, nL):
                    c = L_list[k]
                    s_ab = a + b
                    s_ac = a + c
                    s_bc = b + c
                    
                    if (s_ab in L_set or s_ab >= 65) and \
                       (s_ac in L_set or s_ac >= 65) and \
                       (s_bc in L_set or s_bc >= 65):
                        return False
        
        return True

    def greedy_add(L_set):
        """Try to add any element from 1..64 that keeps validity."""
        improved = True
        while improved:
            improved = False
            for x in range(1, 65):
                if x in L_set:
                    continue
                test_L = L_set | {x}
                if is_valid_L(test_L):
                    L_set = test_L
                    improved = True
                    break
        return L_set

    best_L = set()
    best_size = 0
    
    # Seed 1
    L_seed = set(range(17, 33))
    if is_valid_L(L_seed):
        best_L = L_seed
        best_size = len(L_seed)
        
    random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        # Start with small random set
        current_L = set(candidates[:random.randint(3, 10)])
        
        # Greedy expansion
        current_L = greedy_add(current_L)
        
        # Local search: Random removal and re-addition
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try to add
            new_L = greedy_add(current_L)
            if len(new_L) > len(current_L):
                current_L = new_L
                improved = True
                continue
                
            # If stuck, try removing a random element and re-adding
            if current_L:
                x_to_remove = random.choice(list(current_L))
                perturbed_L = current_L - {x_to_remove}
                new_L = greedy_add(perturbed_L)
                if len(new_L) > len(current_L):
                    current_L = new_L
                    improved = True
                elif len(new_L) == len(current_L):
                    # Maybe accept same size to escape local minima?
                    # Only if it allows further growth later.
                    # For now, only accept if strictly better.
                    # But we might be stuck. Let's accept with some probability or if it's different.
                    # Actually, just trying different removals is enough.
                    pass
        
        if len(current_L) > best_size:
            best_L = current_L
            best_size = len(current_L)
            
    # Final polish
    best_L = greedy_add(best_L)
    
    final_set = list(U) + list(best_L)
    final_set = [int(x) for x in final_set]
    
    return final_set
