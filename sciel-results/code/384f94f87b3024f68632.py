import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
    
    # Seed: c[28]=1, c[29]=-1 gives 1/29 - 1/30 = 1/870
    seed_val = A[29] - A[30]
    best_val = abs(seed_val)
    best_c = [0] * 30
    best_c[28] = 1
    best_c[29] = -1
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i+1]
        return s
    
    def try_update(new_c, new_sum):
        nonlocal best_val, best_c
        if new_sum != 0:
            new_abs = abs(new_sum)
            if new_abs < best_val:
                best_val = new_abs
                best_c = new_c[:]
                return True
        return False
    
    def local_search(c, max_steps=5000):
        current_sum = get_sum(c)
        
        if current_sum != 0:
            try_update(c, current_sum)
        
        steps = 0
        no_improve = 0
        while steps < max_steps and time.time() - start_time < deadline:
            steps += 1
            
            if current_sum == 0:
                # Try random double flip to escape zero
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i == j:
                    i = (i + 1) % 30
                old_i = c[i]
                old_j = c[j]
                found = False
                for new_i in [-1, 0, 1]:
                    for new_j in [-1, 0, 1]:
                        if new_i == old_i and new_j == old_j:
                            continue
                        delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                        new_sum = current_sum + delta
                        if new_sum != 0 and abs(new_sum) < best_val:
                            c[i] = new_i
                            c[j] = new_j
                            current_sum = new_sum
                            try_update(c, new_sum)
                            found = True
                            no_improve = 0
                            break
                    if found:
                        break
                if not found:
                    # Try triple flip
                    i = random.randint(0, 29)
                    j = random.randint(0, 29)
                    k = random.randint(0, 29)
                    if i == j:
                        i = (i + 1) % 30
                    if j == k:
                        j = (j + 1) % 30
                    if i == k:
                        k = (k + 1) % 30
                    old_i, old_j, old_k = c[i], c[j], c[k]
                    found = False
                    for ni in [-1, 0, 1]:
                        for nj in [-1, 0, 1]:
                            for nk in [-1, 0, 1]:
                                if ni == old_i and nj == old_j and nk == old_k:
                                    continue
                                delta = (ni - old_i)*A[i+1] + (nj - old_j)*A[j+1] + (nk - old_k)*A[k+1]
                                new_sum = current_sum + delta
                                if new_sum != 0 and abs(new_sum) < best_val:
                                    c[i], c[j], c[k] = ni, nj, nk
                                    current_sum = new_sum
                                    try_update(c, new_sum)
                                    found = True
                                    no_improve = 0
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        break
            else:
                # Normal hill climbing: try all single moves, pick best
                best_delta = None
                best_idx = -1
                best_new_val = None
                best_new_sum = None
                
                for idx in range(30):
                    old_val = c[idx]
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * A[idx+1]
                        new_sum = current_sum + delta
                        if new_sum != 0:
                            if best_new_sum is None or abs(new_sum) < abs(best_new_sum):
                                best_new_sum = new_sum
                                best_idx = idx
                                best_new_val = new_val
                
                if best_idx >= 0 and abs(best_new_sum) < abs(current_sum):
                    c[best_idx] = best_new_val
                    current_sum = best_new_sum
                    if try_update(c, current_sum):
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Try random perturbation to escape
                    if random.random() < 0.3:
                        idx = random.randint(0, 29)
                        c[idx] = random.choice([-1, 0, 1])
                        current_sum = get_sum(c)
                        if current_sum != 0:
                            try_update(c, current_sum)
                    else:
                        # Try double flip
                        i = random.randint(0, 29)
                        j = random.randint(0, 29)
                        if i == j:
                            j = (j + 1) % 30
                        old_i, old_j = c[i], c[j]
                        found = False
                        for ni in [-1, 0, 1]:
                            for nj in [-1, 0, 1]:
                                if ni == old_i and nj == old_j:
                                    continue
                                delta = (ni - old_i)*A[i+1] + (nj - old_j)*A[j+1]
                                new_sum = current_sum + delta
                                if new_sum != 0 and abs(new_sum) < abs(current_sum):
                                    c[i], c[j] = ni, nj
                                    current_sum = new_sum
                                    try_update(c, new_sum)
                                    found = True
                                    no_improve = 0
                                    break
                            if found:
                                break
                        if not found:
                            no_improve += 1
                            if no_improve > 100:
                                break
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < deadline:
        num_restarts += 1
        
        # Generate diverse random starts
        if num_restarts % 3 == 1:
            # Sparse random
            c = [0] * 30
            num_nonzero = random.randint(2, 8)
            indices = random.sample(range(30), num_nonzero)
            for idx in indices:
                c[idx] = random.choice([-1, 1])
        elif num_restarts % 3 == 2:
            # Dense random
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
        else:
            # Start near seed
            c = best_c[:]
            # Perturb a few elements
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 0, 1])
        
        local_search(c)
        
        if best_val == 1:
            break
    
    return best_c
