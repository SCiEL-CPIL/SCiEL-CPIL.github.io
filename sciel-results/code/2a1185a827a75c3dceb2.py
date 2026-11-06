import time
import random
import numpy as np
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    primes = _sieve(1000)
    L_max = 100000
    
    covered = np.zeros(L_max, dtype=np.int8)
    classes = []
    
    def add_prime(p, r):
        idx = (r - 1) % p
        covered[idx::p] += 1
        classes.append([p, r])
        
    def remove_prime(p, r):
        idx = (r - 1) % p
        covered[idx::p] -= 1
        for i, c in enumerate(classes):
            if c[0] == p and c[1] == r:
                classes.pop(i)
                break

    def get_L():
        if np.all(covered > 0):
            return L_max
        first_zero = np.nonzero(covered == 0)[0][0]
        return first_zero
    
    # Initialize
    current_sum = 0.0
    for p in primes:
        if current_sum + 1.0/p <= 0.9:
            r = random.randint(0, p-1)
            add_prime(p, r)
            current_sum += 1.0/p
        else:
            break
            
    best_L = get_L()
    best_classes = [list(c) for c in classes]
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            for i in range(len(classes)):
                p, r = classes[i]
                remove_prime(p, r)
                new_r = random.randint(0, p-1)
                add_prime(p, new_r)
            no_improve_count = 0
            cur_L = get_L()
            if cur_L > best_L:
                best_L = cur_L
                best_classes = [list(c) for c in classes]
            continue

        cur_L = get_L()
        if cur_L == L_max:
            break
            
        improved = False
        
        # Strategy 1: Target the gap
        gap = cur_L + 1
        if classes:
            idx = random.randint(0, len(classes) - 1)
            p, r = classes[idx]
            new_r = gap % p
            if new_r != r:
                remove_prime(p, r)
                add_prime(p, new_r)
                new_L = get_L()
                if new_L > cur_L:
                    improved = True
                    cur_L = new_L
                    if cur_L > best_L:
                        best_L = cur_L
                        best_classes = [list(c) for c in classes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    remove_prime(p, new_r)
                    add_prime(p, r)
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Strategy 2: Random Residue Change
        if not improved and classes and random.random() < 0.5:
            idx = random.randint(0, len(classes) - 1)
            p, r = classes[idx]
            new_r = random.randint(0, p-1)
            if new_r != r:
                remove_prime(p, r)
                add_prime(p, new_r)
                new_L = get_L()
                if new_L > cur_L:
                    improved = True
                    cur_L = new_L
                    if cur_L > best_L:
                        best_L = cur_L
                        best_classes = [list(c) for c in classes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    remove_prime(p, new_r)
                    add_prime(p, r)
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif not improved:
             no_improve_count += 1

        # Strategy 3: Prime Swap
        if not improved and classes:
            current_sum = sum(1.0/p for p, r in classes)
            slack = 1.0 - current_sum
            
            if slack > 0.1:
                redundant_idx = -1
                for i, (p, r) in enumerate(classes):
                    remove_prime(p, r)
                    temp_L = get_L()
                    add_prime(p, r)
                    if temp_L >= cur_L:
                        redundant_idx = i
                        break
                
                if redundant_idx != -1:
                    p_old, r_old = classes[redundant_idx]
                    remove_prime(p_old, r_old)
                    
                    used_primes = set(p for p, r in classes)
                    candidates = [p for p in primes if p not in used_primes and 1.0/p <= slack + 1.0/p_old]
                    
                    if candidates:
                        p_new = random.choice(candidates)
                        r_new = random.randint(0, p_new-1)
                        add_prime(p_new, r_new)
                        
                        new_L = get_L()
                        if new_L > cur_L:
                            improved = True
                            cur_L = new_L
                            if cur_L > best_L:
                                best_L = cur_L
                                best_classes = [list(c) for c in classes]
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            remove_prime(p_new, r_new)
                            add_prime(p_old, r_old)
                            no_improve_count += 1
                    else:
                        add_prime(p_old, r_old)
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    total_sum = Fraction(0)
    for p, r in best_classes:
        total_sum += Fraction(1, p)
        
    if total_sum > 1:
        while total_sum > 1 and best_classes:
            p, r = best_classes.pop()
            total_sum -= Fraction(1, p)
            
    return best_classes
