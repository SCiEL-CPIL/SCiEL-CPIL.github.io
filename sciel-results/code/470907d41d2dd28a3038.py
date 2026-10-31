import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit, 5s buffer)

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        if not U or 0 not in U:
            return -1
        U = sorted(set(U))
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        
        # Direct counting for small n is faster and exact
        if n <= 20:
            sums = set()
            diffs = set()
            for i in range(n):
                ui = U[i]
                for j in range(n):
                    uj = U[j]
                    sums.add(ui + uj)
                    diffs.add(ui - uj)
            n_sum = len(sums)
            n_diff = len(diffs)
        else:
            # FFT for larger n
            min_N = 2 * max_u + 1
            # Find next power of 2 for FFT speed
            N = 1
            while N < min_N:
                N *= 2
            
            f = np.zeros(N)
            for u in U:
                f[u] = 1.0
                
            F = np.fft.fft(f)
            conv_sum = np.fft.ifft(F * F)
            n_sum = np.sum(np.abs(conv_sum.real) > 0.5)
            
            f_neg = np.zeros(N)
            for u in U:
                idx = (-u) % N
                f_neg[idx] = 1.0
            F_neg = np.fft.fft(f_neg)
            conv_diff = np.fft.ifft(F * F_neg)
            n_diff = np.sum(np.abs(conv_diff.real) > 0.5)
            
        if n_sum == 0:
            return -1
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def update_best(cand):
        nonlocal best_score, best_set
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Initial Candidates
    # Small exhaustive search
    for n in range(2, 6):
        # Limit range to keep time low
        max_val = 30
        for combo in combinations(range(1, max_val + 1), n - 1):
            cand = [0] + list(combo)
            update_best(cand)
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break

    # Structured initializations
    # Powers of 2
    for k in range(2, 10):
        U = [0]
        v = 1
        for _ in range(k - 1):
            U.append(v)
            v *= 2
        update_best(U)

    # Geometric progressions
    for b in range(2, 8):
        for k in range(2, 10):
            U = [0]
            v = 1
            for _ in range(k - 1):
                U.append(v)
                v *= b
            update_best(U)
            if time.time() > deadline - 10:
                break

    # Phase 2: Simulated Annealing / Hill Climbing
    def mutate(U):
        U = list(U)
        r = random.random()
        if r < 0.1:
            # Add element
            max_v = max(U)
            new_v = random.randint(1, max_v + 15)
            if new_v not in U:
                U.append(new_v)
        elif r < 0.2 and len(U) > 2:
            # Remove element
            non_zero = [x for x in U if x != 0]
            if non_zero:
                rem = random.choice(non_zero)
                U.remove(rem)
        else:
            # Replace element
            non_zero = [x for x in U if x != 0]
            if non_zero:
                idx = random.choice(non_zero)
                old_v = U[idx]
                # Local move or Jump
                if random.random() < 0.2:
                    new_v = random.randint(1, max(U) + 20)
                else:
                    lower = max(1, old_v - 5)
                    upper = old_v + 5
                    new_v = random.randint(lower, upper)
                
                # Avoid duplicates
                if new_v in U and new_v != old_v:
                    # Try a few times to find a unique value
                    for _ in range(5):
                        new_v = random.randint(max(1, old_v - 10), old_v + 10)
                        if new_v not in U or new_v == old_v:
                            break
                
                U[idx] = new_v
        
        return sorted(set(U))

    curr = list(best_set)
    curr_score = best_score
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    no_improve = 0
    
    while time.time() < deadline - 2:
        if temp < min_temp:
            # Restart annealing from best found
            temp = 1.0
            curr = list(best_set)
            curr_score = best_score
            no_improve = 0
            
        neighbor = mutate(curr)
        n_score = compute_score(neighbor)
        
        if n_score > curr_score:
            curr = neighbor
            curr_score = n_score
            no_improve = 0
            if curr_score > best_score:
                best_score = curr_score
                best_set = list(curr)
        else:
            # Accept worse with probability exp((new-old)/temp)
            delta = n_score - curr_score
            if delta > 0:
                prob = 1.0
            else:
                prob = math.exp(delta / temp)
            
            if random.random() < prob:
                curr = neighbor
                curr_score = n_score
            else:
                no_improve += 1
                
        temp *= cool_rate

    return list(best_set)
