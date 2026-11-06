import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    N = 512
    MAX_A = N - 1  # 511
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        p = 2
        while p * p <= x:
            while x % p == 0:
                prime_factors[a].add(p)
                x //= p
            p += 1
        if x > 1:
            prime_factors[a].add(x)
    
    # Precompute compatibility matrix for speed
    # compat[a] is a set of b such that gcd(a,b)==1
    compat = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            if not (prime_factors[a] & prime_factors[b]):
                compat[a].add(b)
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(0) for _ in range(N)]
    for a in range(1, N):
        weights[a] = Fraction(1, N - a)
    
    # Precompute sorted candidate lists by weight (descending)
    sorted_candidates = list(range(MAX_A, 0, -1))
    
    def score(S):
        s = Fraction(0)
        for a in S:
            s += weights[a]
        return s
    
    best_S = set()
    best_score = Fraction(0)
    
    # Greedy initialization with randomness
    def greedy_init():
        S = set()
        # Try different randomization strategies
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            # Start with random subset of high-weight elements
            candidates = list(range(MAX_A, 0, -1))
            random.shuffle(candidates)
            seed_count = random.randint(3, 8)
            for a in candidates[:seed_count]:
                if S.issubset(compat[a]):
                    S.add(a)
        elif strategy == 1:
            # Start with the top few by weight
            for a in sorted_candidates[:random.randint(2, 5)]:
                if S.issubset(compat[a]):
                    S.add(a)
        else:
            # Random selection from all candidates
            candidates = list(range(1, MAX_A + 1))
            random.shuffle(candidates)
            for a in candidates[:random.randint(5, 15)]:
                if S.issubset(compat[a]):
                    S.add(a)
        
        # Then greedily add remaining in order of weight
        for a in sorted_candidates:
            if a in S:
                continue
            if S.issubset(compat[a]):
                S.add(a)
        return S
    
    # Local Search with multiple move types
    def local_search(S):
        current_score = score(S)
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and iteration < max_iterations and (time.time() - start_time) < time_limit - 5.0:
            improved = False
            iteration += 1
            
            # Try 1-opt: single swaps (remove one, add one)
            for a_out in list(S):
                best_a_in = None
                best_gain = Fraction(0)
                S_minus = S - {a_out}
                for a_in in sorted_candidates:
                    if a_in in S:
                        continue
                    if S_minus.issubset(compat[a_in]):
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.remove(a_out)
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
                    break
            
            # Try adds (if set can grow)
            if not improved:
                best_a_in = None
                best_gain = Fraction(0)
                for a_in in sorted_candidates:
                    if a_in in S:
                        continue
                    if S.issubset(compat[a_in]):
                        gain = weights[a_in]
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
            
            # Try 2-opt: swap two elements out, one in
            if not improved:
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        a_out1 = S_list[i]
                        a_out2 = S_list[j]
                        S_minus = S - {a_out1, a_out2}
                        best_a_in = None
                        best_gain = Fraction(0)
                        for a_in in sorted_candidates:
                            if a_in in S:
                                continue
                            if S_minus.issubset(compat[a_in]):
                                gain = weights[a_in] - weights[a_out1] - weights[a_out2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_a_in = a_in
                        if best_a_in is not None:
                            S.remove(a_out1)
                            S.remove(a_out2)
                            S.add(best_a_in)
                            current_score += best_gain
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 3-opt: remove one, add two (FIXED: check compatibility between new elements)
            if not improved:
                for a_out in list(S):
                    S_minus = S - {a_out}
                    best_pair = None
                    best_gain = Fraction(0)
                    candidates = [a for a in sorted_candidates if a not in S and S_minus.issubset(compat[a])]
                    for i in range(len(candidates)):
                        a_in1 = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            a_in2 = candidates[j]
                            # FIXED: Check that a_in1 and a_in2 are compatible with each other
                            if a_in1 in compat[a_in2]:
                                gain = weights[a_in1] + weights[a_in2] - weights[a_out]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (a_in1, a_in2)
                    if best_pair is not None:
                        S.remove(a_out)
                        S.add(best_pair[0])
                        S.add(best_pair[1])
                        current_score += best_gain
                        improved = True
                        break
            
            # Try 2-opt swap: remove two, add two
            if not improved:
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        a_out1 = S_list[i]
                        a_out2 = S_list[j]
                        S_minus = S - {a_out1, a_out2}
                        best_pair = None
                        best_gain = Fraction(0)
                        candidates = [a for a in sorted_candidates if a not in S and S_minus.issubset(compat[a])]
                        for i2 in range(len(candidates)):
                            a_in1 = candidates[i2]
                            for j2 in range(i2 + 1, len(candidates)):
                                a_in2 = candidates[j2]
                                if a_in1 in compat[a_in2]:
                                    gain = weights[a_in1] + weights[a_in2] - weights[a_out1] - weights[a_out2]
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_pair = (a_in1, a_in2)
                        if best_pair is not None:
                            S.remove(a_out1)
                            S.remove(a_out2)
                            S.add(best_pair[0])
                            S.add(best_pair[1])
                            current_score += best_gain
                            improved = True
                            break
                    if improved:
                        break
        
        return S, current_score
    
    # Run multiple restarts
    num_restarts = 0
    while (time.time() - start_time) < time_limit - 10.0:
        num_restarts += 1
        S = greedy_init()
        S, sc = local_search(S)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
            
    # Final polish on best_S with extended local search
    S = set(best_S)
    S, sc = local_search(S)
    if sc > best_score:
        best_score = sc
        best_S = set(S)
        
    return list(best_S)
