"""Rewritten program using Simulated Annealing to optimize the set of pairwise coprime integers."""
# EVOLVE-BLOCK-START
import math
import random

def construct():
    MAX_VAL = 511
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        if not factors[i]: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                factors[j].add(i)
                
    # Precompute values
    values = [0.0] * (MAX_VAL + 1)
    for a in range(1, MAX_VAL + 1):
        values[a] = 1.0 / (N - a)

    def get_score(S):
        return sum(values[a] for a in S)

    # 1. Generate initial greedy solution
    S = []
    for a in range(MAX_VAL, 0, -1):
        fa = factors[a]
        valid = True
        for b in S:
            if fa & factors[b]:
                valid = False
                break
        if valid:
            S.append(a)
            
    current_score = get_score(S)
    best_S = list(S)
    best_score = current_score
    
    # 2. Simulated Annealing
    start_temp = 0.01
    end_temp = 0.0001
    cooling_rate = 0.999
    max_iterations = 50000
    
    current_temp = start_temp
    S_set = set(S)
    
    for i in range(max_iterations):
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            a = random.randint(1, MAX_VAL)
            if a not in S_set:
                fa = factors[a]
                valid = True
                for b in S_set:
                    if fa & factors[b]:
                        valid = False
                        break
                if valid:
                    new_S_set = S_set | {a}
                    delta = values[a]
                    new_score = current_score + delta
                    
                    if delta > 0 or random.random() < math.exp(delta / current_temp):
                        S_set = new_S_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_S = list(S_set)
                            
        elif move_type == 'remove':
            if S_set:
                a = random.choice(list(S_set))
                new_S_set = S_set - {a}
                delta = -values[a]
                new_score = current_score + delta
                
                if delta > 0 or random.random() < math.exp(delta / current_temp):
                    S_set = new_S_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = list(S_set)
                        
        else: # swap
            if S_set:
                a = random.choice(list(S_set))
                b = random.randint(1, MAX_VAL)
                if b not in S_set and b != a:
                    fb = factors[b]
                    valid = True
                    for c in S_set:
                        if c == a:
                            continue
                        if fb & factors[c]:
                            valid = False
                            break
                    if valid:
                        new_S_set = (S_set - {a}) | {b}
                        delta = values[b] - values[a]
                        new_score = current_score + delta
                        
                        if delta > 0 or random.random() < math.exp(delta / current_temp):
                            S_set = new_S_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_S = list(S_set)
        
        current_temp *= cooling_rate
        if current_temp < end_temp:
            break
            
    return best_S
# EVOLVE-BLOCK-END
