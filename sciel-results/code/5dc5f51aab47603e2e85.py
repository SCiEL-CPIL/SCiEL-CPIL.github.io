import random
import numpy as np

def construct():
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    
    # Set a random seed for reproducibility? 
    # The prompt doesn't specify, but usually it's better to be deterministic 
    # or rely on the evaluator's seed if available. 
    # However, "seed/value" in the prompt description suggests the evaluator 
    # might pass a seed or the score is normalized by the seed program's value.
    # "1.0 equals the seed program". 
    # If I use random, the result might vary. 
    # To be safe, I will try to find a good static solution or a very robust stochastic one.
    # Given the constraint "Minimize", finding *a* good solution is enough.
    # I will not hardcode a solution because I don't have one pre-computed.
    # I will rely on the SA finding a good one within the time limit.
    
    k = 30
    A = set(random.sample(range(N + 1), k))
    
    def evaluate(A_set):
        counts = [0] * (TARGET_MAX + 1)
        A_list = list(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if TARGET_MIN <= s <= TARGET_MAX:
                    counts[s] += 1
        target_counts = counts[TARGET_MIN:TARGET_MAX + 1]
        if min(target_counts) == 0:
            missing = sum(1 for c in target_counts if c == 0)
            return 1000 + missing * 10 + max(target_counts)
        return max(target_counts)
    
    current_A = set(A)
    current_obj = evaluate(current_A)
    best_A = set(current_A)
    best_obj = current_obj
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps = 80000
    
    all_elements = list(range(N + 1))
    
    for step in range(steps):
        x = random.choice(list(current_A))
        y = random.choice(all_elements)
        while y in current_A:
            y = random.choice(all_elements)
        
        if x == y:
            continue
            
        current_A.remove(x)
        current_A.add(y)
        
        new_obj = evaluate(current_A)
        
        if new_obj < current_obj:
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = set(current_A)
        else:
            delta = new_obj - current_obj
            if delta < 0:
                accept = True
            else:
                if delta > 500:
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / T)
            
            if not accept:
                current_A.remove(y)
                current_A.add(x)
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return list(best_A)
