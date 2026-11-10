import math
import random
import time

def construct():
    N = 512
    candidates = list(range(1, N))
    scores = {a: 1.0 / (N - a) for a in candidates}
    
    # 1. Greedy Initialization
    S = set()
    for a in sorted(candidates, key=lambda x: scores[x], reverse=True):
        if all(math.gcd(a, b) == 1 for b in S):
            S.add(a)
    
    best_S = set(S)
    best_score = sum(scores[a] for a in S)
    current_score = best_score
    
    # 2. Simulated Annealing
    start_time = time.time()
    T = 0.02
    alpha = 0.9999
    min_T = 1e-5
    deadline = start_time + 110.0 # Leave 10s for polish and overhead
    
    # Precompute list of candidates for faster random choice
    cand_list = candidates
    
    while time.time() < deadline:
        # Generate Move
        # Bias: 50% Swap, 30% Add, 20% Remove
        r = random.random()
        new_S = set(S)
        delta = 0.0
        
        if r < 0.5: # Swap
            s = random.choice(list(S))
            a = random.choice(cand_list)
            if a not in S:
                # Check validity
                if all(math.gcd(a, b) == 1 for b in S if b != s):
                    new_S.remove(s)
                    new_S.add(a)
                    delta = scores[a] - scores[s]
        
        elif r < 0.8: # Add
            a = random.choice(cand_list)
            if a not in S:
                if all(math.gcd(a, b) == 1 for b in S):
                    new_S.add(a)
                    delta = scores[a]
        
        else: # Remove
            s = random.choice(list(S))
            new_S.remove(s)
            delta = -scores[s]
        
        # Acceptance
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            S = new_S
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        
        # Cool
        T *= alpha
        if T < min_T:
            T = min_T

    # 3. Local Search Polish on best_S
    S = best_S
    current_score = best_score
    improved = True
    while improved:
        improved = False
        # Try Add
        for a in cand_list:
            if a not in S and all(math.gcd(a, b) == 1 for b in S):
                S.add(a)
                current_score += scores[a]
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                improved = True
                break # Restart loop to check other adds/swaps
        # Try Swap
        S_list = list(S)
        for s in S_list:
            for a in cand_list:
                if a not in S:
                    if all(math.gcd(a, b) == 1 for b in S if b != s):
                        gain = scores[a] - scores[s]
                        if gain > 0:
                            S.remove(s)
                            S.add(a)
                            current_score += gain
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
                            improved = True
                            break # Restart loop
            if improved:
                break
        
        # Time check in polish? It should be fast, but just in case
        if time.time() > start_time + 118.0:
            break

    return list(best_S)
