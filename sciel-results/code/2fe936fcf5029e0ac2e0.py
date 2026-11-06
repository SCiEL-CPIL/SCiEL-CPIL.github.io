import time
import random
import sys

# Constants
N = 32
MAX_VAL = 512
TIME_LIMIT = 140.0 # Safety margin from 150s

def construct():
    start_time = time.time()
    
    # Precompute valid indices
    indices = list(range(1, MAX_VAL + 1))
    
    best_score = -1
    best_set = None
    
    # Optimized score calculation
    def score_set_fast(S):
        S_set = set(S)
        covered = set()
        # Iterate over all pairs a < b
        # We can optimize by pre-sorting? No, set is fine.
        # O(N^2) lookups.
        for a in S:
            for b in S:
                if b <= a: continue
                d = b - a
                # Check if d is covered
                # 1. a is middle: a-d in S? (a-d, a, a+d=b)
                if (a - d) in S_set:
                    covered.add(d)
                # 2. a is start: a+2d in S? (a, a+d=b, a+2d)
                if (b + d) in S_set:
                    covered.add(d)
                # 3. b is end: a+d/2 in S? (a, a+d/2, a+d=b)
                if d % 2 == 0:
                    d2 = d // 2
                    if (a + d2) in S_set:
                        covered.add(d2)
        return len(covered)

    # Greedy Initialization
    def greedy_init():
        """Construct a set using a greedy approach to maximize distinct differences."""
        S = []
        S_set = set()
        # Start with a random spread
        # To maximize coverage, we want to cover small d (dense) and large d (spread).
        # A good heuristic is to start with a random set and greedily add elements
        # that cover the most NEW differences.
        
        # Initial random set
        S = random.sample(indices, N)
        S_set = set(S)
        
        # Greedy swap improvement
        improved = True
        while improved:
            improved = False
            current_score = score_set_fast(S)
            for i in range(N):
                old_val = S[i]
                best_new_val = None
                best_new_score = current_score
                # Try a few random replacements
                for _ in range(50):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in S_set:
                        continue
                    S[i] = new_val
                    S_set.remove(old_val)
                    S_set.add(new_val)
                    new_score = score_set_fast(S)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    S[i] = old_val
                    S_set.remove(new_val)
                    S_set.add(old_val)
                
                if best_new_val is not None:
                    S[i] = best_new_val
                    S_set.remove(old_val)
                    S_set.add(best_new_val)
                    current_score = best_new_score
                    improved = True
        
        return S, score_set_fast(S)

    # Simulated Annealing
    def simulated_annealing(S_init, T_init=5.0, T_min=0.01, alpha=0.999):
        S = list(S_init)
        S_set = set(S)
        current_score = score_set_fast(S)
        best_score_local = current_score
        best_S_local = list(S)
        
        T = T_init
        moves = 0
        last_improvement = time.time()
        
        while time.time() - start_time < TIME_LIMIT:
            # Pick random x_out
            idx_out = random.randint(0, N-1)
            x_out = S[idx_out]
            
            # Pick random x_in
            x_in = random.randint(1, MAX_VAL)
            if x_in in S_set:
                continue
            
            # Perform swap
            S[idx_out] = x_in
            S_set.remove(x_out)
            S_set.add(x_in)
            
            new_score = score_set_fast(S)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_S_local = list(S)
                    last_improvement = time.time()
            else:
                # Metropolis acceptance
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = pow(2.71828, delta / T)
                    if random.random() > accept_prob:
                        # Reject
                        S[idx_out] = x_out
                        S_set.remove(x_in)
                        S_set.add(x_out)
            
            T *= alpha
            if T < T_min:
                # Restart temperature
                T = T_init
                # Optionally restart with a perturbed state
                # For now, just reset T to keep exploring
                # If stuck for too long, we could restart the whole SA, 
                # but the outer loop handles that.
                # Let's just continue with high T.
            
            moves += 1
            
            # Safety break if no improvement for a long time and T is low?
            # The outer loop manages restarts.
            
        return best_S_local, best_score_local

    # Main Loop: Multiple restarts
    # We will run SA on several diverse initializations.
    
    # 1. Greedy Init
    S_greedy, score_greedy = greedy_init()
    best_score = max(best_score, score_greedy)
    best_set = S_greedy
    
    # Run SA on the greedy result
    S_sa, score_sa = simulated_annealing(S_greedy, T_init=10.0)
    if score_sa > best_score:
        best_score = score_sa
        best_set = S_sa
    
    # 2. Other diverse inits
    # Random
    S_rand = random.sample(indices, N)
    score_rand = score_set_fast(S_rand)
    if score_rand > best_score:
        best_score = score_rand
        best_set = S_rand
        
    S_sa2, score_sa2 = simulated_annealing(S_rand, T_init=10.0)
    if score_sa2 > best_score:
        best_score = score_sa2
        best_set = S_sa2
        
    # 3. Structured: Consecutive
    S_cons = list(range(1, 33))
    score_cons = score_set_fast(S_cons)
    if score_cons > best_score:
        best_score = score_cons
        best_set = S_cons
        
    S_sa3, score_sa3 = simulated_annealing(S_cons, T_init=10.0)
    if score_sa3 > best_score:
        best_score = score_sa3
        best_set = S_sa3
        
    # 4. Structured: Spread out
    S_spread = list(range(1, 513, 16))[:32] # 1, 17, 33...
    score_spread = score_set_fast(S_spread)
    if score_spread > best_score:
        best_score = score_spread
        best_set = S_spread
        
    S_sa4, score_sa4 = simulated_annealing(S_spread, T_init=10.0)
    if score_sa4 > best_score:
        best_score = score_sa4
        best_set = S_sa4
        
    return best_set

# Run a quick test to ensure it works
# print(construct())
