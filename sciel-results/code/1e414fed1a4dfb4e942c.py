import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    def calc_score(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    # Multiple starts
    best_global = None
    best_global_score = float('inf')
    
    # Start 1: Baseline
    starts = [list(range(1, 21))]
    
    # Start 2: Random small ints
    for _ in range(5):
        s = random.sample(range(1, 100), 20)
        starts.append(s)
        
    # Start 3: Geometric-ish
    s = []
    x = 1
    while len(s) < 20:
        s.append(x)
        x = int(x * 1.5) + 1
        if x > 10000: break
    if len(s) == 20:
        starts.append(s)
        
    for init_set in starts:
        current_set = init_set
        current_score = calc_score(current_set)
        
        # Local search
        while time.time() < deadline:
            improved = False
            # Try all single swaps? 
            # 20 * 100 candidates = 2000 evals per full pass.
            # 1 eval is ~1ms? 2000ms = 2s per pass.
            # We can do ~20 passes in 50s.
            
            # To be faster, random sampling of moves
            for _ in range(100): # 100 random moves per "step"
                idx = random.randint(0, 19)
                old_val = current_set[idx]
                
                # Sample from 1..100
                new_val = random.randint(1, 100)
                
                if new_val in current_set:
                    continue
                
                new_set = current_set.copy()
                new_set[idx] = new_val
                
                new_score = calc_score(new_set)
                
                if new_score < current_score:
                    current_score = new_score
                    current_set = new_set
                    improved = True
                    break # Restart sampling loop? Or continue?
                elif new_score == current_score and random.random() < 0.05:
                    current_set = new_set
                    # Don't break, keep exploring
            if not improved:
                # Stuck, try a kick
                # Replace 2 random elements with random small ints
                idx1, idx2 = random.sample(range(20), 2)
                new_set = current_set.copy()
                new_set[idx1] = random.randint(1, 50)
                new_set[idx2] = random.randint(1, 50)
                # Ensure distinct
                if new_set[idx1] == new_set[idx2]:
                    new_set[idx2] += 1
                # Check validity
                if len(set(new_set)) == 20:
                    kick_score = calc_score(new_set)
                    # Accept kick if it's not terrible, or just accept to escape
                    # Simple: accept if better or equal
                    if kick_score <= current_score:
                        current_set = new_set
                        current_score = kick_score
            
            if current_score < best_global_score:
                best_global_score = current_score
                best_global = current_set.copy()
                
    return best_global
