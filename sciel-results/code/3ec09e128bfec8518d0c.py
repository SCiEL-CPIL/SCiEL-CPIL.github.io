import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Generate candidate pool: 2^i * 3^j <= 10000
    candidates = set()
    for i in range(14):
        val_i = 2 ** i
        if val_i > 10000:
            break
        for j in range(10):
            val = val_i * (3 ** j)
            if val > 10000:
                break
            candidates.add(val)
    
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    n = 20
    best_set = None
    best_score = float('inf')
    
    def compute_score(A):
        sums = set()
        prods = set()
        A_list = list(A)
        for i in range(n):
            for j in range(i, n):
                sums.add(A_list[i] + A_list[j])
                prods.add(A_list[i] * A_list[j])
        return max(len(sums), len(prods))
        
    # Try a few random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize A with 20 random candidates
        # To help sumset, maybe pick from a smaller subset? 
        # Let's just pick 20 random from the full candidate list.
        # Or better, pick 20 from a random subset of exponents to ensure variety but not too spread out?
        # Actually, just random 20 from candidate_list is a good start.
        
        A = set(random.sample(candidate_list, n))
        current_score = compute_score(A)
        
        if current_score < best_score:
            best_score = current_score
            best_set = set(A)
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle A to randomize the order of checking
            A_list = list(A)
            random.shuffle(A_list)
            
            for a in A_list:
                if time.time() > deadline:
                    break
                # Try swapping a with b
                # To be efficient, we can just iterate through a subset of candidates
                # But candidate_list is small (~100), so iterating all is fine.
                for b in candidate_list:
                    if b in A:
                        continue
                    
                    new_A = set(A)
                    new_A.remove(a)
                    new_A.add(b)
                    
                    new_score = compute_score(new_A)
                    
                    if new_score < current_score:
                        A = new_A
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = set(A)
                        # After an improvement, restart the inner loop? 
                        # Or just continue? Continuing is usually fine for hill climbing.
                        break # Restart inner loop with new A? 
                        # Actually, breaking the 'for b' loop and continuing 'for a' is standard.
                        # But since A changed, the 'for a' loop is iterating over old A_list.
                        # It's safer to restart the 'for a' loop or just continue and accept that 'a' is stale.
                        # Given small N, let's just break and let the outer 'while' handle it?
                        # No, 'improved' is set to True, so 'while' continues.
                        # But we are still in 'for a'. 
                        # Let's just break out of 'for b' and continue 'for a'.
                        # The next 'a' might be the one we just removed or added? 
                        # It doesn't matter much, we'll check swaps again.
        
    if best_set is None:
        return list(range(1, 21))
        
    return list(best_set)
