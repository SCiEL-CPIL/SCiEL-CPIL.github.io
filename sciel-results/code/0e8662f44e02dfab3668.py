import random
import itertools
import time

def is_sidon(S):
    # Check if a set/list S is Sidon (all pair sums i<=j unique)
    # S is a tuple of distinct integers
    k = len(S)
    sums = set()
    for i in range(k):
        for j in range(i, k):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_k(S, k):
    count = 0
    for subset in itertools.combinations(S, k):
        if is_sidon(subset):
            count += 1
    return count

def score(S):
    # Minimize this score
    # Priority: Avoid 5-Sidon sets, then avoid 4-Sidon sets
    c5 = count_sidon_k(S, 5)
    if c5 > 0:
        return 1000 * c5 + count_sidon_k(S, 4)
    else:
        return count_sidon_k(S, 4)

def construct():
    start_time = time.time()
    best_S = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    # Random starts
    for _ in range(50):
        S = random.sample(range(0, 129), 12)
        candidates.append(S)
    
    # Structured starts (e.g., arithmetic progression, powers of 2, etc.)
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, ... sum fits in 128?
    # 1+2+4+8+16+32+64 = 127. 12 elements?
    # 2^0..2^10 is 1024, too big.
    # Maybe sparse sets.
    
    for restart in range(200):
        if time.time() - start_time > 55:
            break
            
        if restart < len(candidates):
            S = candidates[restart][:]
        else:
            S = random.sample(range(0, 129), 12)
            
        current_score = score(S)
        
        if current_score < best_score:
            best_score = current_score
            best_S = S[:]
            
        if best_score == 0:
            return best_S
            
        # Local Search
        for step in range(2000):
            if time.time() - start_time > 55:
                break
                
            if current_score == 0:
                return S
                
            # Find a problematic subset
            # We want to find a 5-Sidon or 4-Sidon set to break
            # If 5-Sidon exists, it's higher priority.
            target_subset = None
            target_k = 0
            
            # Check 5-Sidon first
            for sub in itertools.combinations(S, 5):
                if is_sidon(sub):
                    target_subset = sub
                    target_k = 5
                    break
                    
            if target_subset is None:
                # Check 4-Sidon
                for sub in itertools.combinations(S, 4):
                    if is_sidon(sub):
                        target_subset = sub
                        target_k = 4
                        break
                        
            if target_subset is None:
                return S # Should not happen if score > 0
                
            # Pick a random element from the target subset to move
            # Or pick the one that seems most "central" to the collisions?
            # Random is fine.
            idx = random.randint(0, 11)
            # Ensure idx is in target_subset
            # Map values to indices in S
            val_to_idx = {v: i for i, v in enumerate(S)}
            target_vals = set(target_subset)
            possible_idxs = [val_to_idx[v] for v in target_vals]
            idx = random.choice(possible_idxs)
            
            old_val = S[idx]
            
            # Try to move to a new value
            improved = False
            # Try a few random new values
            # Also try values close to old_val? Or just random?
            # Random is safer to escape local minima.
            attempts = 0
            while attempts < 20:
                new_val = random.randint(0, 128)
                if new_val == old_val:
                    attempts += 1
                    continue
                if new_val in S:
                    attempts += 1
                    continue
                    
                S[idx] = new_val
                new_score = score(S)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_S = S[:]
                    improved = True
                    break
                else:
                    S[idx] = old_val
                attempts += 1
                
            if not improved:
                # Stuck, do a random kick
                idx = random.randint(0, 11)
                S[idx] = random.randint(0, 128)
                while len(set(S)) < 12:
                    S[idx] = random.randint(0, 128)
                current_score = score(S)
                
    return best_S
