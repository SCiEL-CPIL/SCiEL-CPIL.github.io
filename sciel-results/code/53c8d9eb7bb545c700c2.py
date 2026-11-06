import itertools
import random
import time

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Check sizes from 12 down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def collision_score(S):
    """
    Counts the number of pairs of pairs with the same sum.
    This serves as a proxy for 'hardness' against Sidon extraction.
    """
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    
    score = 0
    for count in sums.values():
        score += count * (count - 1) // 2
    return score

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 55  # Leave some margin for evaluation overhead
    
    # Initial random set
    S = random.sample(range(0, 129), 12)
    current_score = collision_score(S)
    
    best_S = S[:]
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try local moves
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            i = random.randint(0, 11)
            old_val = S[i]
            
            # Pick a new value
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            if new_val in S:
                continue
            
            # Calculate new score
            S[i] = new_val
            new_score = collision_score(S)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
            else:
                # Accept equal score moves with some probability to escape plateaus
                if new_score == current_score and random.random() < 0.05:
                    current_score = new_score
                    improved = True
                else:
                    S[i] = old_val
        
        if not improved:
            # Perturbation: change 2 random elements
            idx1, idx2 = random.sample(range(12), 2)
            # Ensure new values are distinct and not in the rest of the set
            remaining = [x for j, x in enumerate(S) if j != idx1 and j != idx2]
            valid_vals = [v for v in range(129) if v not in remaining]
            if len(valid_vals) >= 2:
                v1, v2 = random.sample(valid_vals, 2)
                S[idx1] = v1
                S[idx2] = v2
            else:
                # Fallback: just random
                S[idx1] = random.randint(0, 128)
                S[idx2] = random.randint(0, 128)
                # Fix distinctness
                while len(set(S)) < 12:
                    idx = random.randint(0, 11)
                    S[idx] = random.randint(0, 128)
            
            current_score = collision_score(S)
            if current_score > best_score:
                best_score = current_score
                best_S = S[:]

    return best_S
