import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    candidates = list(range(2, 129))
    
    def evaluate(seq):
        """
        Returns (is_valid, length, collision_count)
        """
        if not seq:
            return True, 0, 0
        n = len(seq)
        seen = set()
        collisions = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in seen:
                    collisions += 1
                else:
                    seen.add(prod)
        is_valid = (collisions == 0)
        return is_valid, n, collisions

    def score(seq):
        is_valid, n, coll = evaluate(seq)
        if is_valid:
            return n
        else:
            # Penalize invalid sequences
            # We want to move towards validity.
            # If we are invalid, we want to reduce collisions.
            # If we are valid, we want to increase n.
            # A simple metric: n - 100*coll
            # But if we jump from valid (n=20, score=20) to invalid (n=21, coll=1, score=21-100=-79), we reject.
            # This is good, it keeps us valid.
            # But what if we need to go invalid to get to a better valid state?
            # SA allows negative delta with probability exp(delta/T).
            # So if T is high, we can accept -79 from 20.
            return n - 100 * coll

    # 1. Greedy Baseline
    def greedy():
        seq = []
        for x in candidates:
            test_seq = seq + [x]
            is_valid, _, _ = evaluate(test_seq)
            if is_valid:
                seq.append(x)
        return seq

    best_seq = greedy()
    best_score = score(best_seq)
    
    # 2. Simulated Annealing
    current_seq = best_seq[:]
    current_score = best_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute set for O(1) lookup
    current_set = set(current_seq)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r = random.random()
        if r < 0.33:
            # Add
            available = [x for x in candidates if x not in current_set]
            if not available:
                continue
            x = random.choice(available)
            new_seq = current_seq + [x]
            new_seq.sort()
        elif r < 0.66:
            # Remove
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = current_seq[idx]
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        else:
            # Swap
            if not current_seq:
                continue
            available = [x for x in candidates if x not in current_set]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            a = current_seq[idx]
            b = random.choice(available)
            new_seq = current_seq[:idx] + [b] + current_seq[idx+1:]
            new_seq.sort()
            
        new_score = score(new_seq)
        
        delta = new_score - current_score
        if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_seq = new_seq
            current_set = set(current_seq)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
        
        T *= cooling
        
    # Ensure the returned sequence is strictly valid
    is_valid, _, _ = evaluate(best_seq)
    if not is_valid:
        # Fallback
        best_seq = greedy()
        
    return best_seq
