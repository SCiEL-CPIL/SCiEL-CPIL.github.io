import random
import time
import math
from collections import defaultdict

def construct():
    MAX_VAL = 128
    MIN_VAL = 2
    CANDIDATES = list(range(MIN_VAL, MAX_VAL + 1))
    N_CANDS = len(CANDIDATES)
    
    def get_metrics(seq):
        """
        Returns (length, collision_count).
        """
        if not seq:
            return 0, 0
        n = len(seq)
        prods = set()
        collision_count = 0
        # We need to count how many times a product appears.
        # If a product appears k times, it contributes k-1 to the collision count?
        # Or is a "collision" a pair?
        # The problem says "Products ... must be distinct".
        # So if a product appears twice, it's invalid.
        # For the penalty, we want to measure "how far" it is from valid.
        # Let's define collision_count as the number of products that appear more than once.
        # Or the total number of duplicate occurrences?
        # Let's use: sum over all products p of max(0, count(p) - 1).
        # This is the number of elements we would need to remove/change to make all products unique?
        # Not exactly, but it's a good metric.
        
        # To be efficient, we can just count duplicates.
        seen = {}
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in seen:
                    collision_count += 1
                else:
                    seen[p] = 1
        return n, collision_count

    def score(seq):
        """
        Higher is better.
        """
        length, colls = get_metrics(seq)
        # We want valid (colls=0) and long.
        # If colls > 0, we penalize heavily.
        # Penalty: 1000 * colls
        # So valid seq of len L scores L.
        # Invalid seq of len L and colls C scores L - 1000*C.
        # This ensures we prioritize validity.
        return length - 1000 * colls

    def random_initial_seq():
        # Start with a small random valid sequence
        # Greedy pick
        seq = []
        cands = CANDIDATES[:]
        random.shuffle(cands)
        for x in cands:
            new_seq = sorted(seq + [x])
            l, c = get_metrics(new_seq)
            if c == 0:
                seq = new_seq
            if len(seq) >= 20: # Get a decent start
                break
        return seq

    # Simulated Annealing
    start_time = time.time()
    time_limit = 55.0
    
    current_seq = random_initial_seq()
    current_score = score(current_seq)
    
    best_seq = current_seq[:]
    best_score = current_score
    
    T = 100.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    
    unused_set = set(CANDIDATES) - set(current_seq)
    used_set = set(current_seq)
    
    # Precompute indices for candidates?
    # We just need to pick random items.
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        move_type = random.random()
        new_seq = current_seq[:]
        
        if move_type < 0.3 and unused_set:
            # Add
            x = random.choice(list(unused_set))
            new_seq.append(x)
            new_seq.sort()
        elif move_type < 0.6 and current_seq:
            # Remove
            x = random.choice(current_seq)
            new_seq.remove(x)
        elif move_type < 0.9 and current_seq and unused_set:
            # Swap
            x = random.choice(current_seq)
            y = random.choice(list(unused_set))
            new_seq.remove(x)
            new_seq.append(y)
            new_seq.sort()
        else:
            # Stay (or do nothing)
            new_seq = current_seq[:]
            
        new_score = score(new_seq)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Accept
            current_seq = new_seq
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < -1000: # Too bad, skip calculation
                prob = 0.0
            else:
                prob = math.exp(delta / T)
            if random.random() < prob:
                current_seq = new_seq
                current_score = new_score
        
        # Update sets
        used_set = set(current_seq)
        unused_set = set(CANDIDATES) - used_set
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat or restart?
            # Let's reheat slightly to escape local minima
            T = 50.0

    return best_seq
