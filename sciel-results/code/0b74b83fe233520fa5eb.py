"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    # Start with a random set of 12 distinct integers in [0, 128]
    current_set = set(random.sample(range(129), 12))
    best_set = set(current_set)
    best_score = 0  # We want to maximize the number of collisions (bad for Sidon)
    
    def count_collisions(s):
        """Count the number of pairs (a,b), (c,d) with a+b=c+d, a<b, c<d, and {a,b} != {c,d}"""
        sums = {}
        for a, b in combinations(sorted(s), 2):
            s_val = a + b
            if s_val in sums:
                sums[s_val].append((a, b))
            else:
                sums[s_val] = [(a, b)]
        
        count = 0
        for s_val, pairs in sums.items():
            n = len(pairs)
            if n > 1:
                count += n * (n - 1) // 2
        return count
    
    # Also count diagonal collisions: a+a = b+b means a=b, but since elements are distinct, 
    # the only diagonal collision is when a+a = b+c with a != b, a != c, b != c.
    # Wait, the problem says "all unordered pair sums unique INCLUDING diagonals".
    # This means for a subset S, the set of sums {x+y : x,y in S, x <= y} must have all distinct values.
    # So if a < b < c < d and a+d = b+c, that's a collision.
    # Also if a < b and a+a = b+b, that's impossible since a != b.
    # But a+a = b+c is a collision (diagonal a,a vs off-diagonal b,c).
    
    def count_all_collisions(s):
        """Count collisions in the set of all pair sums including diagonals."""
        sums = {}
        sorted_s = sorted(s)
        for i, a in enumerate(sorted_s):
            for j, b in enumerate(sorted_s):
                if i <= j:
                    s_val = a + b
                    if s_val in sums:
                        sums[s_val] += 1
                    else:
                        sums[s_val] = 1
        
        count = 0
        for s_val, cnt in sums.items():
            if cnt > 1:
                count += cnt * (cnt - 1) // 2
        return count
    
    best_score = count_all_collisions(current_set)
    
    # Stochastic local search
    while time.time() < deadline:
        # Try to improve by swapping one element
        if random.random() < 0.5:
            # Swap an element with a random new one
            old_elem = random.choice(list(current_set))
            new_elem = random.choice([x for x in range(129) if x not in current_set])
            
            trial_set = current_set - {old_elem} | {new_elem}
            trial_score = count_all_collisions(trial_set)
            
            if trial_score > best_score:
                best_set = set(trial_set)
                best_score = trial_score
                current_set = trial_set
            elif trial_score > count_all_collisions(current_set) or random.random() < 0.1:
                current_set = trial_set
        else:
            # Add and remove two elements
            old_elem1 = random.choice(list(current_set))
            old_elem2 = random.choice([x for x in current_set if x != old_elem1])
            new_elem1 = random.choice([x for x in range(129) if x not in current_set])
            new_elem2 = random.choice([x for x in range(129) if x not in current_set and x != new_elem1])
            
            trial_set = current_set - {old_elem1, old_elem2} | {new_elem1, new_elem2}
            trial_score = count_all_collisions(trial_set)
            
            if trial_score > best_score:
                best_set = set(trial_set)
                best_score = trial_score
                current_set = trial_set
            elif trial_score > count_all_collisions(current_set) or random.random() < 0.1:
                current_set = trial_set
    
    # Return the best set found
    return list(best_set)
# EVOLVE-BLOCK-END
