import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    # Candidates: 1..511
    # We want to maximize sum of 1/(512 - a) subject to pairwise coprime constraint.
    
    # 1. Precompute prime factors and conflict graph
    # Primes up to 511
    primes = []
    for i in range(2, 512):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = 0
                break
        if is_p:
            primes.append(i)
    
    # Map each number to its set of prime factors
    # We use a set of integers for fast intersection checks
    factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                factors[a].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            factors[a].add(x)
    # factors[1] is empty set, which is coprime with everything
    
    # Build conflict graph: conflict[i][j] = True if gcd(i, j) > 1
    # To save memory and time, we can just check on the fly or precompute a bitset.
    # Given N=511, a 511x511 boolean matrix is ~250k entries, which is fine.
    # But checking factors intersection is O(1) average if sets are small.
    # Let's precompute a list of conflicting neighbors for each node to speed up SA moves.
    # Actually, for SA, we need to check if a new node is compatible with ALL current nodes.
    # It is faster to maintain the set of selected nodes and check against it.
    
    candidates = list(range(1, N))
    # Weights
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # 2. Greedy Initialization
    # Sort by weight descending
    sorted_candidates = sorted(candidates, key=lambda x: weights[x], reverse=True)
    
    current_set = set()
    for a in sorted_candidates:
        # Check if a is coprime with all in current_set
        is_coprime = True
        for b in current_set:
            if factors[a] & factors[b]:
                is_coprime = False
                break
        if is_coprime:
            current_set.add(a)
            
    current_score = sum(weights[a] for a in current_set)
    best_set = set(current_set)
    best_score = current_score
    
    # 3. Simulated Annealing
    # Moves:
    # 1. Add a random candidate not in set that is compatible with all in set.
    # 2. Remove a random candidate from set.
    # 3. Swap a candidate in set with a compatible candidate not in set.
    
    # Precompute list of all candidates for random selection
    all_candidates = candidates
    
    # To make "Add" efficient, we might need to know which candidates are compatible.
    # Checking compatibility is O(|current_set| * |factors|).
    # With |current_set| ~ 20-30, this is fast.
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # Random seed for reproducibility in tests, but we want diversity.
    # random.seed(42) # Uncomment for debugging
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a random move
        move_type = random.random()
        
        new_set = set(current_set)
        delta_score = 0.0
        
        if move_type < 0.4:
            # Try to Add
            # Pick a random candidate not in current_set
            candidates_to_add = [a for a in all_candidates if a not in current_set]
            if not candidates_to_add:
                continue
            a = random.choice(candidates_to_add)
            
            # Check compatibility
            compatible = True
            for b in current_set:
                if factors[a] & factors[b]:
                    compatible = False
                    break
            
            if compatible:
                new_set.add(a)
                delta_score = weights[a]
            else:
                continue # Invalid move, retry or skip. Skipping is fine for SA.
                
        elif move_type < 0.7:
            # Try to Remove
            if not current_set:
                continue
            a = random.choice(list(current_set))
            new_set.remove(a)
            delta_score = -weights[a]
            
        else:
            # Try to Swap
            # Pick one to remove
            if not current_set:
                continue
            a_remove = random.choice(list(current_set))
            
            # Pick one to add
            candidates_to_add = [a for a in all_candidates if a not in current_set]
            if not candidates_to_add:
                continue
            a_add = random.choice(candidates_to_add)
            
            # Check if a_add is compatible with (current_set - {a_remove})
            temp_set = set(current_set)
            temp_set.remove(a_remove)
            
            compatible = True
            for b in temp_set:
                if factors[a_add] & factors[b]:
                    compatible = False
                    break
            
            if compatible:
                new_set = temp_set
                new_set.add(a_add)
                delta_score = weights[a_add] - weights[a_remove]
            else:
                continue # Invalid swap
        
        # Acceptance criterion
        if delta_score > 0:
            current_set = new_set
            current_score += delta_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            # Metropolis criterion
            # exp(delta / T)
            # delta is negative
            try:
                prob = math.exp(delta_score / T)
            except OverflowError:
                prob = 0.0
            if random.random() < prob:
                current_set = new_set
                current_score += delta_score
        
        T *= cooling_rate
        
        # Optional: restart if stuck? 
        # If we haven't improved best_score in a long time, we could restart from best_set with higher T.
        # But simple cooling is usually sufficient for this size.
        
    return list(best_set)

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    # Verify pairwise coprime
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
    print("Done")
