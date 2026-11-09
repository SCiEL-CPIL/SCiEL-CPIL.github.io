import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leaving 10s margin
    
    max_val = 4096
    max_size = 80
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed=None, skip_first_k=0, bias_factor=1.0):
        """
        Runs a greedy algorithm to build a B_3 set.
        If seed is not None, it shuffles the candidate list with that seed.
        This allows exploring different orderings to escape local minima of the standard greedy.
        """
        current_set = []
        sum_set = set()
        
        # Create candidate list
        candidates = list(range(1, max_val + 1))
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(candidates)
        else:
            # Standard increasing order
            pass
            
        # Apply bias? The prompt suggests keeping the design family. 
        # A "weighted" greedy where we prefer smaller numbers generally is best for packing.
        # But if we shuffle, we lose that. 
        # Let's stick to: 
        # 1. Deterministic increasing greedy (fast, baseline)
        # 2. Randomized shuffles (to find better packings)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check if x can be added
            # New sums involving x:
            # 1. (x, x, x) -> 3x
            # 2. (x, x, y) -> 2x + y, for y in current_set
            # 3. (x, y, y) -> x + 2y, for y in current_set
            # 4. (x, y, z) -> x + y + z, for y < z in current_set
            
            # Optimization: Early exit if 3x is already in sum_set
            if 3 * x in sum_set:
                continue
                
            # Generate new sums
            # We can do this in a list and then check for duplicates and conflicts
            
            # To speed up, we can check conflicts on the fly?
            # No, checking internal collisions in new_sums is also required.
            # However, for a fixed x, the sums 3x, 2x+y, x+2y, x+y+z are distinct 
            # unless there are coincidences.
            # e.g., 2x + y1 == 2x + y2 implies y1 == y2 (distinct y's, so no).
            # 2x + y1 == x + 2y2 => x + y1 == 2y2. Possible.
            # So we must check internal collisions.
            
            new_sums = [3 * x]
            
            for y in current_set:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
                
            for y, z in combinations(current_set, 2):
                new_sums.append(x + y + z)
                
            # Check internal collisions
            # If len(set) != len(list), there is a collision among the new sums.
            # This is expensive for large sets. 
            # But we can optimize: 
            # The number of new sums is 1 + 2k + k(k-1)/2 ~ k^2/2.
            # If k is ~40, that's ~800 sums. Set creation is fast.
            
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                continue
                
            # Check conflicts with existing sums
            if not new_sums_set.isdisjoint(sum_set):
                continue
                
            # Accept x
            current_set.append(x)
            sum_set.update(new_sums_set)
            
            if len(current_set) >= max_size:
                break
                
        return current_set

    # 1. Run deterministic increasing greedy
    set1 = run_greedy()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
        
    # 2. Try randomized greedy with multiple seeds
    # How many seeds can we try?
    # Each run takes O(N * K^2) roughly? 
    # N=4096, K~40. 
    # Inner loop: for each x (4096), check sums (K^2 ~ 1600).
    # Total ops ~ 4096 * 1600 = 6.5M per run.
    # Python can do ~10-50M simple ops/sec. So ~0.1-0.5s per run.
    # We have 50s. We can try ~100-500 runs.
    
    # Let's use a fixed number of seeds to be safe, or loop until time runs out.
    # We will use a loop that generates seeds and runs, checking time.
    
    num_random_trials = 0
    base_seed = 12345
    
    while time.time() < deadline - 2.0:  # Leave 2s margin for final processing
        seed = base_seed + num_random_trials
        s = run_greedy(seed=seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # If we hit the theoretical max or close to it, we could stop?
            # Theoretical max is ~42. If we hit 42, we can't do better (likely).
            if best_size >= 42:
                break
        num_random_trials += 1
        
    return best_set

# Helper to verify (not used in construct, but good for debugging if needed)
# def is_valid_b3(s):
#     sums = set()
#     for a in range(len(s)):
#         for b in range(a, len(s)):
#             for c in range(b, len(s)):
#                 val = s[a] + s[b] + s[c]
#                 if val in sums:
#                     return False
#                 sums.add(val)
#     return True
