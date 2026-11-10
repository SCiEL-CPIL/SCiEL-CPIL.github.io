import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # Initial solution: Powers of 2
    a = [1 << i for i in range(12)]
    best_a = a[:]
    best_max = max(a)
    
    # Precompute subset sums for initial
    # We will use a helper to check validity and get max
    
    def get_sums(arr):
        sums = [0]
        for x in arr:
            sums += [s + x for s in sums]
        return sums
        
    def is_valid(arr):
        sums = get_sums(arr)
        return len(sums) == len(set(sums))
        
    # Local Search
    current_a = a[:]
    current_max = max(current_a)
    
    # We want to minimize the max element.
    # We can try to reduce the largest element, then the next largest, etc.
    # Or just random perturbations.
    
    # Strategy: 
    # 1. Try to reduce the largest element by 1. If valid, keep.
    # 2. If not, try to modify other elements to make room.
    
    # Better: Simulated Annealing on the vector a.
    # Objective: max(a) + 1000 * (number of collisions)
    
    # To speed up, we only recalculate sums for the changed element.
    # But maintaining the set of sums dynamically is complex.
    # Given n=12, full recalculation is 4096 ops.
    # 1000 iterations/s is safe. 85s -> 85000 iterations.
    
    # Let's try a targeted search:
    # Start with powers of 2.
    # Try to reduce a[11].
    # If fail, try to reduce a[10] and a[11].
    
    # Actually, a random search with restarts might find a better configuration.
    
    # We will use a simple hill climbing with random restarts.
    
    def score(arr):
        if len(set(arr)) != len(arr):
            return 10**9
        if min(arr) <= 0:
            return 10**9
        sums = get_sums(arr)
        collisions = len(sums) - len(set(sums))
        return max(arr) + 1000 * collisions
        
    best_score = score(current_a)
    
    # We'll run multiple restarts or a long SA.
    # Given the structure, let's try to reduce the max element directly.
    
    # Heuristic: 
    # The optimal set is likely "dense" at the bottom and "sparse" at the top?
    # Or "balanced"?
    
    # Let's try to find a set with max < 2048.
    # We know for n=4, max=7.
    # For n=12, maybe max is around 1000?
    
    # We will perform a random search biased towards reducing the max.
    
    current_a = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    best_a = current_a[:]
    best_max = max(current_a)
    
    # Precompute prefix sums to speed up? No, just full recalc.
    
    while time.time() < deadline:
        # Generate a new candidate
        # 1. Pick a random index i
        # 2. Change a[i] to a[i] + delta
        # 3. Ensure sorted and distinct
        
        # To be effective, we should try to reduce the largest elements.
        # Let's pick an index i with probability proportional to its value?
        # Or just pick the largest one and try to reduce it.
        
        # Try to reduce the current maximum
        max_idx = current_a.index(max(current_a))
        
        # Try reducing max_idx
        new_a = current_a[:]
        # Try reducing by 1, or jumping to a random smaller value
        if random.random() < 0.5:
            new_a[max_idx] -= 1
        else:
            # Random value between current max/2 and current max
            new_val = random.randint(current_a[max_idx]//2, current_a[max_idx]-1)
            new_a[max_idx] = new_val
            
        # Re-sort
        new_a.sort()
        
        # Check distinctness
        if len(set(new_a)) < 12:
            continue
            
        # Check validity
        sums = get_sums(new_a)
        if len(sums) == 4096:
            new_max = max(new_a)
            if new_max < best_max:
                best_max = new_max
                best_a = new_a
                current_a = new_a
            else:
                # Even if not better, accept with some probability?
                # For now, strict hill climbing might get stuck.
                # Let's accept if it's not worse, to explore.
                if new_max <= best_max:
                    current_a = new_a
        else:
            # Collision. Maybe try to fix it by adjusting another element?
            # For simplicity, reject.
            pass
            
    return best_a

# EVOLVE-BLOCK-END
