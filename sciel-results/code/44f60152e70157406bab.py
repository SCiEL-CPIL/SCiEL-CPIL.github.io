"""
Strategy: 
Instead of using Differential Evolution on a continuous relaxation, we treat this as a discrete 
combinatorial optimization problem on the set A. 
We start with a valid baseline (A = range(64, 129)). 
We then use a local search / simulated annealing hybrid that specifically targets the 
"congestion" (peak multiplicity). 
Moves are: 
1. Remove an element and add another (swap) 
2. Add an element 
3. Remove an element 
We accept moves that reduce the peak multiplicity or maintain it while reducing the "variance" 
of multiplicities (to prepare for further reductions). 
Crucially, we use delta updates to compute the new counts in O(1) or O(k) time where k is the 
degree of the node, rather than O(N^2).
"""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute neighbors for fast delta updates
    # For each element x in [0, 128], and for each target sum s in [128, 256],
    # does x contribute to s? It contributes if there exists y in A such that x+y=s.
    # But A changes. So we can't precompute neighbors of x in the graph of A easily.
    # However, we can precompute the potential partners for each x and each sum.
    # If we add/remove x, the counts change for sums s where the partner y = s-x is in A.
    
    # Let's represent A as a boolean array of size 129.
    # Let counts be an array of size 129 (for sums 128..256).
    
    # Initial valid set: A = {64, ..., 128}
    A = np.zeros(129, dtype=int)
    for i in range(64, 129):
        A[i] = 1
        
    # Compute initial counts
    # Counts[s-128] = number of pairs (i,j) in A with i<=j and i+j = s
    counts = np.zeros(num_targets, dtype=int)
    
    def update_counts_add(x):
        """Update counts when x is added to A."""
        for s in range(min_sum, max_sum + 1):
            y = s - x
            if 0 <= y <= 128 and A[y] == 1:
                # Pair (x, y) is now valid.
                # We need to count unordered pairs.
                # If x == y, this is a new pair (x,x).
                # If x != y, this is a new pair (x,y). 
                # Note: The pair (y,x) is the same as (x,y).
                # Since we iterate over all s, and we only add x once, 
                # we are adding the contribution of x to all sums where its partner is in A.
                # Does this double count?
                # Let's check.
                # If A={1,2}. Add 3.
                # s=4, y=1. A[1]=1. Count[4] += 1. Pair (3,1).
                # s=5, y=2. A[2]=1. Count[5] += 1. Pair (3,2).
                # s=6, y=3. A[3]=0 (before add). 
                # Wait, if we are adding x, we check A[y]. 
                # If y==x, A[x] is 0 before we set it to 1? 
                # We should update A[x]=1 BEFORE or handle x==y carefully.
                # Let's update A[x]=1 first, then loop.
                # If y==x, A[x] is 1. Count[2x] += 1.
                # If y!=x, A[y] is 1. Count[x+y] += 1.
                # Is it double counted?
                # When we added y previously, did we count (y,x)?
                # When y was added, x was not in A. So no.
                # So this logic is correct: adding x adds exactly one count for each partner y in A (including x itself if 2x is in range).
                idx = s - min_sum
                counts[idx] += 1
        
    def update_counts_remove(x):
        """Update counts when x is removed from A."""
        for s in range(min_sum, max_sum + 1):
            y = s - x
            if 0 <= y <= 128 and A[y] == 1:
                # This pair (x,y) existed.
                # If x==y, we remove 1 count.
                # If x!=y, we remove 1 count.
                idx = s - min_sum
                counts[idx] -= 1
        A[x] = 0 # Set to 0 after processing? 
        # Wait, if x==y, A[x] is 1. We subtract. Then we set A[x]=0. Correct.
        # But if we set A[x]=0 at the start, then for y==x, A[y] becomes 0 and we skip. 
        # So we must check A[y] BEFORE setting A[x]=0, or handle x==y specifically.
        # Let's set A[x]=0 at the END.
        # But what if y != x? A[y] is independent.
        # So:
        # 1. Loop s. If A[s-x] is 1, decrement.
        # 2. Set A[x] = 0.
        # This works.

    # Initialize counts properly
    # The above add/remove assumes A is empty initially.
    # Let's just compute from scratch for initialization to be safe.
    for i in range(129):
        if A[i]:
            for j in range(i, 129):
                if A[j]:
                    s = i + j
                    if min_sum <= s <= max_sum:
                        counts[s - min_sum] += 1
    
    current_peak = np.max(counts)
    best_peak = current_peak
    best_A = A.copy()
    
    # Local Search / Simulated Annealing
    # We want to minimize the peak.
    # If peak is fixed, we want to minimize the "sum of squares" or similar to flatten distribution.
    
    def get_objective(counts):
        # Primary: max count
        # Secondary: sum of counts^2 (to encourage uniformity)
        peak = np.max(counts)
        # To allow moves that keep peak same but improve potential for future reduction,
        # we can use a small penalty for variance.
        # But strictly, we only care about peak.
        # However, if we only look at peak, we might get stuck.
        # Let's try: objective = peak * 1000 + np.sum(counts**2)
        # This prioritizes peak reduction heavily.
        return peak * 1000 + np.sum(counts**2)
    
    current_obj = get_objective(counts)
    best_obj = current_obj
    
    T = 100.0 # Initial temperature
    T_min = 1.0
    alpha = 0.999
    
    # Precompute list of all elements for random selection
    all_elements = list(range(129))
    
    # Random shuffle for reproducibility or just use random module
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Pick a random move
        move_type = rng.randint(0, 2) # 0: swap, 1: add, 2: remove
        
        if move_type == 0: # Swap
            # Pick x in A to remove, y not in A to add
            # To speed up, we can just pick random x, y.
            x = rng.choice(all_elements)
            y = rng.choice(all_elements)
            if A[x] == 0 or A[y] == 1 or x == y:
                continue
            
            # Try swap: remove x, add y
            # Save state
            # We need to calculate delta objective efficiently.
            
            # Remove x
            old_counts = counts.copy() # This is expensive. 129 ints is small, but let's avoid if possible.
            # Actually, 129 ints is ~1KB. Copying is fast.
            # But we can do better.
            
            # Let's just do the update and rollback if rejected.
            
            # Remove x
            for s in range(min_sum, max_sum + 1):
                yy = s - x
                if 0 <= yy <= 128 and A[yy] == 1:
                    counts[s - min_sum] -= 1
            A[x] = 0
            
            # Add y
            for s in range(min_sum, max_sum + 1):
                xx = s - y
                if 0 <= xx <= 128 and A[xx] == 1:
                    counts[s - min_sum] += 1
            A[y] = 1
            
            new_obj = get_objective(counts)
            
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A.copy()
                    best_peak = np.max(counts)
            else:
                # Probabilistic acceptance
                delta = new_obj - current_obj
                if delta > 0:
                    # Accept with probability exp(-delta/T)
                    if rng.random() < np.exp(-delta / T):
                        current_obj = new_obj
                    else:
                        # Rollback
                        # Remove y
                        for s in range(min_sum, max_sum + 1):
                            xx = s - y
                            if 0 <= xx <= 128 and A[xx] == 1:
                                counts[s - min_sum] -= 1
                        A[y] = 0
                        
                        # Add x
                        for s in range(min_sum, max_sum + 1):
                            yy = s - x
                            if 0 <= yy <= 128 and A[yy] == 1:
                                counts[s - min_sum] += 1
                        A[x] = 1
        elif move_type == 1: # Add
            y = rng.choice(all_elements)
            if A[y] == 1:
                continue
            
            # Add y
            for s in range(min_sum, max_sum + 1):
                xx = s - y
                if 0 <= xx <= 128 and A[xx] == 1:
                    counts[s - min_sum] += 1
            A[y] = 1
            
            new_obj = get_objective(counts)
            
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A.copy()
                    best_peak = np.max(counts)
            else:
                delta = new_obj - current_obj
                if delta > 0:
                    if rng.random() < np.exp(-delta / T):
                        current_obj = new_obj
                    else:
                        # Rollback
                        for s in range(min_sum, max_sum + 1):
                            xx = s - y
                            if 0 <= xx <= 128 and A[xx] == 1:
                                counts[s - min_sum] -= 1
                        A[y] = 0
        else: # Remove
            # Pick x in A
            # To avoid empty set, check size
            if np.sum(A) <= 1:
                continue
            x = rng.choice(all_elements)
            if A[x] == 0:
                continue
            
            # Check if removing x breaks coverage?
            # We can check if any count becomes 0.
            # But objective function penalizes? No, it doesn't penalize coverage explicitly in get_objective.
            # We MUST ensure coverage.
            # Let's check coverage before accepting a remove move.
            
            # Simulate remove
            for s in range(min_sum, max_sum + 1):
                yy = s - x
                if 0 <= yy <= 128 and A[yy] == 1:
                    counts[s - min_sum] -= 1
            
            A[x] = 0
            
            # Check coverage
            if np.min(counts) == 0:
                # Rollback immediately
                for s in range(min_sum, max_sum + 1):
                    yy = s - x
                    if 0 <= yy <= 128 and A[yy] == 1:
                        counts[s - min_sum] += 1
                A[x] = 1
                continue
            
            new_obj = get_objective(counts)
            
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A.copy()
                    best_peak = np.max(counts)
            else:
                delta = new_obj - current_obj
                if delta > 0:
                    if rng.random() < np.exp(-delta / T):
                        current_obj = new_obj
                    else:
                        # Rollback
                        for s in range(min_sum, max_sum + 1):
                            yy = s - x
                            if 0 <= yy <= 128 and A[yy] == 1:
                                counts[s - min_sum] += 1
                        A[x] = 1
        
        # Cool down
        T = max(T_min, T * alpha)
        
        # Optional: if we find a very good solution, we can restart or tighten cooling
        # But for now, simple SA.
        
    # Final verification and return
    # best_A should be valid.
    # Convert to list of indices
    result = [i for i, v in enumerate(best_A) if v == 1]
    
    # Double check validity
    if not result:
         return list(range(64, 129))
         
    return result
# EVOLVE-BLOCK-END
