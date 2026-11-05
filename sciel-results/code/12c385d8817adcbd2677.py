import random
from collections import Counter

def construct():
    def count_collisions(s):
        """Count the number of duplicate unordered pair sums (including diagonals) in set s."""
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        c = Counter(sums)
        # Number of "extra" collisions = total pairs - number of distinct sums
        return sum(v - 1 for v in c.values() if v > 1)
    
    def optimize(s, max_iters=500):
        """Greedily swap elements to maximize collisions with local search."""
        s = list(s)
        s_set = set(s)
        current_col = count_collisions(s)
        
        for iteration in range(max_iters):
            improved = False
            # Randomize order to escape local optima
            indices = list(range(12))
            random.shuffle(indices)
            
            for i in indices:
                old_val = s[i]
                best_swap = None
                best_score = current_col
                
                # Try replacing s[i] with each possible value in [0, 128]
                for new_val in range(129):
                    if new_val == old_val:
                        continue
                    if new_val in s_set:
                        continue
                    
                    # Temporarily replace
                    s[i] = new_val
                    s_set.discard(old_val)
                    s_set.add(new_val)
                    
                    score = count_collisions(s)
                    
                    if score > best_score:
                        best_score = score
                        best_swap = new_val
                    
                    # Restore
                    s[i] = old_val
                    s_set.discard(new_val)
                    s_set.add(old_val)
                
                if best_swap is not None:
                    s[i] = best_swap
                    s_set.discard(old_val)
                    s_set.add(best_swap)
                    current_col = best_score
                    improved = True
            
            if not improved:
                break
        
        return s, current_col
    
    def try_cluster_around_centers(s):
        """Try to cluster elements around a few center points to maximize collisions."""
        s = list(s)
        s_set = set(s)
        current_col = count_collisions(s)
        
        # Try different cluster configurations
        for center in range(0, 129, 4):
            # Try placing elements in a tight cluster around center
            candidate = []
            for offset in range(-6, 7):
                val = center + offset
                if 0 <= val <= 128 and val not in candidate:
                    candidate.append(val)
            
            if len(candidate) < 12:
                continue
            
            # Fill remaining slots with nearby values
            for val in range(129):
                if len(candidate) >= 12:
                    break
                if val not in candidate and val not in s_set:
                    candidate.append(val)
            
            if len(candidate) != 12:
                continue
            
            optimized, col = optimize(candidate, max_iters=200)
            if col > current_col:
                s = optimized
                current_col = col
        
        return s, current_col
    
    # Use fixed seed for reproducibility
    random.seed(42)
    
    best_set = None
    best_collision = -1
    
    # Multiple random restarts with greedy optimization
    for trial in range(100):
        # Random initial set
        s = random.sample(range(129), 12)
        optimized, col = optimize(s, max_iters=300)
        if col > best_collision:
            best_collision = col
            best_set = optimized
    
    # Try cluster-based approaches
    for trial in range(30):
        # Start with a random cluster center
        center = random.randint(10, 118)
        offsets = random.sample(range(-15, 16), 12)
        s = []
        for off in offsets:
            val = center + off
            if 0 <= val <= 128 and val not in s:
                s.append(val)
        
        # Fill if needed
        for val in range(129):
            if len(s) >= 12:
                break
            if val not in s:
                s.append(val)
        
        if len(s) != 12:
            continue
            
        optimized, col = optimize(s, max_iters=300)
        if col > best_collision:
            best_collision = col
            best_set = optimized
    
    # Try arithmetic progression based starts
    for start_val in range(0, 20):
        for step in [1, 2, 3, 4, 5]:
            s = []
            for i in range(12):
                val = start_val + i * step
                if 0 <= val <= 128 and val not in s:
                    s.append(val)
            if len(s) == 12:
                optimized, col = optimize(s, max_iters=300)
                if col > best_collision:
                    best_collision = col
                    best_set = optimized
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return list(best_set)
