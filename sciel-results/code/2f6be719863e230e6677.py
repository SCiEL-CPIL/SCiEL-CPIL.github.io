import random
import numpy as np

def construct():
    n = 128
    target_min, target_max = 128, 256
    
    # Helper to compute representation counts and max rep
    def get_reps(A_set):
        reps = np.zeros(target_max - target_min + 1, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    reps[s - target_min] += 1
        return reps

    def is_valid(A_set):
        reps = get_reps(A_set)
        return np.all(reps > 0)

    def max_rep(A_set):
        reps = get_reps(A_set)
        return np.max(reps)

    # 1. Initialize with the full set [0, 128]
    # This guarantees validity and provides a baseline for removal.
    A = set(range(n + 1))
    current_reps = get_reps(A)
    current_max = np.max(current_reps)
    
    # 2. Greedy Removal Phase
    # We iteratively remove elements that do not compromise validity 
    # and ideally reduce the maximum representation count.
    improved = True
    while improved:
        improved = False
        elements_to_check = list(A)
        # Shuffle to break symmetry and avoid bias in removal order
        random.seed(42) # Fixed seed for reproducibility in this phase? 
        # Actually, let's just iterate in sorted order or random. 
        # Random is better for escaping local structures.
        random.shuffle(elements_to_check)
        
        for x in elements_to_check:
            if x not in A:
                continue
            
            new_A = set(A)
            new_A.remove(x)
            
            # Check validity
            if not is_valid(new_A):
                continue
            
            new_reps = get_reps(new_A)
            new_max = np.max(new_reps)
            
            # Accept removal if it reduces max rep, or keeps max rep same but reduces set size
            if new_max < current_max or (new_max == current_max and len(new_A) < len(A)):
                A = new_A
                current_reps = new_reps
                current_max = new_max
                improved = True
        
        # If no removal happened in the last full pass, break
        if not improved:
            break

    # 3. Local Search / Swap Phase
    # Try to swap elements to reduce max rep further.
    # Focus on sums with high multiplicity.
    
    current_reps = get_reps(A)
    current_max = np.max(current_reps)
    
    # Run swap search
    # We use a simple hill-climbing with random restarts/swaps
    for _ in range(2000):
        # Find a sum with max rep
        max_idx = np.argmax(current_reps)
        S_max = target_min + max_idx
        
        # Get pairs for S_max
        pairs = []
        A_list = sorted(A)
        for i in range(len(A_list)):
            a = A_list[i]
            b = S_max - a
            if b >= a and b in A:
                pairs.append((a, b))
        
        if not pairs:
            continue
            
        # Pick a random element from these pairs to try to replace
        a, b = random.choice(pairs)
        
        # Try replacing a or b with a new element y
        x = random.choice([a, b])
        
        # Pick a random y from [0, 128] not in A
        candidates_y = [y for y in range(n + 1) if y not in A]
        if not candidates_y:
            continue
            
        y = random.choice(candidates_y)
        
        new_A = set(A)
        new_A.remove(x)
        new_A.add(y)
        
        if not is_valid(new_A):
            continue
            
        new_reps = get_reps(new_A)
        new_max = np.max(new_reps)
        
        # Accept if improvement
        if new_max < current_max:
            A = new_A
            current_reps = new_reps
            current_max = new_max
        elif new_max == current_max and random.random() < 0.05:
            # Accept neutral moves occasionally to explore
            A = new_A
            current_reps = new_reps
            current_max = new_max

    return sorted(A)
