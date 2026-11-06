import numpy as np
import random
import time

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses Simulated Annealing with a fast O(N^2) scoring function.
    """
    
    def fast_score(s):
        """
        s: set of integers.
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        valid_ds = set()
        # Convert to list for faster iteration if needed, but set lookup is O(1)
        # We iterate over all pairs (u, v) with u < v
        items = list(s)
        n = len(items)
        for i in range(n):
            u = items[i]
            for j in range(i + 1, n):
                v = items[j]
                # Case 1: u, v are 1st and 2nd terms. d = v - u. Check 2v - u.
                d1 = v - u
                if 2 * v - u in s:
                    valid_ds.add(d1)
                
                # Case 2: u, v are 1st and 3rd terms. d = (v - u) / 2. Check (u + v) / 2.
                diff = v - u
                if diff % 2 == 0:
                    d2 = diff // 2
                    if (u + v) // 2 in s:
                        valid_ds.add(d2)
                    
                # Case 3: u, v are 2nd and 3rd terms. d = v - u. Check 2u - v.
                # d is same as d1, but check condition is different.
                if 2 * u - v in s:
                    valid_ds.add(d1)
        return len(valid_ds)

    random.seed(42)
    
    # Initial state: Random set
    current_set = set(random.sample(range(1, 513), 32))
    current_score = fast_score(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated Annealing Parameters
    start_temp = 10.0
    end_temp = 0.001
    iterations = 200000 # Increase iterations due to faster scoring
    start_time = time.time()
    time_limit = 110.0
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (i / iterations)
        
        # Generate a random move
        if len(current_set) == 0:
            break
        
        # Pick a random element to remove
        to_remove = random.choice(list(current_set))
        
        # Pick a random element to add
        to_add = random.randint(1, 512)
        while to_add in current_set:
            to_add = random.randint(1, 512)
            
        new_set = set(current_set)
        new_set.remove(to_remove)
        new_set.add(to_add)
        
        new_score = fast_score(new_set)
        
        delta = new_score - current_score
        if delta > 0 or (random.random() < np.exp(delta / temp)):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    return sorted(best_set)
