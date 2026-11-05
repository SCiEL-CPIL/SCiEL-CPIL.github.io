import time
import random
from itertools import combinations

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s
    
    best_set = list(range(1, 21))
    best_obj = objective(best_set)
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        A = list(A)
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]
            
    # 1. Evaluate shifted Arithmetic Progressions (baseline)
    for k in range(1, 20):
        if time.time() > deadline:
            break
        A = list(range(k, k + 20))
        try_candidate(A)
        
    # 2. Evaluate APs with step > 1
    for d in [1, 2, 3, 4, 5, 10, 100]:
        for a in range(1, 10):
            if time.time() > deadline:
                break
            A = [a + i * d for i in range(20)]
            try_candidate(A)
            
    # 3. Structural Change: Local Search with Diverse Initializations
    # The hypothesis is that "structured" sets (like APs) have large sumsets.
    # Sets with many internal relations (like geometric progressions or sets with small differences)
    # might have smaller product sets or sumsets.
    # We will use a local search (Hill Climbing) starting from various seeds.
    
    def local_search(start_set, max_iters=500):
        current_set = list(start_set)
        current_obj = objective(current_set)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                return current_set, current_obj
            
            # Generate neighbors
            neighbors = []
            
            # Move 1: Replace one element with a random valid integer
            for i in range(20):
                # Try a few random replacements
                for _ in range(5):
                    new_val = random.randint(1, 10000)
                    if new_val in current_set and new_val != current_set[i]:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_val
                    if len(set(new_set)) == 20:
                        neighbors.append(new_set)
                        
            # Move 2: Swap two elements
            for i in range(20):
                for j in range(i+1, 20):
                    for _ in range(2):
                        new_val_i = random.randint(1, 10000)
                        new_val_j = random.randint(1, 10000)
                        new_set = current_set[:]
                        new_set[i] = new_val_i
                        new_set[j] = new_val_j
                        if len(set(new_set)) == 20:
                            neighbors.append(new_set)
                            
            # Move 3: Perturb one element by a small amount
            for i in range(20):
                for delta in [-1, 1, -2, 2, -5, 5]:
                    new_val = current_set[i] + delta
                    if 1 <= new_val <= 10000:
                        new_set = current_set[:]
                        new_set[i] = new_val
                        if len(set(new_set)) == 20:
                            neighbors.append(new_set)
                            
            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_obj = current_obj
            
            # Limit evaluation to avoid timeout
            random.shuffle(neighbors)
            for neighbor in neighbors[:100]:
                if time.time() > deadline:
                    break
                obj = objective(neighbor)
                if obj < best_neighbor_obj:
                    best_neighbor_obj = obj
                    best_neighbor = neighbor
                    
            if best_neighbor is not None:
                current_set = best_neighbor
                current_obj = best_neighbor_obj
                improved = True
                
        return current_set, current_obj
        
    # Initialize random seeds
    random.seed(42)
    
    # Seed 1: Consecutive integers
    seed1 = list(range(1, 21))
    # Seed 2: Geometric progression (truncated to fit 10000)
    # 1, 2, 4, ..., 2^19 is too big. 1, 2, 4... up to 2^13=8192.
    # Let's try a GP with base 2, but we need 20 elements.
    # 2^0..2^19 doesn't fit.
    # Let's try base 1.1? No, integers.
    # How about a set that is "dense" in log scale?
    # Or just random sets.
    
    seeds = [
        list(range(1, 21)),
        list(range(10, 30)),
        [random.randint(1, 10000) for _ in range(20)],
        [random.randint(1, 100) for _ in range(20)],
        [random.randint(1, 1000) for _ in range(20)],
        # A "mixed" seed: small AP + large random
        list(range(1, 10)) + [random.randint(100, 10000) for _ in range(10)],
    ]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        # Ensure seed is valid
        if len(set(seed)) != 20:
            # Fix duplicates
            seed = list(set(seed))
            while len(seed) < 20:
                seed.append(random.randint(1, 10000))
            seed = seed[:20]
        if any(x < 1 or x > 10000 for x in seed):
            seed = [random.randint(1, 10000) for _ in range(20)]
            
        try_candidate(seed)
        
        # Run local search
        ls_set, ls_obj = local_search(seed, max_iters=100)
        if ls_obj < best_obj:
            best_obj = ls_obj
            best_set = ls_set[:]
            
    # Final check: try a few more random restarts if time permits
    while time.time() < deadline - 2.0:
        seed = [random.randint(1, 10000) for _ in range(20)]
        if len(set(seed)) == 20:
            try_candidate(seed)
            ls_set, ls_obj = local_search(seed, max_iters=20)
            if ls_obj < best_obj:
                best_obj = ls_obj
                best_set = ls_set[:]
                
    return best_set
