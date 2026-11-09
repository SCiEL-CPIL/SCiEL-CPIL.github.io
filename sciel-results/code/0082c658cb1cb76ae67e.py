import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 192
    all_vals = list(range(1, N + 1))
    
    def is_valid(S):
        """
        Check if set S is valid.
        Optimized: For each a in S, check if any 3 distinct others sum to 0 mod a.
        Uses O(n^3) total complexity with low constant factor.
        """
        n = len(S)
        if n < 4:
            return True
            
        for i in range(n):
            a = S[i]
            # Build list of others
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            if m < 3:
                continue
                
            # Precompute residues and counts for 'others' mod a
            # We need to check if there exist b, c, d in others (distinct) 
            # such that (b+c+d) % a == 0.
            
            # Map residue to count
            res_count = {}
            # List of (value, residue) to iterate pairs
            val_res = []
            for x in others:
                r = x % a
                val_res.append((x, r))
                res_count[r] = res_count.get(r, 0) + 1
            
            # Iterate over pairs of indices in val_res
            for bi in range(m):
                b_val, rb = val_res[bi]
                for ci in range(bi + 1, m):
                    c_val, rc = val_res[ci]
                    target_r = (-rb - rc) % a
                    
                    # We need a d in others, distinct from b and c, with residue target_r.
                    # Count of elements in others with residue target_r
                    count_target = res_count.get(target_r, 0)
                    
                    # Subtract b and c if they have the target residue
                    # because d must be distinct from b and c
                    if rb == target_r:
                        count_target -= 1
                    if rc == target_r:
                        count_target -= 1
                    
                    if count_target > 0:
                        return False
                        
        return True

    # 1. Greedy Initialization: Add numbers from largest to smallest
    # Large numbers are less likely to divide sums of smaller numbers.
    current = []
    for x in range(N, 0, -1):
        new_set = current + [x]
        if is_valid(new_set):
            current.append(x)
            
    best = list(current)
    best_size = len(best)
    
    # 2. Local Search / Simulated Annealing to try to improve
    # We want to maximize size.
    
    random.seed(42)
    
    T = 1.0
    cooling = 0.999
    
    # Track current set as a list and a set for fast lookup
    curr_list = list(current)
    curr_set = set(curr_list)
    best_list = list(best)
    best_set = set(best_list)
    
    # Precompute all values not in current for fast random selection?
    # Since N=192, checking 'x not in curr_set' is O(1).
    
    while time.time() < deadline - 5:
        # Choose a move
        r = random.random()
        
        if r < 0.5:
            # Try to ADD an element
            # Pick a random candidate not in curr_set
            # Optimization: Keep a list of available elements? 
            # For N=192, just picking random int and checking is fine.
            added = False
            attempts = 0
            while attempts < 10 and not added:
                x = random.randint(1, N)
                if x not in curr_set:
                    new_list = curr_list + [x]
                    if is_valid(new_list):
                        curr_list = new_list
                        curr_set.add(x)
                        added = True
                        if len(curr_list) > best_size:
                            best_size = len(curr_list)
                            best_list = list(curr_list)
                            best_set = set(curr_list)
                attempts += 1
                
        elif r < 0.8:
            # Try to SWAP an element
            # Remove y from curr, add x not in curr
            if len(curr_list) > 0:
                idx = random.randint(0, len(curr_list) - 1)
                y = curr_list[idx]
                
                # Find a candidate x
                attempts = 0
                swapped = False
                while attempts < 10 and not swapped:
                    x = random.randint(1, N)
                    if x not in curr_set and x != y:
                        new_list = [z for z in curr_list if z != y]
                        new_list.append(x)
                        if is_valid(new_list):
                            curr_list = new_list
                            curr_set.remove(y)
                            curr_set.add(x)
                            swapped = True
                            if len(curr_list) > best_size:
                                best_size = len(curr_list)
                                best_list = list(curr_list)
                                best_set = set(curr_list)
                    attempts += 1
                    
        else:
            # Try to REMOVE an element (to escape local optima)
            # Only accept if it leads to a better configuration later, 
            # but for now, accept with low probability to allow exploration.
            if len(curr_list) > 1:
                idx = random.randint(0, len(curr_list) - 1)
                y = curr_list[idx]
                new_list = [z for z in curr_list if z != y]
                # This move reduces size by 1. 
                # In SA, we accept worse moves with prob exp(-delta/T).
                # Here, "energy" is -size. Delta E = 1.
                # Prob = exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    curr_list = new_list
                    curr_set.remove(y)
                    
        T *= cooling
        
    # Final cleanup: try to add any remaining elements greedily
    # This is a "polish" step
    for x in range(N, 0, -1):
        if x not in best_set:
            new_set = list(best_list) + [x]
            if is_valid(new_set):
                best_list = new_set
                best_set.add(x)
                
    # One more pass with random order for polish
    remaining = [x for x in all_vals if x not in best_set]
    random.shuffle(remaining)
    for x in remaining:
        if time.time() > deadline - 1:
            break
        new_set = list(best_list) + [x]
        if is_valid(new_set):
            best_list = new_set
            best_set.add(x)
            
    return best_list

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # print(res)
