import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    def count_3aps(A):
        # A is a boolean numpy array of size N+1 (index 0 unused)
        count = 0
        for d in range(1, N // 2 + 1):
            # a, a+d, a+2d
            # a from 1 to N-2d
            # Vectorized check
            # A[1:N-2d+1] & A[1+d:N-d+1] & A[1+2d:N+1]
            # Indices:
            # i in 0..len-1 corresponds to a = i+1
            # a+d = i+1+d
            # a+2d = i+1+2d
            # We need A[i] & A[i+d] & A[i+2d] for i such that i+2d < N
            # Let's just loop d and use numpy slicing
            a_start = 1
            a_end = N - 2 * d + 1
            if a_end < a_start:
                continue
            # Slices
            s1 = A[a_start:a_end]
            s2 = A[a_start+d:a_end+d]
            s3 = A[a_start+2*d:a_end+2*d]
            count += np.sum(s1 & s2 & s3)
        return count

    def find_4aps(A):
        aps = []
        for d in range(1, N // 3 + 1):
            a_start = 1
            a_end = N - 3 * d + 1
            if a_end < a_start:
                continue
            s1 = A[a_start:a_end]
            s2 = A[a_start+d:a_end+d]
            s3 = A[a_start+2*d:a_end+2*d]
            s4 = A[a_start+3*d:a_end+3*d]
            # Find indices where all are True
            # np.where returns indices in the slice, need to add a_start - 1
            indices = np.where(s1 & s2 & s3 & s4)[0]
            for idx in indices:
                a = a_start + idx
                aps.append((a, a+d, a+2*d, a+3*d))
        return aps

    def repair(A):
        # Remove elements to eliminate all 4-APs
        # We want to minimize loss of 3-APs
        while True:
            aps = find_4aps(A)
            if not aps:
                break
            
            # Collect all elements involved in 4-APs
            involved = set()
            for ap in aps:
                for x in ap:
                    involved.add(x)
            
            # For each involved element, count 3-APs it's part of
            # To do this efficiently, we can just calculate the total 3-APs if we remove each candidate
            # But that's expensive. 
            # Heuristic: Remove the element that is part of the most 4-APs? 
            # Or just pick one randomly from the first 4-AP?
            # Better: Pick the element in the first 4-AP that has the lowest "degree" in 3-APs.
            
            # Let's just pick a random 4-AP and a random element from it to remove.
            # This is simple and often works well with restarts.
            ap = random.choice(aps)
            x = random.choice(ap)
            A[x] = False
            
    def improve(A):
        # Try to add elements that don't create 4-APs and increase 3-APs
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_x = -1
            
            # Current 3-AP count
            cur_3aps = count_3aps(A)
            
            # Try adding each x not in A
            # Optimization: Only try x that might form 3-APs
            for x in range(1, N + 1):
                if A[x]:
                    continue
                
                # Check if adding x creates a 4-AP
                # x can be 1st, 2nd, 3rd, 4th term
                creates_4ap = False
                
                # x as 1st: x, x+d, x+2d, x+3d
                max_d = (N - x) // 3
                for d in range(1, max_d + 1):
                    if A[x+d] and A[x+2*d] and A[x+3*d]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # x as 2nd: x-d, x, x+d, x+2d
                max_d = min(x - 1, (N - x) // 2)
                for d in range(1, max_d + 1):
                    if A[x-d] and A[x+d] and A[x+2*d]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # x as 3rd: x-2d, x-d, x, x+d
                max_d = min((x - 1) // 2, N - x)
                for d in range(1, max_d + 1):
                    if A[x-2*d] and A[x-d] and A[x+d]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # x as 4th: x-3d, x-2d, x-d, x
                max_d = (x - 1) // 3
                for d in range(1, max_d + 1):
                    if A[x-3*d] and A[x-2*d] and A[x-d]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # If no 4-AP, calculate gain in 3-APs
                # x as middle: x-d, x, x+d
                gain = 0
                max_d = min(x - 1, N - x)
                for d in range(1, max_d + 1):
                    if A[x-d] and A[x+d]:
                        gain += 1
                # x as start: x, x+d, x+2d
                max_d = (N - x) // 2
                for d in range(1, max_d + 1):
                    if A[x+d] and A[x+2*d]:
                        gain += 1
                # x as end: x-2d, x-d, x
                max_d = (x - 1) // 2
                for d in range(1, max_d + 1):
                    if A[x-2*d] and A[x-d]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_gain > 0 and best_x != -1:
                A[best_x] = True
                improved = True
                # Early exit if time is tight? No, let it finish the local search.
                
    best_A = None
    best_score = -1
    
    # Multiple restarts
    while time.time() < deadline - 5.0: # Leave 5s for final processing
        # Initialize random set
        A = np.zeros(N + 1, dtype=bool)
        # Random density, maybe 0.5
        for i in range(1, N + 1):
            if random.random() < 0.5:
                A[i] = True
        
        # Repair
        repair(A)
        
        # Improve
        improve(A)
        
        # Evaluate
        if not find_4aps(A):
            score = count_3aps(A)
            if score > best_score:
                best_score = score
                best_A = A.copy()
        
        # Perturb: Remove some random elements to escape local optimum
        # We only perturb if we have a valid set to improve from
        if best_A is not None:
             # Create a new A from best_A
             A = best_A.copy()
             # Remove 5-10 random elements
             num_remove = random.randint(5, 10)
             indices = [i for i in range(1, N + 1) if A[i]]
             if len(indices) > num_remove:
                 to_remove = random.sample(indices, num_remove)
                 for x in to_remove:
                     A[x] = False
             
             # Repair and Improve the perturbed set
             repair(A)
             improve(A)
             
             if not find_4aps(A):
                 score = count_3aps(A)
                 if score > best_score:
                     best_score = score
                     best_A = A.copy()
    
    if best_A is None:
        # Fallback: Return a simple valid set (e.g. empty or single element)
        # But task says nonempty.
        return [1]
        
    return [i for i in range(1, N + 1) if best_A[i]]
