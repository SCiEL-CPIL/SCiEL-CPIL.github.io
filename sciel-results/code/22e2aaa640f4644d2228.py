import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    max_target = 128
    
    # Generate 2,3,5-smooth numbers up to max_target
    smooth = []
    for x in range(1, max_target + 1):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth.append(x)
            
    # Map smooth numbers to index for faster lookup if needed, but list is small
    smooth_set = set(smooth)
    
    answers = [None] * (max_target + 1)
    
    # Precompute compatibility? 
    # For a small list, checking pairs is fast.
    
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            break
            
        # Candidates are smooth numbers <= n
        candidates = [x for x in smooth if x <= n]
        
        found = None
        # Try increasing number of terms
        for k in range(1, len(candidates) + 1):
            if time.time() > deadline:
                break
                
            # If k=1, just check if n is in candidates
            if k == 1:
                if n in smooth_set:
                    found = [n]
                    break
                else:
                    continue
            
            # For k > 1, use combinations
            # We can prune: if sum of smallest k candidates > n, break?
            # Sorted candidates
            c_sorted = sorted(candidates)
            if sum(c_sorted[:k]) > n:
                break # Even the smallest k sum to more than n
            
            # Check combinations of size k
            for terms in itertools.combinations(c_sorted, k):
                if sum(terms) != n:
                    continue
                
                # Check antichain property
                is_antichain = True
                for i in range(len(terms)):
                    for j in range(i + 1, len(terms)):
                        a, b = terms[i], terms[j]
                        if a % b == 0 or b % a == 0:
                            is_antichain = False
                            break
                    if not is_antichain:
                        break
                
                if is_antichain:
                    found = list(terms)
                    break
            
            if found is not None:
                break
        
        answers[n] = found
    
    # Return 1..128
    return answers[1:]
