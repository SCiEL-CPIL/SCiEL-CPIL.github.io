import time

def construct():
    N = 123457
    start_time = time.time()
    
    # Primes to test as upper bounds for smoothness
    primes_to_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will incrementally build the set of smooth numbers
    # smooth_set starts with {1}
    # For each new prime p, we update smooth_set to include numbers formed by multiplying existing smooths by powers of p
    smooth_set = {1}
    
    # Helper to check if a pair exists in smooth_set
    def find_pair(smooth_set, N):
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                if a > 0 and b > 0:
                    return [a, b]
        return None

    # Check for y=1 (only 1 is smooth, 1+1=2 != N, so no)
    # But N is large, so 1 is not enough.
    
    for p in primes_to_test:
        # Check time budget
        if time.time() - start_time > 50:
            break
            
        # Expand smooth_set with prime p
        # New numbers are of the form s * p^k where s is in current smooth_set
        new_numbers = set()
        for s in smooth_set:
            val = s
            while val <= N:
                new_numbers.add(val)
                val *= p
                if val > N:
                    break
        smooth_set.update(new_numbers)
        
        # Check for pair
        found_pair = find_pair(smooth_set, N)
        if found_pair:
            return found_pair
            
    # Fallback
    return [1, 123456]
