import time
import random

def construct():
    start_time = time.time()
    timeout = 80
    
    max_sum = 20000
    prime_factors = [set() for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            for j in range(i, max_sum + 1, i):
                prime_factors[j].add(i)
    
    def get_prime_count(nums):
        primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s = nums[i] + nums[j]
                primes.update(prime_factors[s])
        return len(primes)
        
    def local_search(initial_set):
        current_set = initial_set[:]
        current_score = get_prime_count(current_set)
        best_set_local = current_set[:]
        best_score_local = current_score
        
        # Simulated Annealing or Hill Climbing
        # Let's do Hill Climbing with random restarts within the time budget
        for _ in range(10000):
            if time.time() - start_time > timeout:
                break
            
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Sample new value
            # Bias towards small numbers? Or random?
            new_val = random.randint(1, 10000)
            if new_val == old_val or new_val in current_set:
                continue
                
            current_set[idx] = new_val
            new_score = get_prime_count(current_set)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score_local:
                    best_score_local = new_score
                    best_set_local = current_set[:]
            else:
                current_set[idx] = old_val
                
        return best_set_local, best_score_local

    best_global_set = []
    best_global_score = 9999
    
    initial_sets = [
        [2**i for i in range(12)],
        list(range(1, 13)),
        [random.randint(1, 10000) for _ in range(12)]
    ]
    
    # Ensure distinctness for random init
    while len(set(initial_sets[2])) < 12:
        initial_sets[2] = [random.randint(1, 10000) for _ in range(12)]
        
    for init in initial_sets:
        if time.time() - start_time > timeout:
            break
        s_set, s_score = local_search(init)
        if s_score < best_global_score:
            best_global_score = s_score
            best_global_set = s_set
            
    return best_global_set
