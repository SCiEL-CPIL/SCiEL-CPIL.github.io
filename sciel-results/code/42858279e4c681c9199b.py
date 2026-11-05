import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Primes to consider. 
    # We don't need primes > 1000 because 2*3*...*p_k grows fast.
    # 2*3*5*...*47 (15 primes) is ~ 6e17.
    # 2*3*...*53 (16 primes) is ~ 3e19 > 1e18.
    # So max distinct primes is 15.
    # We can use primes up to 100 to be safe for exponents.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    num_primes = len(primes)
    
    def get_divisors_from_exp(exp_list):
        divs = [1]
        for i, e in enumerate(exp_list):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return sorted(divs)

    def calc_score(n, exp_list):
        # Reconstruct divisors
        divs = get_divisors_from_exp(exp_list)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initialize best state
    # Strategy: Start with a primorial
    best_n = 2
    best_score = 0
    best_exp = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Try a few initial primorials
    for k in range(10, 17):
        exp_test = [0] * num_primes
        prod = 1
        valid = True
        for i in range(k):
            exp_test[i] = 1
            prod *= primes[i]
        if prod <= 10**18:
            score = calc_score(prod, exp_test)
            if score > best_score:
                best_score = score
                best_n = prod
                best_exp = exp_test[:]
        else:
            break
            
    # Simulated Annealing
    current_exp = best_exp[:]
    current_n = best_n
    current_score = best_score
    
    # Temperature
    T = 10.0
    cooling = 0.999
    
    # Helper to calculate new n from exp change
    def get_n_from_exp(exp_list):
        n = 1
        for i, e in enumerate(exp_list):
            if e > 0:
                n *= (primes[i] ** e)
        return n

    # Initial n from best_exp
    current_n = get_n_from_exp(current_exp)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random prime index to modify
        # Bias towards smaller primes? Or active primes?
        # Let's pick a random index from 0 to num_primes-1
        idx = random.randint(0, num_primes - 1)
        
        # Decide delta: +1 or -1
        # If e is 0, only +1 makes sense (unless we want to remove a prime that was added? No, if e=0 it's not there)
        # If e > 0, we can -1 or +1.
        
        if current_exp[idx] == 0:
            delta = 1
        else:
            # 80% chance to decrease, 20% to increase? 
            # Or just random.
            delta = random.choice([-1, 1])
            
        new_exp = current_exp[:]
        new_exp[idx] += delta
        
        if new_exp[idx] < 0:
            continue
            
        # Check if n exceeds limit
        # We can estimate the change
        if delta == 1:
            new_n = current_n * primes[idx]
        else:
            new_n = current_n // primes[idx]
            
        if new_n > 10**18:
            continue
            
        if new_n < 2:
            continue
            
        # Evaluate new state
        new_score = calc_score(new_n, new_exp)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept better
            current_exp = new_exp
            current_n = new_n
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_exp = current_exp[:]
        else:
            # Accept worse with probability exp(-delta_score / T)
            delta_score = current_score - new_score
            if delta_score == 0:
                # Equal score, accept to move around?
                if random.random() < 0.5:
                    current_exp = new_exp
                    current_n = new_n
                    current_score = new_score
            else:
                prob = math.exp(-delta_score / T)
                if random.random() < prob:
                    current_exp = new_exp
                    current_n = new_n
                    current_score = new_score
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor to keep some randomness
            
        # Restart logic? If stuck, maybe reheat?
        # For now, simple cooling.
        
        # Periodic check for time
        if steps % 1000 == 0:
             pass # print debug if needed

    return best_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")
