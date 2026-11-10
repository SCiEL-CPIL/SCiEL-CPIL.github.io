# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    factors = {i: prime_factors(i) for i in numbers}
    primes = [i for i in range(2, N + 1) if len(factors[i]) == 1]
    
    # Start with primes + 1
    current_set = set([1] + primes)
    
    def conflicts_of(num, s):
        return [x for x in s if x != num and gcd(num, x) != 1]
    
    def is_coprime_to_all(num, s):
        return all(x == num or gcd(num, x) == 1 for x in s)
    
    current_sum = sum(current_set)
    best_set = set(current_set)
    best_sum = current_sum
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        best_gain = 0
        best_action = None  # (add_list, remove_list)
        
        # Try single add (with necessary removals)
        for num in numbers:
            if num in current_set:
                continue
            conf = conflicts_of(num, current_set)
            gain = num - sum(conf)
            if gain > best_gain:
                best_gain = gain
                best_action = ([num], conf)
        
        # Try single remove + single add
        for rem in current_set:
            temp = current_set - {rem}
            for num in numbers:
                if num in temp:
                    continue
                if is_coprime_to_all(num, temp):
                    gain = num - rem
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ([num], [rem])
        
        # Try single remove + two adds
        for rem in current_set:
            if time.time() > deadline:
                break
            temp = current_set - {rem}
            for num1 in numbers:
                if num1 in temp:
                    continue
                if not is_coprime_to_all(num1, temp):
                    continue
                temp2 = temp | {num1}
                for num2 in numbers:
                    if num2 in temp2:
                        continue
                    if not is_coprime_to_all(num2, temp2):
                        continue
                    gain = num1 + num2 - rem
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ([num1, num2], [rem])
        
        if best_action is not None and best_gain > 0:
            add_list, remove_list = best_action
            current_set.difference_update(remove_list)
            current_set.update(add_list)
            current_sum += best_gain
            improved = True
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
        
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
