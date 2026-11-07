"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import product

def construct():
    primes = []
    for i in range(2, 257):
        is_prime = True
        for j in range(2, isqrt(i) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    def best_num_for_group(group):
        # Returns (value, number)
        if not group:
            return 0, 0
            
        ranges = []
        for p in group:
            powers = []
            v = 1
            while v * p <= 256:
                v *= p
                powers.append(v)
            if not powers:
                return 0, 0
            ranges.append(powers)
            
        max_val = 0
        best_num = 0
        for combo in product(*ranges):
            prod = 1
            for v in combo:
                prod *= v
            if prod <= 256 and prod > max_val:
                max_val = prod
                best_num = prod
        return max_val, best_num

    def get_partitions(elements):
        if not elements:
            yield []
        else:
            first = elements[0]
            for partition in get_partitions(elements[1:]):
                for i in range(len(partition)):
                    new_partition = [list(g) for g in partition]
                    new_partition[i].append(first)
                    yield new_partition
                yield [ [first] ] + [list(g) for g in partition]

    best_sum = 0
    best_nums = None
    
    for partition in get_partitions(small_primes):
        curr_sum = sum(large_primes)
        nums = list(large_primes)
        valid = True
        
        for group in partition:
            val, num = best_num_for_group(group)
            if val == 0:
                valid = False
                break
            curr_sum += val
            nums.append(num)
            
        if not valid:
            continue
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_nums = nums
            
    if best_nums:
        return best_nums
    else:
        # Fallback
        res = []
        for p in primes:
            v = 1
            while v * p <= 256:
                v *= p
            res.append(v)
        return res
# EVOLVE-BLOCK-END
