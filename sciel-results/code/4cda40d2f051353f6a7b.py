import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial set
    N = 256
    S = set(range(2, N + 1))
    
    # Precompute multiples for fast lookup
    multiples = [set() for _ in range(N + 1)]
    for x in S:
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                multiples[d].add(x)
                other = x // d
                if other != d:
                    multiples[other].add(x)
    
    def check_conflicts():
        counts = {a: 0 for a in S}
        for a in S:
            for b in S:
                if b == a:
                    continue
                g = gcd(a, b)
                a_prime = a // g
                
                if a_prime == 1:
                    if len(S) > 2:
                        counts[a] += 1
                else:
                    if b in multiples[a_prime]:
                        if len(multiples[a_prime]) > 2:
                            counts[a] += 1
                    else:
                        if len(multiples[a_prime]) > 1:
                            counts[a] += 1
        
        return counts
    
    def local_search(S, time_limit):
        """Try to improve S by removing one element and adding another"""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try each x in S: remove x, try to add some y not in S
            for x in list(S):
                if time.time() > time_limit:
                    break
                S_temp = S - {x}
                # Try to add each y not in S_temp
                for y in range(2, N + 1):
                    if y in S_temp:
                        continue
                    # Check if y can be added to S_temp
                    valid = True
                    for b in S_temp:
                        g = gcd(y, b)
                        y_prime = y // g
                        # Need c in S_temp, c != y, c != b, with y_prime | c
                        # y is not in S_temp, so just need c != b
                        if y_prime == 1:
                            if len(S_temp) > 1:
                                valid = False
                                break
                        else:
                            # Check if there's c in S_temp, c != b, with y_prime | c
                            for c in S_temp:
                                if c != b and c % y_prime == 0:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        S = S_temp | {y}
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_S = set()
    best_size = 0
    
    # Run multiple randomized greedy removals
    for run in range(50):
        if time.time() > deadline - 10:
            break
            
        S = set(range(2, N + 1))
        
        multiples = [set() for _ in range(N + 1)]
        for x in S:
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    multiples[d].add(x)
                    other = x // d
                    if other != d:
                        multiples[other].add(x)
        
        while True:
            if time.time() > deadline - 10:
                break
                
            counts = check_conflicts()
            bads = [a for a, c in counts.items() if c > 0]
            
            if not bads:
                break
            
            max_count = max(counts[a] for a in bads)
            candidates = [a for a in bads if counts[a] == max_count]
            
            if len(candidates) == 1:
                a_to_remove = candidates[0]
            else:
                a_to_remove = random.choice(candidates)
            
            S.remove(a_to_remove)
            
            for d in range(1, int(a_to_remove**0.5) + 1):
                if a_to_remove % d == 0:
                    multiples[d].discard(a_to_remove)
                    other = a_to_remove // d
                    if other != d:
                        multiples[other].discard(a_to_remove)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        if best_size > 60:
            break

    # Local search polish
    if time.time() < deadline - 5:
        best_S = local_search(best_S, deadline - 2)
    
    return list(best_S)
