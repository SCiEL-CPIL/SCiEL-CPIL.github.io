import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin before 60s limit

    # Universe
    universe = list(range(1, 23))
    
    # Precompute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, universe)
    weights = {i: L // i for i in universe}
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def try_greedy(order):
        current_set = []
        sums = {0}
        for x in order:
            w = weights[x]
            # Check if adding w causes a collision
            # new sums are s + w for s in sums
            # Collision if (s + w) in sums
            # This is equivalent to w in sums - sums
            # But checking intersection is O(2^k)
            
            # Optimization: 
            # if any (s + w) is in sums, it's a collision.
            # We can break early.
            
            collision = False
            new_sums = []
            for s in sums:
                ns = s + w
                if ns in sums:
                    collision = True
                    break
                new_sums.append(ns)
                
            if not collision:
                current_set.append(x)
                sums.update(new_sums)
                
        return current_set

    # Strategy 1: Deterministic orders
    orders = [
        sorted(universe),
        sorted(universe, reverse=True),
    ]
    
    # Add random permutations
    for _ in range(20):
        p = universe[:]
        rng.shuffle(p)
        orders.append(p)
        
    for order in orders:
        if time.time() - start_time > timeout:
            break
            
        current_set = try_greedy(order)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # Expansion phase: try to add any remaining element
            remaining = [x for x in universe if x not in current_set]
            # Rebuild sums for best_set to allow incremental adding
            sums = {0}
            for x in best_set:
                w = weights[x]
                sums.update([s + w for s in list(sums)])
            
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    w = weights[x]
                    collision = False
                    new_sums = []
                    for s in sums:
                        ns = s + w
                        if ns in sums:
                            collision = True
                            break
                        new_sums.append(ns)
                    
                    if not collision:
                        best_set.append(x)
                        remaining.remove(x)
                        sums.update(new_sums)
                        added = True
                        best_size = len(best_set)
                        if time.time() - start_time > timeout:
                            return best_set

    # Strategy 2: More random restarts
    for _ in range(500):
        if time.time() - start_time > timeout:
            break
            
        p = universe[:]
        rng.shuffle(p)
        current_set = try_greedy(p)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # Expansion phase
            remaining = [x for x in universe if x not in current_set]
            sums = {0}
            for x in best_set:
                w = weights[x]
                sums.update([s + w for s in list(sums)])
            
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    w = weights[x]
                    collision = False
                    new_sums = []
                    for s in sums:
                        ns = s + w
                        if ns in sums:
                            collision = True
                            break
                        new_sums.append(ns)
                    
                    if not collision:
                        best_set.append(x)
                        remaining.remove(x)
                        sums.update(new_sums)
                        added = True
                        best_size = len(best_set)
                        if time.time() - start_time > timeout:
                            return best_set

    return best_set
