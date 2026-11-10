import time
import math

def construct():
    start_time = time.time()
    deadline = 90.0
    margin = 5.0
    
    N = 256
    
    # Sieve of Eratosthenes to find primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate max power for each prime
    max_power = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        max_power[p] = power
        
    # Initial assignment: each prime is its own group
    # Group is a frozenset of primes
    # Value of group: if len==1, max_power[p]. If len>1, product of primes (if <= N)
    groups = [frozenset([p]) for p in primes]
    
    def get_value(group):
        prod = 1
        for p in group:
            prod *= p
        if prod > N:
            return -1 # Invalid
        if len(group) == 1:
            return max_power[list(group)[0]]
        return prod
        
    def get_sum(groups):
        total = 1 # Include 1
        for g in groups:
            v = get_value(g)
            if v < 0:
                return -1
            total += v
        return total
        
    # Greedy Merge Phase
    # Sort groups by potential gain?
    # Just iterate and merge if beneficial
    improved = True
    while improved:
        improved = False
        for i in range(len(groups)):
            if time.time() > deadline - margin:
                break
            for j in range(i + 1, len(groups)):
                if time.time() > deadline - margin:
                    break
                g1 = groups[i]
                g2 = groups[j]
                # Try merging
                new_group = frozenset(g1.union(g2))
                v_new = get_value(new_group)
                if v_new < 0:
                    continue
                v_old1 = get_value(g1)
                v_old2 = get_value(g2)
                if v_new > v_old1 + v_old2:
                    # Perform merge
                    groups[i] = new_group
                    groups.pop(j)
                    improved = True
                    break # Restart loop for i
            
            if time.time() > deadline - margin:
                break
                
    # Local Search Phase
    # Try moving a prime from one group to another
    # Or splitting a group
    for _ in range(1000): # Limited iterations
        if time.time() > deadline - margin:
            break
        best_gain = 0
        best_move = None
        
        # Try moving a prime p from group A to group B
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for p in groups[i]:
                    # Move p from i to j
                    new_g_i = frozenset(groups[i] - {p})
                    new_g_j = frozenset(groups[j] | {p})
                    
                    if not new_g_i: # Group i becomes empty
                        # Remove group i, add group j
                        v_new_i = 0
                        v_new_j = get_value(new_g_j)
                        if v_new_j < 0: continue
                        v_old_i = get_value(groups[i])
                        v_old_j = get_value(groups[j])
                        gain = (v_new_i + v_new_j) - (v_old_i + v_old_j)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('move_remove', i, j, p)
                    else:
                        v_new_i = get_value(new_g_i)
                        v_new_j = get_value(new_g_j)
                        if v_new_i < 0 or v_new_j < 0: continue
                        v_old_i = get_value(groups[i])
                        v_old_j = get_value(groups[j])
                        gain = (v_new_i + v_new_j) - (v_old_i + v_old_j)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('move', i, j, p)
                            
        if best_gain > 0 and best_move:
            action = best_move[0]
            if action == 'move_remove':
                _, i, j, p = best_move
                groups.pop(i)
                groups[j] = frozenset(groups[j] | {p})
            else:
                _, i, j, p = best_move
                groups[i] = frozenset(groups[i] - {p})
                groups[j] = frozenset(groups[j] | {p})
        else:
            break
            
    # Final Check and Return
    if time.time() > deadline - margin:
        # Fallback to simple prime powers if search is too slow (unlikely)
        return [1] + [max_power[p] for p in primes]
        
    result = [1]
    for g in groups:
        v = get_value(g)
        if v < 0:
            return [1] + [max_power[p] for p in primes]
        result.append(v)
        
    # Ensure distinctness (should be guaranteed by logic, but check)
    if len(set(result)) != len(result):
        return [1] + [max_power[p] for p in primes]
        
    # Ensure all are int
    return [int(x) for x in result]
