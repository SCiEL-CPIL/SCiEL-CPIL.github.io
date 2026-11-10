"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 78  # 78s safety margin (90s kill limit)
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses integer bitsets for fast DP."""
        n = len(others)
        if n < 2:
            return False
        target = element
        max_sum = sum(others)
        
        for k in range(2, n + 1):
            target_sum = k * target
            if target_sum > max_sum:
                break
            
            # Bitset DP: dp[j] is an integer where bit s is set if sum s is achievable with j elements
            dp = [0] * (k + 1)
            dp[0] = 1  # sum 0 achievable with 0 elements
            
            for val in others:
                for j in range(k, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
                # Early exit
                if (dp[k] >> target_sum) & 1:
                    return True
        
        return False
    
    def count_violations(selected):
        """Count the number of mean violations in the selected set."""
        n = len(selected)
        if n < 3:
            return 0
        violations = 0
        for i in range(n):
            elem = selected[i]
            others = [selected[j] for j in range(n) if j != i]
            if is_mean_of_subset(elem, others):
                violations += 1
        return violations
    
    def can_add_element(element, current_set):
        """Check if element can be added to current_set without violating the constraint."""
        # Check 1: is element the mean of any subset of current_set?
        if len(current_set) >= 2 and is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            n = len(others_without_a)
            if n == 0:
                continue
            
            max_sum = sum(others_without_a)
            for j in range(1, n + 1):
                target_sum = (j + 1) * a - element
                if target_sum <= 0 or target_sum > max_sum:
                    continue
                
                # Bitset DP
                dp = [0] * (j + 1)
                dp[0] = 1
                
                for val in others_without_a:
                    for jj in range(j, 0, -1):
                        if dp[jj - 1]:
                            dp[jj] |= (dp[jj - 1] << val)
                    if (dp[j] >> target_sum) & 1:
                        return False
        
        return True
    
    # Phase 1: Greedy construction to get a good starting set
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    orderings = []
    
    # Ordering 1: alternating small/large (spread out)
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
    orderings.append(alt)
    
    # Ordering 2: descending
    orderings.append(list(range(512, 0, -1)))
    
    # Ordering 3: ascending
    orderings.append(list(range(1, 513)))
    
    # Ordering 4: multiples of large primes first
    def get_order_by_mod(mod):
        groups = [[] for _ in range(mod)]
        for x in range(1, 513):
            groups[x % mod].append(x)
        result = []
        while any(groups):
            for g in groups:
                if g:
                    result.append(g.pop(0))
        return result
    
    orderings.append(get_order_by_mod(7))
    orderings.append(get_order_by_mod(5))
    
    # Ordering 5: powers of 2 interleaved
    p2 = [2**i for i in range(9) if 2**i <= 512]
    rest = [x for x in range(1, 513) if x not in set(p2)]
    orderings.append(p2 + rest)
    
    # Ordering 6: random
    rnd = list(range(1, 513))
    random.seed(42)
    random.shuffle(rnd)
    orderings.append(rnd)
    
    # Ordering 7: primes first
    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # Ordering 8: odds then evens
    orderings.append([x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0])
    
    best_greedy = []
    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if time.time() > deadline - 20:
                break
            if x in selected_set:
                continue
            
            if can_add_element(x, selected):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_greedy):
            best_greedy = selected[:]
            if len(best_greedy) >= 48:
                break
    
    # Phase 2: Simulated Annealing to escape local optima
    # State: a set of selected elements
    # Move: swap one element in the set with one element outside
    # Objective: maximize |set| - penalty * violations
    
    if time.time() < deadline - 10 and len(best_greedy) > 0:
        # Start from the best greedy solution
        current_set = list(best_greedy)
        current_set_set = set(current_set)
        current_penalty = count_violations(current_set)
        current_score = len(current_set) - current_penalty * 10  # Heavy penalty for violations
        
        best_sa_set = current_set[:]
        best_sa_score = current_score
        best_sa_penalty = current_penalty
        
        all_values = list(range(1, 513))
        
        # Simulated Annealing parameters
        T_start = 5.0
        T_end = 0.1
        n_iterations = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        
        T = T_start
        rng = random.Random(123)
        
        for it in range(n_iterations):
            if time.time() > deadline - 5:
                break
            
            # Generate a random move: swap one element in set with one outside
            if len(current_set) > 0:
                in_idx = rng.randint(0, len(current_set) - 1)
                out_pool = [x for x in all_values if x not in current_set_set]
                if not out_pool:
                    break
                out_val = rng.choice(out_pool)
                
                old_in = current_set[in_idx]
                
                # Apply move
                current_set[in_idx] = out_val
                current_set_set.discard(old_in)
                current_set_set.add(out_val)
                
                new_penalty = count_violations(current_set)
                new_score = len(current_set) - new_penalty * 10
                
                delta = new_score - current_score
                
                # Accept if better or with probability exp(-delta/T)
                if delta >= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_penalty = new_penalty
                    current_score = new_score
                else:
                    # Reject: revert
                    current_set[in_idx] = old_in
                    current_set_set.discard(out_val)
                    current_set_set.add(old_in)
            
            # Update best
            if current_score > best_sa_score or (current_score == best_sa_score and current_penalty < best_sa_penalty):
                best_sa_score = current_score
                best_sa_penalty = current_penalty
                best_sa_set = current_set[:]
            
            # Cool down
            T *= cooling_rate
        
        # If SA found a valid set with more elements, use it
        if best_sa_penalty == 0 and len(best_sa_set) > len(best_greedy):
            best_greedy = best_sa_set[:]
    
    # Phase 3: Final greedy improvement from the best set
    if best_greedy and time.time() < deadline - 5:
        improved = best_greedy[:]
        improved_set = set(improved)
        remaining = [x for x in range(1, 513) if x not in improved_set]
        
        for x in remaining:
            if time.time() > deadline - 2:
                break
            if can_add_element(x, improved):
                improved.append(x)
                improved_set.add(x)
        
        if len(improved) > len(best_greedy):
            best_greedy = improved[:]
    
    return best_greedy
# EVOLVE-BLOCK-END
