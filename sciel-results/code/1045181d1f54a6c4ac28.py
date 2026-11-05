import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute prime factors for numbers 2..256
    prime_factors = [set() for _ in range(257)]
    for i in range(2, 257):
        if not prime_factors[i]:  # i is prime
            for j in range(i, 257, i):
                prime_factors[j].add(i)
    
    def check_pairwise_coprime(sums_list):
        """Check if all elements in sums_list are pairwise coprime."""
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[sums_list[i]] & prime_factors[sums_list[j]]:
                    return False
        return True
    
    def max_coprime_subset(sums_list):
        """Find the maximum size subset of sums_list that is pairwise coprime.
        Try multiple greedy orderings to get a good approximation."""
        if not sums_list:
            return 0
        
        unique_sums = sorted(set(sums_list))
        n = len(unique_sums)
        
        if n <= 9:
            # Brute force for small sets
            best = 0
            for r in range(n, 0, -1):
                found = False
                for combo in combinations(range(n), r):
                    subset = [unique_sums[i] for i in combo]
                    if check_pairwise_coprime(subset):
                        best = r
                        found = True
                        break
                if found:
                    break
            return best
        
        # For larger sets, try multiple greedy strategies
        best_size = 0
        
        # Strategy 1: sort by number of prime factors (ascending)
        order1 = sorted(unique_sums, key=lambda x: len(prime_factors[x]))
        selected = []
        for s in order1:
            compatible = True
            for t in selected:
                if prime_factors[s] & prime_factors[t]:
                    compatible = False
                    break
            if compatible:
                selected.append(s)
        best_size = max(best_size, len(selected))
        
        # Strategy 2: sort by value (ascending)
        order2 = sorted(unique_sums)
        selected = []
        for s in order2:
            compatible = True
            for t in selected:
                if prime_factors[s] & prime_factors[t]:
                    compatible = False
                    break
            if compatible:
                selected.append(s)
        best_size = max(best_size, len(selected))
        
        # Strategy 3: sort by value (descending)
        order3 = sorted(unique_sums, reverse=True)
        selected = []
        for s in order3:
            compatible = True
            for t in selected:
                if prime_factors[s] & prime_factors[t]:
                    compatible = False
                    break
            if compatible:
                selected.append(s)
        best_size = max(best_size, len(selected))
        
        # Strategy 4: random ordering, try a few times
        for _ in range(5):
            order4 = unique_sums[:]
            random.shuffle(order4)
            selected = []
            for s in order4:
                compatible = True
                for t in selected:
                    if prime_factors[s] & prime_factors[t]:
                        compatible = False
                        break
                if compatible:
                    selected.append(s)
            best_size = max(best_size, len(selected))
        
        return best_size
    
    def score(A, B):
        """Count distinct sums that can be made pairwise coprime (maximum subset size)."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if not sums:
            return 0
        
        return max_coprime_subset(list(sums))
    
    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)
    
    def hill_climb(A, B, max_time):
        """Full neighborhood hill climbing: try all 128 values for each position."""
        cur_A = A[:]
        cur_B = B[:]
        cur_score = score(cur_A, cur_B)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try improving A
            for i in range(3):
                old_val = cur_A[i]
                best_new_val = old_val
                best_new_score = cur_score
                
                for new_val in range(1, 129):
                    if new_val == old_val or new_val in cur_A:
                        continue
                    cur_A[i] = new_val
                    new_score = score(cur_A, cur_B)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    if new_score > 8:  # Early exit if we're close to max
                        break
                cur_A[i] = best_new_val
                if best_new_val != old_val:
                    cur_score = best_new_score
                    improved = True
            
            # Try improving B
            for j in range(3):
                old_val = cur_B[j]
                best_new_val = old_val
                best_new_score = cur_score
                
                for new_val in range(1, 129):
                    if new_val == old_val or new_val in cur_B:
                        continue
                    cur_B[j] = new_val
                    new_score = score(cur_A, cur_B)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    if new_score > 8:  # Early exit if we're close to max
                        break
                cur_B[j] = best_new_val
                if best_new_val != old_val:
                    cur_score = best_new_score
                    improved = True
        
        return cur_A, cur_B, cur_score
    
    # Strategy 1: Parity-restricted search
    # For all sums to be pairwise coprime, no two sums can share factor 2.
    # So all sums must be the same parity. Since a+b parity = parity(a) XOR parity(b),
    # all a's must have same parity and all b's must have opposite parity.
    
    for a_parity in [1, 0]:  # 1=odd, 0=even
        b_parity = 1 - a_parity
        
        # Generate initial candidates with correct parity
        if a_parity == 1:
            a_candidates = list(range(1, 129, 2))
        else:
            a_candidates = list(range(2, 129, 2))
        
        if b_parity == 1:
            b_candidates = list(range(1, 129, 2))
        else:
            b_candidates = list(range(2, 129, 2))
        
        # Try multiple random starts with hill climbing
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            cur_A = sorted(random.sample(a_candidates, 3))
            cur_B = sorted(random.sample(b_candidates, 3))
            
            cur_A, cur_B, cur_score = hill_climb(cur_A, cur_B, deadline - 10)
            
            if cur_score > best_score:
                best_score = cur_score
                best_A = cur_A[:]
                best_B = cur_B[:]
            
            if best_score == 9:
                return [best_A, best_B]
    
    # Strategy 2: Unrestricted search from current best
    if best_score < 9:
        cur_A = best_A[:]
        cur_B = best_B[:]
        cur_score = best_score
        
        # Do a few more rounds of hill climbing from the best
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            
            cur_A, cur_B, cur_score = hill_climb(cur_A, cur_B, deadline - 5)
            
            if cur_score > best_score:
                best_score = cur_score
                best_A = cur_A[:]
                best_B = cur_B[:]
            
            if best_score == 9:
                break
            
            # If stuck, try a random restart
            if round_num % 3 == 2:
                cur_A = sorted(random.sample(range(1, 129), 3))
                cur_B = sorted(random.sample(range(1, 129), 3))
    
    # Strategy 3: Final polish - exhaustive local search on the best solution
    if best_score < 9 and time.time() < deadline - 2:
        # Try all pairs of values for each position
        for i in range(3):
            for j in range(3):
                if time.time() > deadline - 2:
                    break
                
                old_Ai = best_A[i]
                old_Bj = best_B[j]
                
                best_new_Ai = old_Ai
                best_new_Bj = old_Bj
                best_pair_score = best_score
                
                for new_Ai in range(1, 129):
                    if new_Ai == old_Ai or new_Ai in best_A:
                        continue
                    for new_Bj in range(1, 129):
                        if new_Bj == old_Bj or new_Bj in best_B:
                            continue
                        best_A[i] = new_Ai
                        best_B[j] = new_Bj
                        new_score = score(best_A, best_B)
                        if new_score > best_pair_score:
                            best_pair_score = new_score
                            best_new_Ai = new_Ai
                            best_new_Bj = new_Bj
                        if new_score == 9:
                            break
                    best_A[i] = old_Ai
                    best_B[j] = old_Bj
                    if best_pair_score == 9:
                        break
                
                best_A[i] = best_new_Ai
                best_B[j] = best_new_Bj
                if best_pair_score > best_score:
                    best_score = best_pair_score
                    if best_score == 9:
                        break
                best_B[j] = best_new_Bj
            if best_score == 9:
                break
    
    return [best_A, best_B]
