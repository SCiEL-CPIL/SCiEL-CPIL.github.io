import itertools
import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s

    def is_pairwise_coprime(nums):
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B) # 5

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(256)
    prime_set = set(primes)
    
    found = False

    # Search for a 3x3 grid of distinct primes P[i][j] = A[i] + B[j]
    # P[i][j] = P[i][0] + d_j
    # d_0 = 0, d_1, d_2
    # B = {x, x+d1, x+d2}
    # A = {P[i][0] - x}
    # Constraints:
    # 1 <= A[i] <= 128
    # 1 <= B[j] <= 128
    
    # d2 must be <= 127 for B to fit (since min x=1, max B=1+d2 <= 128)
    # Also P[i][0] + d2 <= 256
    
    for d1 in range(1, 128):
        if time.time() > deadline - 5:
            break
        for d2 in range(d1 + 1, 128):
            if time.time() > deadline - 5:
                break
            
            # Find all p such that p, p+d1, p+d2 are prime
            starts = []
            # p must be in [2, 256 - d2]
            for p in range(2, 256 - d2 + 1):
                if p in prime_set and (p + d1) in prime_set and (p + d2) in prime_set:
                    starts.append(p)
            
            if len(starts) < 3:
                continue
            
            # We need to pick 3 distinct starts P00, P10, P20 from `starts`
            # such that there exists x satisfying:
            # 1 <= P00 - x <= 128
            # 1 <= P10 - x <= 128
            # 1 <= P20 - x <= 128
            # 1 <= x <= 128
            # 1 <= x + d1 <= 128
            # 1 <= x + d2 <= 128
            
            # x_min = max(1, P00-128, P10-128, P20-128, 1-d1, 1-d2)
            # Since d1, d2 >= 1, 1-d1, 1-d2 <= 0, so max with 1 is 1.
            # x_min = max(1, P00-128, P10-128, P20-128)
            # x_max = min(128, P00-1, P10-1, P20-1, 128-d1, 128-d2)
            
            # To speed up, we can check if the range [min(starts)-128, max(starts)-1]
            # intersects with [1, 128-d2] (since 128-d2 is the tightest upper bound for x from B)
            # Actually, let's just iterate triplets.
            
            n_starts = len(starts)
            
            for i in range(n_starts):
                if time.time() > deadline - 2:
                    break
                if found:
                    break
                for j in range(i + 1, n_starts):
                    if time.time() > deadline - 2:
                        break
                    if found:
                        break
                    for k in range(j + 1, n_starts):
                        if time.time() > deadline - 2:
                            break
                        if found:
                            break
                        
                        P00 = starts[i]
                        P10 = starts[j]
                        P20 = starts[k]
                        
                        # Check distinctness of 9 primes
                        # P[i][j] = starts[idx] + d_j
                        # If starts are distinct and d's are distinct, are sums distinct?
                        # P[i][j] = P[k][l] => starts[i] - starts[k] = d_l - d_j
                        # This is possible. We must check.
                        P = [
                            [P00, P00 + d1, P00 + d2],
                            [P10, P10 + d1, P10 + d2],
                            [P20, P20 + d1, P20 + d2]
                        ]
                        all_primes = [P[i][j] for i in range(3) for j in range(3)]
                        if len(set(all_primes)) != 9:
                            continue
                        
                        # Check bounds for A and B
                        x_min = max(1, P00 - 128, P10 - 128, P20 - 128)
                        x_max = min(128 - d2, P00 - 1, P10 - 1, P20 - 1)
                        
                        if x_min > x_max:
                            continue
                        
                        x = x_min
                        A = [P00 - x, P10 - x, P20 - x]
                        B = [x, x + d1, x + d2]
                        
                        # Final verification
                        if len(set(A)) != 3 or len(set(B)) != 3:
                            continue
                        if any(v < 1 or v > 128 for v in A + B):
                            continue
                        
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a + b)
                        
                        if len(sums) == 9 and is_pairwise_coprime(sums):
                            best_A = A
                            best_B = B
                            best_score = 9
                            found = True
                            break

    if best_score == 9:
        return [list(best_A), list(best_B)]

    # Fallback: try local search for score 8 or better
    def random_set():
        return sorted(random.sample(range(1, 129), 3))
    
    def local_search():
        A = random_set()
        B = random_set()
        curr_score = evaluate(A, B)
        
        for step in range(5000):
            if time.time() > deadline - 2:
                return A, B, curr_score
                
            if curr_score == 9:
                return A, B, curr_score
                
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                while len(set(A)) < 3:
                    A[idx] = random.randint(1, 128)
                A.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    A[idx] = old
                    A.sort()
                else:
                    curr_score = new_score
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                while len(set(B)) < 3:
                    B[idx] = random.randint(1, 128)
                B.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    B[idx] = old
                    B.sort()
                else:
                    curr_score = new_score
                    
        return A, B, curr_score

    for _ in range(20):
        if time.time() > deadline - 2:
            break
        A_ls, B_ls, score_ls = local_search()
        if score_ls > best_score:
            best_score = score_ls
            best_A = A_ls
            best_B = B_ls
            if best_score == 9:
                break

    return [list(best_A), list(best_B)]
