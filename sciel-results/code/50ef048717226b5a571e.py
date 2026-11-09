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

    # Strategy: CSP search for a 3x3 grid of distinct primes
    # This guarantees score 9 if found.
    
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
    
    # We define the grid P[i][j] = A[i] + B[j]
    # P[i][j] must be prime.
    # Additive structure: P[i][j] + P[k][l] = P[i][l] + P[k][j]
    # Let's fix P[0][0], P[0][1], P[0][2], P[1][0], P[2][0].
    # Then P[1][1] = P[1][0] + P[0][1] - P[0][0]
    #      P[1][2] = P[1][0] + P[0][2] - P[0][0]
    #      P[2][1] = P[2][0] + P[0][1] - P[0][0]
    #      P[2][2] = P[2][0] + P[0][2] - P[0][0]
    
    # To optimize, we can iterate over differences.
    # But direct iteration over primes is fine if we prune early.
    
    found = False
    # Iterate over P00, P01, P02 for the first row
    # And P10, P20 for the first column (excluding P00)
    
    # Pruning: P00, P01, P02, P10, P20 must be in [2, 256]
    # Also, the derived values must be in [2, 256] and prime.
    
    # To make it faster, we can iterate over smaller primes first? 
    # Or just iterate all. 54 primes in [2, 256]. 54^5 is ~ 450 million. Too slow.
    # We need to be smarter.
    
    # Let's iterate over P00, P01, P02.
    # For each, we need P10, P20 such that derived values are primes.
    # Let's precompute for a given (P00, P01, P02) which (P10, P20) work?
    # No, just nested loops with pruning.
    
    # Let's try to reduce the search space.
    # Note that P11 - P10 = P01 - P00 = d1
    # P12 - P10 = P02 - P00 = d2
    # So P11 = P10 + d1, P12 = P10 + d2.
    # Similarly P21 = P20 + d1, P22 = P20 + d2.
    
    # So for fixed d1, d2 (differences), we need:
    # P00, P00+d1, P00+d2 are primes.
    # P10, P10+d1, P10+d2 are primes.
    # P20, P20+d1, P20+d2 are primes.
    # And all 9 are distinct.
    
    # This is a much better formulation!
    # Iterate over d1, d2.
    # d1, d2 > 0 (to ensure distinct B's, though we can check later).
    # Actually B's are distinct if d1 != 0, d2 != 0, d1 != d2.
    # A's are distinct if P10 != P00, P20 != P00, P10 != P20.
    
    # Max d1, d2 is such that P+d <= 256. So d <= 254.
    # Number of pairs (d1, d2) is ~ 254^2 / 2 ~ 30k.
    # For each (d1, d2), we need to find P00, P10, P20 such that:
    # P00, P00+d1, P00+d2 are prime.
    # P10, P10+d1, P10+d2 are prime.
    # P20, P20+d1, P20+d2 are prime.
    
    # We can precompute for each d1, d2 the list of valid starting primes P such that P, P+d1, P+d2 are prime.
    # Let valid_starts[d1][d2] = [p1, p2, ...]
    
    # Then we just need to pick 3 distinct values from this list for P00, P10, P20.
    # And then check if the resulting 9 primes are distinct.
    # Wait, if P00, P10, P20 are distinct, are the 9 primes distinct?
    # P[i][j] = P[i][0] + d_j (where d_0=0, d_1=d1, d_2=d2)
    # If P[i][0] are distinct, and d_j are distinct, the sums are distinct?
    # Suppose P[i][j] = P[k][l].
    # P[i][0] + d_j = P[k][0] + d_l.
    # If i=k, then d_j = d_l => j=l.
    # If i!=k, then P[i][0] - P[k][0] = d_l - d_j.
    # This is possible. So we must check distinctness of the 9 sums.
    
    # This approach is O(1) per (d1, d2) after precomputation, or O(N^3) where N is count of valid starts.
    # Precomputation:
    # For each d1 in 1..254:
    #   For each d2 in d1+1..254 (to avoid symmetry and ensure d1!=d2):
    #     Find all p such that p, p+d1, p+d2 are prime.
    
    # Let's do this.
    
    max_sum = 256
    valid_starts = {}
    
    for d1 in range(1, max_sum):
        if time.time() > deadline - 5:
            break
        for d2 in range(d1 + 1, max_sum):
            if time.time() > deadline - 5:
                break
            starts = []
            # p must be in [2, 256 - d2]
            for p in range(2, max_sum - d2 + 1):
                if p in prime_set and (p + d1) in prime_set and (p + d2) in prime_set:
                    starts.append(p)
            if len(starts) >= 3:
                valid_starts[(d1, d2)] = starts
    
    # Now search for a solution
    for (d1, d2), starts in valid_starts.items():
        if time.time() > deadline - 5:
            break
        if found:
            break
            
        # We need to pick 3 distinct starts P00, P10, P20 from `starts`
        # such that the 9 generated primes are distinct.
        # To speed up, we can just pick the first 3 and check, or iterate combinations.
        # Since we just need ONE solution, we can try a few random combinations or just the first few.
        # Let's iterate combinations of 3 from starts.
        # If starts is large, this might be slow. But usually small.
        
        # Optimization: Check if any 3 distinct starts produce distinct 9 sums.
        # Let's just try the first 3, then next 3, etc.
        
        n_starts = len(starts)
        if n_starts < 3:
            continue
            
        # Try a few random subsets of 3 to avoid exhaustive search if many
        # Or just iterate.
        
        # Let's iterate i, j, k
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
                    
                    # Generate the 9 primes
                    P = [
                        [P00, P00 + d1, P00 + d2],
                        [P10, P10 + d1, P10 + d2],
                        [P20, P20 + d1, P20 + d2]
                    ]
                    
                    # Check distinctness
                    all_primes = [P[i][j] for i in range(3) for j in range(3)]
                    if len(set(all_primes)) != 9:
                        continue
                        
                    # All are prime by construction, but let's double check bounds
                    if any(p < 2 or p > 256 for p in all_primes):
                        continue
                        
                    # Now solve for A and B
                    # B0 = x
                    # A0 = P00 - x
                    # B1 = x + d1
                    # B2 = x + d2
                    # A1 = P10 - x
                    # A2 = P20 - x
                    
                    # Constraints on x:
                    # 1 <= P00 - x <= 128  => P00 - 128 <= x <= P00 - 1
                    # 1 <= x <= 128
                    # 1 <= x + d1 <= 128 => 1 - d1 <= x <= 128 - d1
                    # 1 <= x + d2 <= 128 => 1 - d2 <= x <= 128 - d2
                    # 1 <= P10 - x <= 128 => P10 - 128 <= x <= P10 - 1
                    # 1 <= P20 - x <= 128 => P20 - 128 <= x <= P20 - 1
                    
                    x_min = max(1, P00 - 128, 1 - d1, 1 - d2, P10 - 128, P20 - 128)
                    x_max = min(128, P00 - 1, 128 - d1, 128 - d2, P10 - 1, P20 - 1)
                    
                    if x_min > x_max:
                        continue
                        
                    # We need A = {P00-x, P10-x, P20-x} to be distinct
                    # This is guaranteed if P00, P10, P20 are distinct.
                    # And B = {x, x+d1, x+d2} to be distinct
                    # This is guaranteed if d1, d2 are distinct and non-zero.
                    
                    # So any x in [x_min, x_max] works?
                    # Yes, distinctness of A and B is preserved for any x.
                    # We just need to pick one.
                    x = x_min
                    
                    A = [P00 - x, P10 - x, P20 - x]
                    B = [x, x + d1, x + d2]
                    
                    # Final verification
                    if len(set(A)) != 3 or len(set(B)) != 3:
                        continue
                    if any(v < 1 or v > 128 for v in A + B):
                        continue
                    
                    # Check pairwise coprime (should be true for distinct primes)
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
