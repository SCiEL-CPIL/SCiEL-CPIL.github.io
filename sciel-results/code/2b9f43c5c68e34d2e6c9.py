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
    best_score = evaluate(best_A, best_B)  # 5

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

    # Precompute prime sums
    is_prime_sum = [[False] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            s = a + b
            if 2 <= s <= 256 and s in prime_set:
                is_prime_sum[a][b] = True

    # Precompute for each b, the set of a's such that a+b is prime
    prime_a_for_b = [set() for _ in range(129)]
    for b in range(1, 129):
        for a in range(1, 129):
            if is_prime_sum[a][b]:
                prime_a_for_b[b].add(a)
                
    # Precompute for each a, the set of b's such that a+b is prime
    prime_b_for_a = [set() for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            if is_prime_sum[a][b]:
                prime_b_for_a[a].add(b)

    found = False

    # Iterate over pairs (b1, b2)
    for b1 in range(1, 129):
        if time.time() > deadline - 10:
            break
        for b2 in range(b1 + 1, 129):
            if time.time() > deadline - 5:
                break
            
            # V12: a's such that a+b1 and a+b2 are prime
            V12 = prime_a_for_b[b1] & prime_a_for_b[b2]
            if len(V12) < 3:
                continue
                
            # For each b3, check if there are at least 3 a's in V12 such that a+b3 is prime
            # We can precompute for each b3 the intersection V12 & prime_a_for_b[b3]
            # But doing this for all b3 is 128 intersections.
            # Instead, iterate b3 and check size of intersection.
            
            # Optimization: 
            # For a fixed V12, we want to find b3 such that |V12 \cap prime_a_for_b[b3]| >= 3.
            # Let's just iterate b3.
            
            for b3 in range(1, 129):
                if b3 == b1 or b3 == b2:
                    continue
                if time.time() > deadline - 2:
                    break
                
                # C_b3 = {a in V12 | a+b3 is prime}
                C_b3 = V12 & prime_a_for_b[b3]
                if len(C_b3) < 3:
                    continue
                
                # We have at least 3 candidates for a. Pick 3 and check distinctness.
                # To maximize chance of distinct sums, pick a's that are "spread out"?
                # Or just pick the first 3? 
                # The sums are a_i + b_j.
                # Distinctness fails if a_i + b_j = a_k + b_l.
                # Since b's are distinct, this implies a_i - a_k = b_l - b_j.
                # If we pick a's with large differences, it's less likely to collide with small b differences.
                # But let's just try a few combinations from C_b3.
                
                c_list = sorted(C_b3)
                # Try the first 3, then maybe some others if fail?
                # For now, try the first 3. If it fails, try next 3, etc.
                # But limit the number of tries.
                
                # Actually, we can just check all combinations of 3 from C_b3 if it's small.
                # If len(C_b3) is large, we might miss a good one.
                # Let's try a few random samples if len is large, or all if small.
                
                if len(c_list) <= 10:
                    comb_iter = itertools.combinations(c_list, 3)
                else:
                    # Sample 10 random triples
                    comb_iter = []
                    for _ in range(10):
                        comb_iter.append(tuple(sorted(random.sample(c_list, 3))))
                
                for a1, a2, a3 in comb_iter:
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    
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
                        
                if found:
                    break
                    
            if found:
                break
        if found:
            break

    if best_score == 9:
        return [list(best_A), list(best_B)]

    # Fallback: local search
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
