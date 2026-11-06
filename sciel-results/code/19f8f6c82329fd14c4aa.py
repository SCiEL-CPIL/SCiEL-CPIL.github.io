import math
import random
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _validate(A, B):
    # A and B must be 3 distinct integers in [1,128]
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    for x in A:
        if x < 1 or x > 128:
            return False
    for x in B:
        if x < 1 or x > 128:
            return False
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(list(sums)):
        return False
    return True

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def construct():
    deadline = time.time() + 80.0  # 80s limit, 10s safety margin
    
    # Precompute primes up to 256
    primes = [n for n in range(2, 257) if _is_prime(n)]
    prime_set = set(primes)
    
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Precompute for each b, the set of a values in [1,128] such that a+b is prime
    valid_a_for_b = {}
    for b in range(1, 129):
        valid_a_for_b[b] = set()
        for p in primes:
            a = p - b
            if 1 <= a <= 128:
                valid_a_for_b[b].add(a)
    
    # Phase 1: Structured search over small B values
    # Iterate through combinations of small B values
    # This is much faster than random search if a solution exists in small values
    
    # Try B values from 1 to 30 first
    small_range = list(range(1, 31))
    import itertools
    for B_tuple in itertools.combinations(small_range, 3):
        B = list(B_tuple)
        valid_asis = valid_a_for_b[B[0]] & valid_a_for_b[B[1]] & valid_a_for_b[B[2]]
        if len(valid_asis) < 3:
            continue
        
        valid_asis_list = sorted(valid_asis)
        # Try to pick 3 a values that give 9 distinct sums
        # We just need to check if any 3 from valid_asis work
        # Since we want 9 distinct sums, we should pick a's that are "spread out" or just check the first few combinations
        found = False
        # To ensure 9 distinct sums, a_i + b_j must be distinct.
        # This is guaranteed if the difference sets of A and B are disjoint.
        # But checking all 3-combinations from valid_asis is expensive if valid_asis is large.
        # However, valid_asis is usually small.
        # Let's just check the first 10 valid a's for combinations
        candidates = valid_asis_list[:10]
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                for k in range(j+1, len(candidates)):
                    A = [candidates[i], candidates[j], candidates[k]]
                    sums = set()
                    all_prime = True
                    for a in A:
                        for b in B:
                            s = a + b
                            sums.add(s)
                            if s not in prime_set:
                                all_prime = False
                                break
                        if not all_prime:
                            break
                    if all_prime and len(sums) == 9:
                        best_score = 9
                        best_result = [A, B]
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    
    # Phase 2: If not found in small range, try random search
    if best_score < 9 and time.time() < deadline:
        random.seed(42)
        tried_B = set()
        attempts = 0
        while time.time() < deadline and attempts < 200000:
            attempts += 1
            B = tuple(sorted(random.sample(range(1, 129), 3)))
            if B in tried_B:
                continue
            tried_B.add(B)
            
            valid_asis = valid_a_for_b[B[0]] & valid_a_for_b[B[1]] & valid_a_for_b[B[2]]
            if len(valid_asis) < 3:
                continue
            
            valid_asis_list = sorted(valid_asis)
            candidates = valid_asis_list[:10]
            found = False
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    for k in range(j+1, len(candidates)):
                        A = [candidates[i], candidates[j], candidates[k]]
                        sums = set()
                        all_prime = True
                        for a in A:
                            for b in B:
                                s = a + b
                                sums.add(s)
                                if s not in prime_set:
                                    all_prime = False
                                    break
                            if not all_prime:
                                break
                        if all_prime and len(sums) == 9:
                            best_score = 9
                            best_result = [A, list(B)]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if best_score == 9:
                break

    
    # Phase 2: If we found 9, we're done. Otherwise, try to maximize with pairwise coprime (not necessarily prime)
    if best_score < 9:
        # Random search for pairwise coprime sums
        while time.time() < deadline and attempts < 500000:
            attempts += 1
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if _pairwise_coprime(list(sums)):
                s = len(sums)
                if s > best_score:
                    best_score = s
                    best_result = [A, B]
    
    # Phase 3: Local search around best result
    if best_score < 9 and time.time() < deadline:
        # Try perturbing the best result
        A = list(best_result[0])
        B = list(best_result[1])
        for _ in range(100000):
            if time.time() > deadline:
                break
            # Randomly change one element
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                if len(set(A)) != 3:
                    A[idx] = old
                    continue
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                if len(set(B)) != 3:
                    B[idx] = old
                    continue
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if _pairwise_coprime(list(sums)):
                s = len(sums)
                if s > best_score:
                    best_score = s
                    best_result = [A[:], B[:]]
    
    return best_result
