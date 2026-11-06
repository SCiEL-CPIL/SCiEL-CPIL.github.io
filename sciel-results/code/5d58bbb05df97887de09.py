import math
import random
import time
import itertools

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
    vs = list(set(vals))
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _validate(A, B):
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
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
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
    
    # Phase 1: Exhaustive search over small B values with full A candidate check
    # For each B triple, find all A triples where all 9 sums are prime and distinct
    small_range = list(range(1, 50))
    for B_tuple in itertools.combinations(small_range, 3):
        if time.time() > deadline:
            break
        B = list(B_tuple)
        valid_asis = valid_a_for_b[B[0]] & valid_a_for_b[B[1]] & valid_a_for_b[B[2]]
        if len(valid_asis) < 3:
            continue
        
        valid_asis_list = sorted(valid_asis)
        # Check ALL combinations of 3 from valid_asis (usually small set)
        for A_tuple in itertools.combinations(valid_asis_list, 3):
            A = list(A_tuple)
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
                return best_result  # 9 is maximum possible, return immediately
    
    # Phase 2: Broader systematic search over B values
    if best_score < 9 and time.time() < deadline:
        tried_B = set()
        for b1 in range(1, 129):
            if time.time() > deadline:
                break
            for b2 in range(b1 + 1, 129):
                if time.time() > deadline:
                    break
                for b3 in range(b2 + 1, 129):
                    if time.time() > deadline:
                        break
                    B = [b1, b2, b3]
                    key = tuple(B)
                    if key in tried_B:
                        continue
                    tried_B.add(key)
                    
                    valid_asis = valid_a_for_b[b1] & valid_a_for_b[b2] & valid_a_for_b[b3]
                    if len(valid_asis) < 3:
                        continue
                    
                    valid_asis_list = sorted(valid_asis)
                    # Check combinations - limit to first 20 to keep runtime bounded
                    candidates = valid_asis_list[:20]
                    for A_tuple in itertools.combinations(candidates, 3):
                        A = list(A_tuple)
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
                            return best_result
    
    # Phase 3: If still no 9, do random search for pairwise coprime (not necessarily prime) sums
    if best_score < 9 and time.time() < deadline:
        random.seed(123)
        attempts = 0
        while time.time() < deadline and attempts < 200000:
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
    
    # Phase 4: Local search around best result
    if best_score < 9 and time.time() < deadline:
        A = list(best_result[0])
        B = list(best_result[1])
        for _ in range(100000):
            if time.time() > deadline:
                break
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
    
    # Final validation
    if not _validate(best_result[0], best_result[1]):
        # Fallback to a known valid result
        best_result = [[1, 7, 13], [4, 10, 16]]
    
    return best_result
