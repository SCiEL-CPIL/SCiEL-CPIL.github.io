import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    exp_vecs = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
    
    def check_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                tv = tuple(vec)
                if tv in prods:
                    return False
                prods.add(tv)
        return True
    
    def check_insertion(seq, pos, val):
        """Check if inserting val at pos in seq keeps all block products distinct."""
        n = len(seq)
        prefix = [None] * (n + 1)
        prefix[0] = [0] * num_primes
        for i in range(n):
            prefix[i+1] = [prefix[i][k] + exp_vecs[seq[i]][k] for k in range(num_primes)]
        
        existing_prods = set()
        for i in range(n):
            for j in range(i+1, n+1):
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                existing_prods.add(vec)
        
        new_prods = set()
        ev = exp_vecs[val]
        
        for start in range(pos + 1):
            for end in range(pos, n + 1):
                vec = [0] * num_primes
                if start < pos:
                    for k in range(num_primes):
                        vec[k] = prefix[pos][k] - prefix[start][k]
                for k in range(num_primes):
                    vec[k] += ev[k]
                if end > pos:
                    for k in range(num_primes):
                        vec[k] += prefix[end][k] - prefix[pos][k]
                
                tv = tuple(vec)
                if tv in existing_prods or tv in new_prods:
                    return False
                new_prods.add(tv)
        
        return True
    
    def try_append(seq, c):
        if seq and c <= seq[-1]:
            return None
        if check_insertion(seq, len(seq), c):
            return seq + [c]
        return None
    
    seq = []
    for n in range(2, 129):
        if time.time() > deadline:
            break
        new_seq = try_append(seq, n)
        if new_seq is not None:
            seq = new_seq
    
    def local_search(seq, max_iter=2000):
        current = seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            for c in remaining[:]:
                new_seq = None
                for i in range(len(current) + 1):
                    if i > 0 and current[i-1] >= c:
                        continue
                    if i < len(current) and c >= current[i]:
                        continue
                    if check_insertion(current, i, c):
                        new_seq = current[:i] + [c] + current[i:]
                        break
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in range(2, 129) if x not in used]
                    improved = True
                    break
            
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i+1:]
                    for c in remaining:
                        if c <= removed:
                            continue
                        for j in range(len(base) + 1):
                            if j > 0 and base[j-1] >= c:
                                continue
                            if j < len(base) and c >= base[j]:
                                continue
                            if check_insertion(base, j, c):
                                current = base[:j] + [c] + base[j:]
                                used = set(current)
                                remaining = [x for x in range(2, 129) if x not in used]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    improved_seq = local_search(seq)
    if len(improved_seq) > len(seq):
        seq = improved_seq
    
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        candidates = list(range(2, 129))
        primes_list = [x for x in candidates if is_prime(x)]
        composites_list = [x for x in candidates if not is_prime(x)]
        
        strategy = rng.random()
        if strategy < 0.3:
            order = primes_list + composites_list
        elif strategy < 0.6:
            order = composites_list + primes_list
        else:
            order = candidates
        rng.shuffle(order)
        
        s2 = []
        for n in order:
            if time.time() > deadline:
                break
            ns = try_append(s2, n)
            if ns is not None:
                s2 = ns
        
        s2 = local_search(s2, max_iter=500)
        if len(s2) > len(seq):
            seq = s2
    
    return seq
