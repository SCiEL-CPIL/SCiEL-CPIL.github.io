import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute prime factorization vectors for numbers 2..128
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
    
    # Exponent vector for each number 2..128
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
    
    # Efficient incremental validity checker
    class ValidityTracker:
        def __init__(self, seq):
            self.seq = list(seq)
            self.n = len(self.seq)
            self.prefix = [[0] * num_primes]
            for i in range(self.n):
                ej = exp_vecs[self.seq[i]]
                prev = self.prefix[-1]
                self.prefix.append([prev[k] + ej[k] for k in range(num_primes)])
            self.prods = set()
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    vec = tuple(self.prefix[j][k] - self.prefix[i][k] for k in range(num_primes))
                    self.prods.add(vec)
        
        def check_append(self, c):
            """Check if appending c keeps validity. Returns set of new products or None."""
            if self.seq and c <= self.seq[-1]:
                return None
            new_n = self.n + 1
            ej = exp_vecs[c]
            new_prefix = [0] * num_primes
            prev = self.prefix[-1]
            for k in range(num_primes):
                new_prefix[k] = prev[k] + ej[k]
            
            new_prods = set()
            for i in range(self.n + 1):
                vec = tuple(new_prefix[k] - self.prefix[i][k] for k in range(num_primes))
                if vec in self.prods or vec in new_prods:
                    return None
                new_prods.add(vec)
            return new_prods
        
        def commit_append(self, c, new_prods):
            """Commit an append after successful check."""
            self.seq.append(c)
            self.n += 1
            self.prefix.append([self.prefix[-1][k] + exp_vecs[c][k] for k in range(num_primes)])
            self.prods.update(new_prods)
        
        def rollback_append(self):
            """Rollback the last append."""
            if self.n > 0:
                self.n -= 1
                self.seq.pop()
                self.prefix.pop()
                # Recompute prods from scratch for safety
                self._recompute_prods()
        
        def _recompute_prods(self):
            self.prods = set()
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    vec = tuple(self.prefix[j][k] - self.prefix[i][k] for k in range(num_primes))
                    self.prods.add(vec)
    
    def check_valid(seq):
        """Check if sequence has all distinct consecutive block products."""
        n = len(seq)
        if n == 0:
            return True
        prefix = [[0] * num_primes]
        for i in range(n):
            ej = exp_vecs[seq[i]]
            prev = prefix[-1]
            prefix.append([prev[k] + ej[k] for k in range(num_primes)])
        
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                if vec in prods:
                    return False
                prods.add(vec)
        return True
    
    def try_append(seq, c):
        """Try to append c to seq (c must be > seq[-1]). Return new seq or None."""
        if seq and c <= seq[-1]:
            return None
        new_seq = seq + [c]
        if check_valid(new_seq):
            return new_seq
        return None
    
    # CSP-style backtracking search
    def backtrack_search(used_set, current_seq, remaining_candidates, depth=0, max_depth=127):
        """Backtracking search to find a valid sequence."""
        if time.time() > deadline:
            return current_seq
        
        if not remaining_candidates:
            return current_seq
        
        # Try each candidate
        for c in remaining_candidates:
            if time.time() > deadline:
                return current_seq
            
            # Check if c can be appended (must be > last element)
            if current_seq and c <= current_seq[-1]:
                continue
            
            tracker = ValidityTracker(current_seq)
            new_prods = tracker.check_append(c)
            if new_prods is not None:
                tracker.commit_append(c, new_prods)
                new_seq = tracker.seq
                new_remaining = [x for x in remaining_candidates if x != c]
                
                # Recurse
                result = backtrack_search(used_set | {c}, new_seq, new_remaining, depth + 1, max_depth)
                if len(result) > len(current_seq):
                    return result
                
                tracker.rollback_append()
        
        return current_seq
    
    # Greedy initial solution
    seq = []
    for n in range(2, 129):
        if time.time() > deadline:
            break
        new_seq = try_append(seq, n)
        if new_seq is not None:
            seq = new_seq
    
    # Local search with efficient tracking
    def local_search(seq, max_iter=3000):
        current = seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try insertions at various positions
            for c in remaining[:]:
                new_seq = None
                for i in range(len(current) + 1):
                    if i > 0 and current[i-1] >= c:
                        continue
                    if i < len(current) and c >= current[i]:
                        continue
                    candidate = current[:i] + [c] + current[i:]
                    if check_valid(candidate):
                        new_seq = candidate
                        break
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in range(2, 129) if x not in used]
                    improved = True
                    break
            
            # Try swaps
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
                            candidate = base[:j] + [c] + base[j:]
                            if check_valid(candidate) and len(candidate) >= len(current):
                                current = candidate
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
    
    # Improve with local search
    improved_seq = local_search(seq)
    if len(improved_seq) > len(seq):
        seq = improved_seq
    
    # Try backtracking search from different starting points
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Create a partial sequence to start backtracking from
        if restart == 0:
            start_seq = seq[:15]  # Start from first 15 elements of current best
        else:
            rng = random.Random(restart * 42)
            # Random subset of size 10-15
            size = rng.randint(10, 15)
            candidates = list(range(2, 129))
            rng.shuffle(candidates)
            start_seq = sorted(candidates[:size])
        
        if not check_valid(start_seq):
            continue
        
        used_set = set(start_seq)
        remaining = [x for x in range(2, 129) if x not in used_set]
        
        result = backtrack_search(used_set, start_seq, remaining)
        if len(result) > len(seq):
            seq = result
    
    # Final local search polish
    final_seq = local_search(seq, max_iter=1000)
    if len(final_seq) > len(seq):
        seq = final_seq
    
    return seq
