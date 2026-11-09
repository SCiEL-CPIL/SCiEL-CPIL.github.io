import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute primes and exponent vectors
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

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    exp_vecs = {}
    for n in all_numbers:
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

    # Helper to add/subtract vectors
    def add_vec(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
    
    def sub_vec(v1, v2):
        return tuple(a - b for a, b in zip(v1, v2))

    class SequenceState:
        def __init__(self, seq):
            self.seq = seq[:]
            self.n = len(self.seq)
            self.prefix = [tuple([0]*num_primes)]
            for x in self.seq:
                self.prefix.append(add_vec(self.prefix[-1], exp_vecs[x]))
            self.diff_set = set()
            self._rebuild_diff_set()
            
        def _rebuild_diff_set(self):
            self.diff_set = set()
            for i in range(self.n + 1):
                pi = self.prefix[i]
                for j in range(i + 1, self.n + 1):
                    pj = self.prefix[j]
                    self.diff_set.add(sub_vec(pj, pi))
                    
        def get_diffs(self, i, j):
            """Get set of differences for subarray indices [i, j) in prefix terms? 
               No, we need differences P_b - P_a for a in [0, k], b in [k+1, n] etc.
               Let's just compute on the fly for insertion check."""
            pass

        def try_insert(self, x, k):
            """Try to insert x at position k. Returns True if valid, and updates state if so."""
            if k < 0 or k > self.n:
                return False
            if k < self.n and self.seq[k] <= x:
                return False
            if k > 0 and self.seq[k-1] >= x:
                return False
                
            ex = exp_vecs[x]
            
            # Compute new differences D_new
            # a in [0, k], b in [k+1, n] (indices in OLD prefix)
            # New diff = P_old[b-1] - P_old[a] + ex
            # Note: P_old indices are 0..n
            # The new sequence has length n+1.
            # Prefix P'_j = P_j for j<=k.
            # P'_j = P_{j-1} + ex for j>k.
            # Diff P'_b - P'_a for a<=k < b (new indices)
            # If a_new <= k, b_new > k:
            #   P'_a = P_old[a]
            #   P'_b = P_old[b-1] + ex
            #   Diff = P_old[b-1] - P_old[a] + ex
            # Range of a_new: 0..k
            # Range of b_new: k+1..n+1
            # So a_old = a_new, b_old = b_new - 1.
            # Range of a_old: 0..k
            # Range of b_old: k..n
            
            new_diffs = set()
            for a in range(k + 1): # 0..k
                pa = self.prefix[a]
                for b in range(k, self.n + 1): # k..n
                    pb = self.prefix[b]
                    d = add_vec(sub_vec(pb, pa), ex)
                    if d in new_diffs:
                        return False
                    new_diffs.add(d)
                    
            # Check against existing diffs that remain
            # Remaining diffs are those with (a,b) both <= k or both > k (in new indexing)
            # In old indexing:
            # Left part: a < b <= k. Diffs: P_b - P_a.
            # Right part: k < a < b <= n. Diffs: P_b - P_a. (These correspond to P'_{b+1} - P'_{a+1})
            
            # We can just check if any d in new_diffs is in self.diff_set 
            # BUT self.diff_set contains the spanning diffs which are being removed.
            # So we must check against (self.diff_set - spanning_diffs).
            
            # Compute spanning diffs to exclude
            spanning = set()
            for a in range(k + 1): # 0..k
                pa = self.prefix[a]
                for b in range(k + 1, self.n + 1): # k+1..n
                    pb = self.prefix[b]
                    spanning.add(sub_vec(pb, pa))
            
            valid_old_diffs = self.diff_set - spanning
            
            for d in new_diffs:
                if d in valid_old_diffs:
                    return False
            
            # If valid, update state
            # 1. Update seq
            self.seq.insert(k, x)
            self.n += 1
            
            # 2. Update prefix
            new_prefix = self.prefix[:k+1]
            last = new_prefix[-1]
            for i in range(k, self.n): # old indices k..n-1 correspond to new k+1..n
                last = add_vec(last, exp_vecs[self.seq[i-1]]) # Wait, self.seq is already updated?
                # Let's rebuild prefix from scratch, it's O(n^2) which is fine for n<60
                pass
            
            # Rebuilding is safer and fast enough
            self.prefix = [tuple([0]*num_primes)]
            for val in self.seq:
                self.prefix.append(add_vec(self.prefix[-1], exp_vecs[val]))
                
            # 3. Update diff_set
            # Remove spanning, add new_diffs
            self.diff_set = valid_old_diffs | new_diffs
            
            return True

        def try_remove(self, k):
            """Remove element at position k."""
            if k < 0 or k >= self.n:
                return False
            x = self.seq[k]
            ex = exp_vecs[x]
            
            # Current diffs involving x?
            # We are removing x.
            # Old sequence length n.
            # New sequence length n-1.
            # Prefix P'_j = P_j for j<=k.
            # P'_j = P_{j+1} - ex for j>k.
            
            # Diffs in new sequence:
            # 1. Both <= k: P_b - P_a (old)
            # 2. Both > k: P'_{b} - P'_{a} = (P_{b+1}-ex) - (P_{a+1}-ex) = P_{b+1} - P_{a+1} (old)
            # 3. Spanning: a <= k < b (new).
            #    P'_b - P'_a = (P_{b+1}-ex) - P_a = P_{b+1} - P_a - ex.
            #    Range a: 0..k, b: k+1..n-1 (new index) -> old b+1: k+1..n.
            
            new_diffs = set()
            for a in range(k + 1):
                pa = self.prefix[a]
                for b in range(k + 1, self.n): # new b is k+1..n-1, so old b+1 is k+1..n
                    pb_next = self.prefix[b + 1]
                    d = sub_vec(sub_vec(pb_next, pa), ex)
                    if d in new_diffs:
                        return False
                    new_diffs.add(d)
            
            # Old diffs that remain:
            # Left: a < b <= k
            # Right: k < a < b <= n-1 (old indices k+1..n-1? No, old indices k+1..n-1 correspond to new k..n-2)
            # Let's just rebuild diff_set after modification.
            
            self.seq.pop(k)
            self.n -= 1
            self.prefix = [tuple([0]*num_primes)]
            for val in self.seq:
                self.prefix.append(add_vec(self.prefix[-1], exp_vecs[val]))
            self._rebuild_diff_set()
            return True

    # Initial sequence: Primes
    init_seq = primes[:]
    state = SequenceState(init_seq)
    best_seq = state.seq[:]
    best_len = len(best_seq)
    
    used = set(best_seq)
    remaining = [x for x in all_numbers if x not in used]
    
    def local_search(state, max_iters=2000):
        seq = state.seq[:]
        s = SequenceState(seq)
        used = set(seq)
        remaining = [x for x in all_numbers if x not in used]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try Insertion
            if remaining:
                random.shuffle(remaining)
                for x in remaining:
                    # Try all positions
                    for k in range(s.n + 1):
                        # Quick check bounds
                        if k < s.n and s.seq[k] <= x:
                            continue
                        if k > 0 and s.seq[k-1] >= x:
                            continue
                        
                        # Snapshot
                        old_seq = s.seq[:]
                        old_prefix = s.prefix[:]
                        old_diffs = s.diff_set.copy()
                        
                        if s.try_insert(x, k):
                            # Success
                            used.add(x)
                            remaining.remove(x)
                            improved = True
                            break
                        else:
                            # Rollback
                            s.seq = old_seq[:]
                            s.prefix = old_prefix[:]
                            s.diff_set = old_diffs
                            s.n = len(s.seq)
                    if improved:
                        break
            
            # Try Swap/Move
            if not improved:
                # Try moving an element from seq to a different position
                # Or swapping with remaining?
                # Let's try removing one and inserting another (2-opt style)
                # This is expensive.
                pass
                
            if not improved:
                break
                
        return s.seq

    # Run multiple restarts
    for i in range(20):
        if time.time() > deadline:
            break
        
        # Randomize initial state slightly?
        # Start with primes, then add random composites
        seq = primes[:]
        s = SequenceState(seq)
        
        used = set(seq)
        rem = [x for x in all_numbers if x not in used]
        random.shuffle(rem)
        
        for x in rem:
            for k in range(s.n + 1):
                if k < s.n and s.seq[k] <= x:
                    continue
                if k > 0 and s.seq[k-1] >= x:
                    continue
                if s.try_insert(x, k):
                    break
        
        if len(s.seq) > best_len:
            best_len = len(s.seq)
            best_seq = s.seq[:]
            
        # Local search
        final_seq = local_search(s)
        if len(final_seq) > best_len:
            best_len = len(final_seq)
            best_seq = final_seq[:]

    return best_seq
