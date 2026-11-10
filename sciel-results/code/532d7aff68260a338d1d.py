import time

def _sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, i, val):
        # i is 1-based index
        while i <= self.n:
            if val > self.tree[i]:
                self.tree[i] = val
            i += i & -i
            
    def query(self, i):
        # max in [1, i]
        res = 0
        while i > 0:
            if self.tree[i] > res:
                res = self.tree[i]
            i -= i & -i
        return res

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = _sieve_primes(10000)
    n = len(primes)
    max_gap = 10000
    
    if n < 2:
        return primes[:]

    # dp[i] will store the best length ending at i? No.
    # We need to track the predecessor to reconstruct.
    # Let's store for each i, a list of (prev_index, length) or just best length and prev.
    # But we need to query based on gap.
    
    # For each j, we maintain a Fenwick Tree of size 10001.
    # tree[j] is a FenwickTree.
    # tree[j].update(gap, length)
    # tree[j].query(gap) -> max length ending at j with last gap <= gap.
    
    # We also need to store the argmax to reconstruct.
    # This is tricky with BIT. BIT only stores max value.
    # We need to store the index `k` that gave the max value.
    
    # Let's modify FenwickTree to store (value, index).
    
    class FenwickTreeWithIndex:
        def __init__(self, size):
            self.n = size
            self.tree_val = [0] * (self.n + 1)
            self.tree_idx = [-1] * (self.n + 1)
            
        def update(self, i, val, idx):
            while i <= self.n:
                if val > self.tree_val[i] or (val == self.tree_val[i] and idx < self.tree_idx[i]):
                    self.tree_val[i] = val
                    self.tree_idx[i] = idx
                i += i & -i
                
        def query(self, i):
            res_val = 0
            res_idx = -1
            while i > 0:
                if self.tree_val[i] > res_val or (self.tree_val[i] == res_val and self.tree_idx[i] < res_idx):
                    res_val = self.tree_val[i]
                    res_idx = self.tree_idx[i]
                i -= i & -i
            return res_val, res_idx

    # Initialize trees
    trees = [None] * n
    for j in range(n):
        trees[j] = FenwickTreeWithIndex(max_gap)

    dp = [0] * n
    prev = [-1] * n
    
    best_len = 2
    best_end = 1
    
    for i in range(1, n):
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            # Query trees[j] for max length with last gap <= gap
            val, idx = trees[j].query(gap)
            
            # The sequence length ending at i with previous j is:
            # If val > 0, it means we extended a sequence ending at j.
            # The length is val + 1.
            # If val == 0, it means no sequence ending at j had gap <= gap?
            # Wait, trees[j] stores sequences ending at j.
            # The "last gap" of a sequence ending at j is p_j - p_k.
            # We are extending to i, so new gap is p_i - p_j.
            # Condition: p_j - p_k <= p_i - p_j.
            # So we query trees[j] for max length with gap <= p_i - p_j.
            # If found, new length = val + 1.
            # If not found (val=0), we start a new sequence of length 2 (j, i).
            
            if val > 0:
                new_len = val + 1
                prev_i = idx # This idx is the k such that p_j - p_k was the gap.
                # Wait, we need to store prev for i.
                # prev[i] should be j.
                # And we need to be able to trace back from j to k.
                # The tree stores the index of the *previous* element of the sequence ending at j?
                # No, the tree for j stores updates from dp[j].
                # When we updated trees[j], we passed `idx` which was `prev[j]`?
                # Let's trace.
                
                # When processing j, we updated trees[j].
                # The update was for the sequence ending at j.
                # The value was dp[j].
                # The index passed was prev[j].
                # So query returns (dp[j], prev[j]).
                
                if new_len > dp[i]:
                    dp[i] = new_len
                    prev[i] = j
                    
                if new_len > best_len:
                    best_len = new_len
                    best_end = i
            else:
                # Start new sequence
                if 2 > dp[i]:
                    dp[i] = 2
                    prev[i] = j
                if 2 > best_len:
                    best_len = 2
                    best_end = i
            
            # Update trees[i] with the state ending at i
            # The state is: length dp[i], last gap gap.
            # But we need to update *after* we finish all j for this i?
            # No, trees[i] should reflect the best states ending at i.
            # We can update trees[i] immediately.
            # But we must ensure we don't use trees[i] to compute dp[i] (which we don't, we use trees[j]).
            
            # However, if we update trees[i] inside the j loop,
            # does it affect subsequent j? No, we query trees[j] where j < i.
            
            # We need to update trees[i] with (gap, dp[i], prev[i])?
            # Wait, the gap stored in trees[i] should be the gap of the sequence ending at i.
            # The gap is p_i - p_j.
            # The length is dp[i].
            # The index to store is prev[i] (which is j).
            
            # But dp[i] might be updated multiple times.
            # We should only update trees[i] with the *best* dp[i]?
            # No, we need to update for *each* possible last gap.
            # The BIT update is `max`.
            # So we can update trees[i] with (gap, dp[i], j) for each j.
            # The BIT will keep the max length for each gap prefix.
            
            trees[i].update(gap, dp[i], j)
            
        if time.time() > deadline:
            break

    # Reconstruct
    seq = []
    cur = best_end
    while cur != -1:
        seq.append(primes[cur])
        cur = prev[cur]
    seq.reverse()
    
    return seq
