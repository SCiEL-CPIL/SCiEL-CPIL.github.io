# EVOLVE-BLOCK-START
import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [(-1, -1, -1)] * (self.n + 1) # (len, idx, gap)
        
    def update(self, i, val):
        # val is (len, idx, gap)
        while i <= self.n:
            if val[0] > self.tree[i][0]:
                self.tree[i] = val
            i += i & (-i)
            
    def query(self, i):
        # Returns max state with index <= i
        res = (-1, -1, -1)
        while i > 0:
            if self.tree[i][0] > res[0]:
                res = self.tree[i]
            i -= i & (-i)
        return res

def construct():
    # Generate primes up to 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                for multiple in range(p*p, n + 1, p):
                    is_prime[multiple] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(10000)
    N = len(primes)
    
    if N == 0:
        return []
    
    max_gap = 10000
    ft = FenwickTree(max_gap)
    
    # states[j][gap] = (len, prev_idx, prev_gap)
    states = [{} for _ in range(N)]
    
    best_len = 0
    best_idx = -1
    best_gap = -1
    
    for j in range(N):
        p_j = primes[j]
        current_states = {}
        
        for i in range(j):
            g = p_j - primes[i]
            # Query best state with gap <= g
            prev_state = ft.query(g)
            
            if prev_state[0] != -1:
                prev_len, prev_idx, prev_gap = prev_state
                new_len = prev_len + 1
                # The new state for p_j ends with gap g
                # Previous prime is primes[i] (index i)
                # Previous gap in the sequence ending at i is prev_gap
                new_state = (new_len, i, g)
                
                if g not in current_states:
                    current_states[g] = new_state
                else:
                    if new_state[0] > current_states[g][0]:
                        current_states[g] = new_state
        
        # Update Fenwick Tree with states for p_j
        for g, state in current_states.items():
            ft.update(g, state)
            if state[0] > best_len:
                best_len = state[0]
                best_idx = j
                best_gap = g
        
        states[j] = current_states
        
    # Backtrack
    if best_idx == -1:
        return [primes[0]]
        
    seq = []
    curr_idx = best_idx
    curr_gap = best_gap
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        state = states[curr_idx][curr_gap]
        prev_idx = state[1]
        prev_gap = state[2]
        curr_idx = prev_idx
        curr_gap = prev_gap
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
