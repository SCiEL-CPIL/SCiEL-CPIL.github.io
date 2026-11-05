"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import bisect

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
    
    # L[j] is a dict: gap -> (length, prev_idx, prev_gap)
    L = [dict() for _ in range(N)]
    
    # Data structures for fast prefix max queries
    # sorted_gaps[i]: list of gaps for node i, sorted
    # prefix_max_len[i]: list of max lengths for gaps <= sorted_gaps[i][k]
    # prefix_max_state_idx[i]: list of indices into sorted_states[i] that achieve the prefix max
    # sorted_states[i]: list of (gap, (len, prev_idx, prev_gap)) sorted by gap
    sorted_gaps = [None] * N
    prefix_max_len = [None] * N
    prefix_max_state_idx = [None] * N
    sorted_states = [None] * N
    
    global_max_len = 0
    global_max_idx = -1
    global_max_gap = -1
    
    for j in range(N):
        # Initialize with sequence starting at j (length 1, last gap 0)
        # Note: gap 0 is a convention for "no previous prime"
        L[j][0] = (1, -1, 0)
        
        # Process transitions from all i < j
        for i in range(j):
            g_new = primes[j] - primes[i]
            
            # Query node i for max length with last gap <= g_new
            # If i has no states, skip (but i always has at least state 0)
            if sorted_gaps[i] is None:
                continue
                
            # Binary search for largest gap <= g_new in sorted_gaps[i]
            # sorted_gaps[i] is sorted
            idx = bisect.bisect_right(sorted_gaps[i], g_new) - 1
            
            if idx < 0:
                # No gap in i is <= g_new?
                # But gap 0 is always in i, and g_new >= 0 (primes are increasing)
                # So idx should be at least 0 if g_new >= 0
                # Unless primes[j] == primes[i], which is impossible for i < j
                # So idx >= 0 is expected
                continue
                
            best_len_prev = prefix_max_len[i][idx]
            state_idx = prefix_max_state_idx[i][idx]
            
            # Retrieve the specific state in i that gave this max length
            # sorted_states[i][state_idx] is (gap, (len, prev_idx, prev_gap))
            prev_gap_used = sorted_states[i][state_idx][0]
            
            new_len = best_len_prev + 1
            
            # Update L[j][g_new] if new_len is better
            if g_new not in L[j]:
                L[j][g_new] = (new_len, i, prev_gap_used)
            else:
                old_len = L[j][g_new][0]
                if new_len > old_len:
                    L[j][g_new] = (new_len, i, prev_gap_used)
                    
        # After processing all i, build data structures for j
        # Get all states for j
        states = sorted(L[j].items(), key=lambda x: x[0])
        sorted_states[j] = states
        sorted_gaps[j] = [s[0] for s in states]
        
        # Compute prefix maxes
        n_states = len(states)
        pml = [0] * n_states
        pmsi = [0] * n_states
        
        current_max = 0
        current_max_idx = 0
        for k in range(n_states):
            len_k = states[k][1][0]
            if len_k > current_max:
                current_max = len_k
                current_max_idx = k
            pml[k] = current_max
            pmsi[k] = current_max_idx
            
        prefix_max_len[j] = pml
        prefix_max_state_idx[j] = pmsi
        
        # Update global max
        # The max length for j is pml[-1]
        if pml[-1] > global_max_len:
            global_max_len = pml[-1]
            global_max_idx = j
            # Find the gap that achieved this max
            # It's the gap at index pmsi[-1]
            global_max_gap = sorted_gaps[j][pmsi[-1]]
            
    # Backtrack to find the sequence
    if global_max_idx == -1:
        return [primes[0]]
        
    seq = []
    curr_idx = global_max_idx
    curr_gap = global_max_gap
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        # Get state info
        state = L[curr_idx][curr_gap]
        prev_idx = state[1]
        prev_gap = state[2]
        curr_idx = prev_idx
        curr_gap = prev_gap
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
