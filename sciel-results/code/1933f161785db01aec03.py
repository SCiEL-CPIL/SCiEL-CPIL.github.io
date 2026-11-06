import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    vecs = {}
    for n in range(2, 129):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        vecs[n] = tuple(v)
    zero_vec = (0,) * num_primes

    # Beam search
    # State: (length, seq_tuple, prefix_list_tuple, diff_set_frozenset)
    # Using tuples for immutability and easier hashing/comparison
    
    beam = [(0, (), (zero_vec,), frozenset())]
    beam_width = 150
    
    best_seq = []
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            length, seq, pref_list, diff_set = state
            last_pref = pref_list[-1]
            
            # Option 1: Exclude n
            new_beam.append((length, seq, pref_list, diff_set))
            
            # Option 2: Include n
            new_pref = tuple(last_pref[i] + v_n[i] for i in range(num_primes))
            
            valid = True
            new_diffs = []
            
            # Check all previous prefixes for collision
            for p in pref_list:
                diff = tuple(new_pref[i] - p[i] for i in range(num_primes))
                if diff in diff_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                new_pref_list = pref_list + (new_pref,)
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + (n,)
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_pref_list, new_diff_set))
        
        # Score function: prioritize length, then flexibility
        def calc_score(state):
            l, seq, pl, ds = state
            # Potential: count how many of the next 5 numbers are compatible
            potential = 0
            last_p = pl[-1]
            for m in range(n+1, min(n+6, 129)):
                v_m = vecs[m]
                new_p = tuple(last_p[i] + v_m[i] for i in range(num_primes))
                ok = True
                for p in pl:
                    d = tuple(new_p[i] - p[i] for i in range(num_primes))
                    if d in ds:
                        ok = False
                        break
                if ok:
                    potential += 1
            return (l, potential)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:beam_width]
        
        # Track best
        if beam and beam[0][0] > len(best_seq):
            best_seq = list(beam[0][1])
        
        # Early stop if we've used all numbers
        if beam and beam[0][0] == 127:
            return list(beam[0][1])

    return best_seq
