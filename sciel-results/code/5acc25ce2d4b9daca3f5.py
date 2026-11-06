import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Beam Search with deduplication by sequence
    # State: seq -> (prefix_vecs, prod_set)
    # Key insight: deduplicate by seq to avoid redundant states
    
    beam = {(): ([zero_vec], set())}
    beam_width = 200  # Increased for better search
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = {}
        v_n = vecs[n]
        
        for seq, (pvecs, pset) in beam.items():
            # Option 1: Exclude n - keep state as is
            if seq not in new_beam:
                new_beam[seq] = (pvecs, pset)
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            v_new = tuple(last_pvec[i] + v_n[i] for i in range(num_primes))
            
            valid = True
            new_prods = []
            
            for i in range(len(pvecs)):
                diff = tuple(v_new[j] - pvecs[i][j] for j in range(num_primes))
                if diff in pset:
                    valid = False
                    break
                new_prods.append(diff)
            
            if valid:
                new_seq = seq + (n,)
                new_pvecs = pvecs + [v_new]
                new_pset = pset.union(new_prods)
                new_beam[new_seq] = (new_pvecs, new_pset)
        
        # Select top beam_width states from new_beam
        sorted_items = sorted(new_beam.items(), key=lambda x: len(x[0]), reverse=True)
        beam = dict(sorted_items[:beam_width])
        
        # Early termination
        if beam:
            best_seq = max(beam.keys(), key=len)
            if len(best_seq) == 127:
                return list(best_seq)

    # Return best found
    if beam:
        best_seq = max(beam.keys(), key=len)
        return list(best_seq)
    else:
        return []
