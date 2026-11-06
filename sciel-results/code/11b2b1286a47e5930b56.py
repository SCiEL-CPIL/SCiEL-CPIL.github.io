import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as numpy arrays for fast arithmetic
    def get_vec(n):
        v = np.zeros(num_primes, dtype=np.int8)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return v
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = np.zeros(num_primes, dtype=np.int8)
    
    # Beam search with optimized state representation
    # State: (seq, prefix_vecs_array, prod_set)
    # prefix_vecs_array: 2D numpy array of shape (k+1, num_primes)
    
    beam = [([], np.array([zero_vec]), set())]
    beam_width = 200
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            # New prefix vector
            v_new = pvecs[-1] + v_n
            
            # Check all suffix products ending at n
            valid = True
            new_prod_ids = []
            
            for i in range(len(pvecs)):
                diff = v_new - pvecs[i]
                diff_key = diff.tobytes()
                
                if diff_key in pset:
                    valid = False
                    break
                new_prod_ids.append(diff_key)
            
            if valid:
                new_seq = seq + [n]
                new_pvecs = np.vstack([pvecs, v_new])
                new_pset = pset.union(new_prod_ids)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states by sequence length
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Early termination if we found a very long sequence
        if beam and len(beam[0][0]) >= 100:
            # Continue but with tighter beam to save time
            beam_width = min(beam_width, 50)
    
    # Get best sequence from beam
    if beam:
        best_seq = beam[0][0]
    else:
        best_seq = []
    
    # Local improvement pass: try to insert missing numbers
    if time.time() < deadline - 5.0:
        best_seq = sorted(best_seq)
        seq_set = set(best_seq)
        candidates = [x for x in range(2, 129) if x not in seq_set]
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for c in candidates:
                if c in seq_set:
                    continue
                
                # Find insertion position
                ins_idx = 0
                for i, x in enumerate(best_seq):
                    if x < c:
                        ins_idx = i + 1
                    else:
                        break
                
                # Try inserting c at ins_idx
                test_seq = best_seq[:ins_idx] + [c] + best_seq[ins_idx:]
                
                # Verify validity
                valid = True
                prefix_vecs = [zero_vec.copy()]
                prod_keys = set()
                
                for x in test_seq:
                    v_x = vecs[x]
                    v_new = prefix_vecs[-1] + v_x
                    prefix_vecs.append(v_new)
                    
                    for i in range(len(prefix_vecs) - 1):
                        diff = v_new - prefix_vecs[i]
                        diff_key = diff.tobytes()
                        if diff_key in prod_keys:
                            valid = False
                            break
                        prod_keys.add(diff_key)
                    
                    if not valid:
                        break
                
                if valid:
                    best_seq = test_seq
                    seq_set.add(c)
                    improved = True
        
        best_seq = sorted(best_seq)
    
    return best_seq

# Run a quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
