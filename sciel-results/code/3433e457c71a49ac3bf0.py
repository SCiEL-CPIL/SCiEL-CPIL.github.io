import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)

    # Precompute factor vectors using numpy for speed
    def get_vec(n):
        v = np.zeros(num_primes, dtype=np.int32)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return v

    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = np.zeros(num_primes, dtype=np.int32)

    # Beam State: (length, seq, prefix_array, diff_set)
    # Use numpy arrays for prefixes to speed up vector operations
    
    beam = [(0, [], np.array([zero_vec]), frozenset())]

    # Dynamic beam width: wider early for diversity, narrower later
    def get_beam_width(step, total_steps):
        if step < total_steps * 0.15:
            return 300
        elif step < total_steps * 0.4:
            return 200
        elif step < total_steps * 0.7:
            return 120
        else:
            return 80

    total_steps = 127  # numbers 2 to 128

    for step, n in enumerate(range(2, 129)):
        if time.time() > deadline:
            break

        current_beam_width = get_beam_width(step, total_steps)
        v_n = vecs[n]

        new_beam = []

        for state in beam:
            length, seq, pref_array, diff_set = state
            last_pref = pref_array[-1]

            # Option 1: Exclude n
            new_beam.append((length, seq, pref_array, diff_set))

            # Option 2: Include n
            new_pref = last_pref + v_n

            # Check all previous prefixes for collision using vectorized ops
            valid = True
            new_diffs = []

            # Vectorized difference computation
            diffs = new_pref - pref_array  # shape: (len(pref_array), num_primes)
            
            for i in range(len(pref_array)):
                diff_tup = tuple(diffs[i])
                if diff_tup in diff_set:
                    valid = False
                    break
                new_diffs.append(diff_tup)

            if valid:
                new_pref_array = np.vstack([pref_array, new_pref])
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + [n]
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_pref_array, new_diff_set))

        # Score function: prioritize length, then flexibility
        def calc_score(state):
            l, seq, pa, ds = state
            last_p = pa[-1]

            # Quick lookahead: check compatibility with next few numbers
            potential = 0.0
            for m in range(n + 1, min(n + 8, 129)):
                v_m = vecs[m]
                new_p = last_p + v_m
                diffs = new_p - pa
                ok = True
                for i in range(len(pa)):
                    if tuple(diffs[i]) in ds:
                        ok = False
                        break
                if ok:
                    potential += 1.0 / (m - n)

            # Crowd penalty: large diff_set relative to length
            crowd_penalty = 0.0
            if l > 0 and len(ds) > l * l * 2.0:
                crowd_penalty = (len(ds) - l * l * 2.0) * 0.05

            return (l, potential - crowd_penalty)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:current_beam_width]

        # Early stop if we've used all numbers
        if beam and beam[0][0] == 127:
            return beam[0][1]

    if beam:
        return beam[0][1]
    return []
