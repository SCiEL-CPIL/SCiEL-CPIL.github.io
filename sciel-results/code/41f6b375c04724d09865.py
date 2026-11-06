import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_seq = []
    best_len = 0
    
    def is_valid_step(prefixes, diffs, last_a, new_a):
        if new_a <= last_a:
            return False
        if new_a > 256:
            return False
        new_p = prefixes[-1] + new_a
        for p in prefixes:
            d = new_p - p
            if d in diffs:
                return False
        return True

    def add_step(prefixes, diffs, a):
        new_p = prefixes[-1] + a
        new_prefixes = prefixes + [new_p]
        new_diffs = set(diffs)
        for p in prefixes:
            new_diffs.add(new_p - p)
        return new_prefixes, new_diffs

    def greedy(mode):
        seq = []
        prefixes = [0]
        diffs = set()
        last_a = 0
        while True:
            found = False
            if mode == 'smallest':
                candidates = range(last_a + 1, 257)
            else:
                candidates = range(256, last_a, -1)
                
            for a in candidates:
                if is_valid_step(prefixes, diffs, last_a, a):
                    prefixes, diffs = add_step(prefixes, diffs, a)
                    seq.append(a)
                    last_a = a
                    found = True
                    break
            if not found:
                break
        return seq

    # 1. Greedy Baselines
    s1 = greedy('smallest')
    if len(s1) > best_len:
        best_len = len(s1)
        best_seq = s1
        
    s2 = greedy('largest')
    if len(s2) > best_len:
        best_len = len(s2)
        best_seq = s2

    # 2. Beam Search
    def beam_search(beam_width=50, max_depth=100):
        # State: (seq, prefixes, diffs, last_a, total_sum)
        # We use total_sum for sorting
        
        states = [([], [0], set(), 0, 0)]
        
        current_best = 0
        current_best_seq = []
        
        for depth in range(max_depth):
            if time.time() > deadline:
                break
                
            next_states = []
            for state in states:
                seq, prefixes, diffs, last_a, total_sum = state
                
                # Find all valid next steps
                valid_steps = []
                for a in range(last_a + 1, 257):
                    if is_valid_step(prefixes, diffs, last_a, a):
                        valid_steps.append(a)
                        
                if not valid_steps:
                    continue
                    
                # Select top K candidates
                # Heuristic: Smallest a
                selected = valid_steps[:beam_width] 
                
                for a in selected:
                    new_prefixes, new_diffs = add_step(prefixes, diffs, a)
                    new_seq = seq + [a]
                    new_last_a = a
                    new_total_sum = total_sum + a
                    next_states.append((new_seq, new_prefixes, new_diffs, new_last_a, new_total_sum))
                    
                    if len(new_seq) > current_best:
                        current_best = len(new_seq)
                        current_best_seq = new_seq
                        
            if not next_states:
                break
                
            # Prune next_states to keep only the best 'beam_width'
            # Sort by total_sum (smaller is better)
            next_states.sort(key=lambda x: x[4])
            states = next_states[:beam_width]
            
        return current_best_seq

    # Run multiple beam searches with different parameters
    for bw in [10, 20, 50]:
        bs_seq = beam_search(beam_width=bw, max_depth=100)
        if len(bs_seq) > best_len:
            best_len = len(bs_seq)
            best_seq = bs_seq

    # 3. Randomized Restarts
    while time.time() < deadline:
        seq = []
        prefixes = [0]
        diffs = set()
        last_a = 0
        while True:
            valid_steps = []
            for a in range(last_a + 1, 257):
                if is_valid_step(prefixes, diffs, last_a, a):
                    valid_steps.append(a)
            if not valid_steps:
                break
            # Pick randomly, maybe with bias towards smaller
            if valid_steps:
                # Weighted random? Or just uniform?
                # Uniform is fine for diversity
                a = random.choice(valid_steps)
                prefixes, diffs = add_step(prefixes, diffs, a)
                seq.append(a)
                last_a = a
                
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    return best_seq
