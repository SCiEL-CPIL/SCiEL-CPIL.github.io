import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    def calculate_initial_state(S):
        """Calculate valid_diffs and diff_count for an initial set S."""
        S_set = set(S)
        valid_diffs = set()
        diff_count = {}
        for a in S:
            for b in S:
                if b > a:
                    d = b - a
                    c = a + 2 * d
                    if c in S_set:
                        if d not in diff_count:
                            diff_count[d] = 0
                            valid_diffs.add(d)
                        diff_count[d] += 1
        return valid_diffs, diff_count

    def add_element(x, S_set, valid_diffs, diff_count):
        """Add x to S_set and update valid_diffs, diff_count. Returns None."""
        # x is end: a, m, x
        for a in S_set:
            if a < x and (x - a) % 2 == 0:
                d = (x - a) // 2
                m = a + d
                if m in S_set and m != x: # m must be in old set, x is new so m!=x is true if m in S_set
                    if d not in diff_count:
                        diff_count[d] = 0
                        valid_diffs.add(d)
                    diff_count[d] += 1
        
        # x is middle: a, x, c
        for a in S_set:
            if a < x:
                d = x - a
                c = x + d
                if c in S_set and c != x:
                    if d not in diff_count:
                        diff_count[d] = 0
                        valid_diffs.add(d)
                    diff_count[d] += 1
        
        # x is start: x, b, c
        for b in S_set:
            if b > x:
                d = b - x
                c = x + 2 * d
                if c in S_set and c != x:
                    if d not in diff_count:
                        diff_count[d] = 0
                        valid_diffs.add(d)
                    diff_count[d] += 1
        
        S_set.add(x)

    def remove_element(x, S_set, valid_diffs, diff_count):
        """Remove x from S_set and update valid_diffs, diff_count. Returns None."""
        # x was end: a, m, x
        for a in S_set:
            if a < x and (x - a) % 2 == 0:
                d = (x - a) // 2
                m = a + d
                if m in S_set and m != x:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        valid_diffs.discard(d)
                        # del diff_count[d] # optional cleanup
        
        # x was middle: a, x, c
        for a in S_set:
            if a < x:
                d = x - a
                c = x + d
                if c in S_set and c != x:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        valid_diffs.discard(d)
        
        # x was start: x, b, c
        for b in S_set:
            if b > x:
                d = b - x
                c = x + 2 * d
                if c in S_set and c != x:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        valid_diffs.discard(d)
        
        S_set.discard(x)

    def make_initial_structured():
        """Create a structured initial set likely to have many 3-AP differences."""
        candidates = []
        
        # Option 1: Spread evenly
        s1 = [1 + i * 16 for i in range(32)]
        candidates.append(s1)
        
        # Option 2: Two interleaved APs
        s2 = []
        for i in range(16):
            s2.append(1 + i * 32)
            s2.append(17 + i * 32)
        candidates.append(s2)
        
        # Option 3: Use a mix of small and large gaps
        s3 = list(range(1, 17)) + [100 + i * 10 for i in range(16)]
        s3 = [x for x in s3 if 1 <= x <= 512][:32]
        if len(s3) == 32:
            candidates.append(s3)
        
        # Option 4: Random spread
        s4 = sorted(random.sample(range(1, 513), 32))
        candidates.append(s4)
        
        # Option 5: Clustered with some spread
        s5 = []
        base_positions = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        for bp in base_positions:
            s5.append(bp)
            if bp + 5 <= 512:
                s5.append(bp + 5)
        s5 = s5[:32]
        if len(s5) == 32:
            candidates.append(s5)
        
        # Pick the best
        best = max(candidates, key=lambda s: len(calculate_initial_state(s)[0]))
        return best

    # Generate multiple initial candidates
    initial_candidates = []
    for _ in range(3):
        initial_candidates.append(make_initial_structured())
    
    # Also try a greedy construction
    def greedy_construct():
        """Greedy: add elements that maximize new distinct differences."""
        S = []
        S_set = set()
        valid_diffs, diff_count = calculate_initial_state(S)
        
        # Start with a small seed
        seed = [1, 2, 3, 100, 200, 300]
        for x in seed:
            if x not in S_set:
                S.append(x)
                add_element(x, S_set, valid_diffs, diff_count)
        
        while len(S) < 32:
            best_val = None
            best_gain = -1
            # Sample a subset of candidates for speed
            candidates = random.sample(range(1, 513), min(200, 512 - len(S_set)))
            for v in candidates:
                if v in S_set:
                    continue
                
                # Simulate adding v
                temp_valid = set(valid_diffs)
                temp_count = dict(diff_count)
                temp_set = set(S_set)
                add_element(v, temp_set, temp_valid, temp_count)
                
                gain = len(temp_valid) - len(valid_diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_val = v
            
            if best_val is None:
                while True:
                    v = random.randint(1, 512)
                    if v not in S_set:
                        break
                best_val = v
            
            S.append(best_val)
            add_element(best_val, S_set, valid_diffs, diff_count)
        
        return S

    initial_candidates.append(greedy_construct())
    
    # Find best initial
    best_S = max(initial_candidates, key=lambda s: len(calculate_initial_state(s)[0]))
    best_score = len(calculate_initial_state(best_S)[0])
    
    # Initialize state for SA
    current_S = best_S[:]
    current_set = set(current_S)
    current_valid_diffs, current_diff_count = calculate_initial_state(current_S)
    current_score = len(current_valid_diffs)
    
    T = 10.0
    cooling = 0.999
    no_improve = 0
    
    while time.time() - start_time < timeout:
        if T < 0.1:
            break
            
        move_type = random.random()
        
        if move_type < 0.6:
            # Replace one element
            i = random.randint(0, 31)
            old_val = current_S[i]
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
            
            # Apply move
            remove_element(old_val, current_set, current_valid_diffs, current_diff_count)
            add_element(new_val, current_set, current_valid_diffs, current_diff_count)
            current_S[i] = new_val
            
            new_score = len(current_valid_diffs)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Undo
                remove_element(new_val, current_set, current_valid_diffs, current_diff_count)
                add_element(old_val, current_set, current_valid_diffs, current_diff_count)
                current_S[i] = old_val
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            
            # Apply move
            remove_element(old_i, current_set, current_valid_diffs, current_diff_count)
            remove_element(old_j, current_set, current_valid_diffs, current_diff_count)
            add_element(old_j, current_set, current_valid_diffs, current_diff_count)
            add_element(old_i, current_set, current_valid_diffs, current_diff_count)
            
            current_S[i], current_S[j] = old_j, old_i
            
            new_score = len(current_valid_diffs)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Undo
                remove_element(old_j, current_set, current_valid_diffs, current_diff_count)
                remove_element(old_i, current_set, current_valid_diffs, current_diff_count)
                add_element(old_i, current_set, current_valid_diffs, current_diff_count)
                add_element(old_j, current_set, current_valid_diffs, current_diff_count)
                current_S[i], current_S[j] = old_i, old_j
                no_improve += 1
                
        else:
            # Replace two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            new_i = random.randint(1, 512)
            new_j = random.randint(1, 512)
            if new_i in current_set or new_j in current_set or new_i == new_j:
                continue
            
            # Apply move
            remove_element(old_i, current_set, current_valid_diffs, current_diff_count)
            remove_element(old_j, current_set, current_valid_diffs, current_diff_count)
            add_element(new_i, current_set, current_valid_diffs, current_diff_count)
            add_element(new_j, current_set, current_valid_diffs, current_diff_count)
            
            current_S[i] = new_i
            current_S[j] = new_j
            
            new_score = len(current_valid_diffs)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Undo
                remove_element(new_i, current_set, current_valid_diffs, current_diff_count)
                remove_element(new_j, current_set, current_valid_diffs, current_diff_count)
                add_element(old_i, current_set, current_valid_diffs, current_diff_count)
                add_element(old_j, current_set, current_valid_diffs, current_diff_count)
                current_S[i] = old_i
                current_S[j] = old_j
                no_improve += 1
        
        T *= cooling
        
        # Restart if stuck
        if no_improve > 5000:
            # Reset to best and restart with higher temperature
            current_S = best_S[:]
            current_set = set(current_S)
            current_valid_diffs, current_diff_count = calculate_initial_state(current_S)
            current_score = len(current_valid_diffs)
            T = 5.0
            no_improve = 0

    return best_S
