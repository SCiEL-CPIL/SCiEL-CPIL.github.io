import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def greedy_construct():
        """Greedy: start with a base set, then add elements that maximize new 3-AP differences."""
        # Start with a small structured set
        base = [1, 2, 3]
        S_set = set(base)
        current_diffs = set()
        for a in base:
            for b in base:
                if a < b:
                    d = b - a
                    c = 2*b - a
                    if c in S_set:
                        current_diffs.add(d)
        
        while len(base) < 32:
            best_candidate = None
            best_new_diffs = -1
            # Try all available values
            for v in range(1, 513):
                if v in S_set:
                    continue
                # Check how many new differences this adds
                new_diffs = set()
                for a in base:
                    d1 = v - a
                    if d1 > 0:
                        c1 = v + d1  # a + 2*d1 = 2*v - a
                        if c1 in S_set or c1 == v:  # c1 can't be v since d1>0
                            if c1 in S_set:
                                new_diffs.add(d1)
                    d2 = a - v
                    if d2 > 0:
                        c2 = a + d2  # v + 2*d2 = 2*a - v
                        if c2 in S_set:
                            new_diffs.add(d2)
                    # Also check if v is the middle: a, v, a+2*(v-a)
                    # or v is the end: a, a+d, v where d = v-a, need a+2d = a+2(v-a) = 2v-a in S
                # Actually let me just evaluate properly
                temp_S = base + [v]
                temp_set = S_set | {v}
                new_valid = set()
                for i in range(len(temp_S)):
                    a = temp_S[i]
                    for j in range(i+1, len(temp_S)):
                        b = temp_S[j]
                        d = b - a
                        c = 2*b - a
                        if c in temp_set:
                            new_valid.add(d)
                added = len(new_valid - current_diffs)
                if added > best_new_diffs:
                    best_new_diffs = added
                    best_candidate = v
                    if added > 0:
                        # Don't break early, keep looking for better
                        pass
            if best_candidate is None:
                break
            base.append(best_candidate)
            S_set.add(best_candidate)
            # Update current_diffs
            current_diffs = set()
            for i in range(len(base)):
                a = base[i]
                for j in range(i+1, len(base)):
                    b = base[j]
                    d = b - a
                    c = 2*b - a
                    if c in S_set:
                        current_diffs.add(d)
        return base
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    candidates.append(list(range(2, 34)))
    candidates.append(greedy_construct())
    
    # Structured: blocks of 3 with increasing gaps
    s = []
    for i in range(11):
        base = 1 + i * 3
        for j in range(3):
            val = base + j
            if 1 <= val <= 512 and val not in s:
                s.append(val)
    if len(s) >= 32:
        candidates.append(s[:32])
    
    # Random sets
    for _ in range(5):
        candidates.append(random.sample(range(1, 513), 32))
    
    # Find best initial
    best_S = max(candidates, key=evaluate)
    best_score = evaluate(best_S)
    
    # Simulated Annealing with better moves
    T = 5.0
    cooling = 0.9995
    current_S = best_S[:]
    current_score = best_score
    
    while time.time() - start_time < timeout:
        if T < 0.05:
            break
            
        # Choose move type
        move_type = random.random()
        if move_type < 0.5:
            # Replace one element
            i = random.randint(0, 31)
            old_val = current_S[i]
            new_val = random.randint(1, 512)
            if new_val in current_S:
                continue
            current_S[i] = new_val
            new_score = evaluate(current_S)
            
            if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
            else:
                current_S[i] = old_val
        elif move_type < 0.8:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            current_S[i], current_S[j] = old_j, old_i
            new_score = evaluate(current_S)
            
            if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
            else:
                current_S[i], current_S[j] = old_i, old_j
        else:
            # Replace two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            new_i = random.randint(1, 512)
            new_j = random.randint(1, 512)
            if new_i in current_S or new_j in current_S or new_i == new_j:
                continue
            current_S[i] = new_i
            current_S[j] = new_j
            new_score = evaluate(current_S)
            
            if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
            else:
                current_S[i] = old_i
                current_S[j] = old_j
        
        T *= cooling
        
    return best_S
