import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        for a in S:
            for b in S:
                if a >= b:
                    continue
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def greedy_construct():
        """Greedy: start with a base set, then add elements that maximize new 3-AP differences."""
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
            for v in range(1, 513):
                if v in S_set:
                    continue
                temp_set = S_set | {v}
                new_valid = set()
                for a in base:
                    for b in base:
                        if a < b:
                            d = b - a
                            c = 2*b - a
                            if c in temp_set:
                                new_valid.add(d)
                    d1 = v - a
                    if d1 > 0:
                        c1 = 2*v - a
                        if c1 in temp_set:
                            new_valid.add(d1)
                    d2 = a - v
                    if d2 > 0:
                        c2 = 2*a - v
                        if c2 in temp_set:
                            new_valid.add(d2)
                added = len(new_valid - current_diffs)
                if added > best_new_diffs:
                    best_new_diffs = added
                    best_candidate = v
            if best_candidate is None:
                break
            base.append(best_candidate)
            S_set.add(best_candidate)
            current_diffs = set()
            for a in base:
                for b in base:
                    if a < b:
                        d = b - a
                        c = 2*b - a
                        if c in S_set:
                            current_diffs.add(d)
        return base
    
    def get_targeted_moves(S, S_set, current_diffs, max_moves=50):
        """Generate moves that are likely to create new 3-APs."""
        moves = []
        for i in range(32):
            old_val = S[i]
            # For each existing pair (a,b) in S\{S[i]}, find if adding some v completes a 3-AP
            for a in S:
                if a == old_val:
                    continue
                for b in S:
                    if b == old_val or b == a:
                        continue
                    # Case 1: a, b, c where c = 2b-a, need to add c
                    c = 2*b - a
                    if 1 <= c <= 512 and c not in S_set:
                        moves.append((i, c))
                    # Case 2: a, c, b where c = (a+b)/2, need to add c
                    if (a + b) % 2 == 0:
                        c = (a + b) // 2
                        if 1 <= c <= 512 and c not in S_set:
                            moves.append((i, c))
                    # Case 3: c, a, b where c = 2a-b, need to add c
                    c = 2*a - b
                    if 1 <= c <= 512 and c not in S_set:
                        moves.append((i, c))
            # Also try values near existing elements
            for delta in [-3, -2, -1, 1, 2, 3, 4, 5, 7, 10, 15, 20]:
                v = old_val + delta
                if 1 <= v <= 512 and v not in S_set:
                    moves.append((i, v))
        random.shuffle(moves)
        return moves[:max_moves]
    
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
    
    # Arithmetic progressions with various common differences
    for d in [2, 3, 4, 5, 6, 8, 10, 12]:
        s = []
        start = 1
        while len(s) < 32 and start + 31*d <= 512:
            s.append(start + len(s)*d)
        if len(s) == 32:
            candidates.append(s)
    
    # Mixed: dense cluster + sparse spread
    s = list(range(1, 17))
    for i in range(16, 32):
        s.append(17 + (i-16)*15)
    if len(s) == 32 and max(s) <= 512:
        candidates.append(s)
    
    # Random sets
    for _ in range(8):
        candidates.append(random.sample(range(1, 513), 32))
    
    # Find best initial
    best_S = max(candidates, key=evaluate)
    best_score = evaluate(best_S)
    
    # Simulated Annealing with better moves
    T = 5.0
    cooling = 0.9995
    current_S = best_S[:]
    current_score = best_score
    S_set = set(current_S)
    current_diffs = set()
    for a in current_S:
        for b in current_S:
            if a < b:
                d = b - a
                c = 2*b - a
                if c in S_set:
                    current_diffs.add(d)
    
    # Multi-start: run several rounds with different initial solutions
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
        
        if restart == 0:
            current_S = best_S[:]
            current_score = best_score
        else:
            # Use a different initial solution
            init_idx = random.randint(0, len(candidates) - 1)
            current_S = candidates[init_idx][:]
            current_score = evaluate(current_S)
        
        S_set = set(current_S)
        current_diffs = set()
        for a in current_S:
            for b in current_S:
                if a < b:
                    d = b - a
                    c = 2*b - a
                    if c in S_set:
                        current_diffs.add(d)
        
        T = 5.0
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() - start_time < timeout and T > 0.05:
            if no_improve_count > max_no_improve:
                break
            
            # Choose move type
            move_type = random.random()
            if move_type < 0.4:
                # Targeted move: replace one element with a value that may create new 3-APs
                targeted = get_targeted_moves(current_S, S_set, current_diffs, max_moves=20)
                if not targeted:
                    continue
                i, new_val = random.choice(targeted)
                old_val = current_S[i]
                if new_val == old_val or new_val in S_set:
                    continue
                S_set.remove(old_val)
                S_set.add(new_val)
                current_S[i] = new_val
                new_score = evaluate(current_S)
                
                if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                    current_score = new_score
                    no_improve_count = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_S = current_S[:]
                else:
                    current_S[i] = old_val
                    S_set.remove(new_val)
                    S_set.add(old_val)
                    no_improve_count += 1
            elif move_type < 0.7:
                # Random replace one element
                i = random.randint(0, 31)
                old_val = current_S[i]
                new_val = random.randint(1, 512)
                if new_val in S_set:
                    continue
                S_set.remove(old_val)
                S_set.add(new_val)
                current_S[i] = new_val
                new_score = evaluate(current_S)
                
                if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                    current_score = new_score
                    no_improve_count = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_S = current_S[:]
                else:
                    current_S[i] = old_val
                    S_set.remove(new_val)
                    S_set.add(old_val)
                    no_improve_count += 1
            elif move_type < 0.85:
                # Swap two elements
                i, j = random.sample(range(32), 2)
                old_i, old_j = current_S[i], current_S[j]
                current_S[i], current_S[j] = old_j, old_i
                new_score = evaluate(current_S)
                
                if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                    current_score = new_score
                    no_improve_count = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_S = current_S[:]
                else:
                    current_S[i], current_S[j] = old_i, old_j
                    no_improve_count += 1
            else:
                # Replace two elements
                i, j = random.sample(range(32), 2)
                old_i, old_j = current_S[i], current_S[j]
                new_i = random.randint(1, 512)
                new_j = random.randint(1, 512)
                if new_i in S_set or new_j in S_set or new_i == new_j:
                    continue
                S_set.remove(old_i)
                S_set.remove(old_j)
                S_set.add(new_i)
                S_set.add(new_j)
                current_S[i] = new_i
                current_S[j] = new_j
                new_score = evaluate(current_S)
                
                if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                    current_score = new_score
                    no_improve_count = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_S = current_S[:]
                else:
                    current_S[i] = old_i
                    current_S[j] = old_j
                    S_set.remove(new_i)
                    S_set.remove(new_j)
                    S_set.add(old_i)
                    S_set.add(old_j)
                    no_improve_count += 1
            
            T *= cooling
    
    # Final polish: local search with only improving moves
    S_set = set(best_S)
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
        improved = False
        for i in range(32):
            old_val = best_S[i]
            # Try targeted moves
            for a in best_S:
                if a == old_val:
                    continue
                for b in best_S:
                    if b == old_val or b == a:
                        continue
                    c = 2*b - a
                    if 1 <= c <= 512 and c not in S_set:
                        S_set.remove(old_val)
                        S_set.add(c)
                        best_S[i] = c
                        new_score = evaluate(best_S)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            best_S[i] = old_val
                            S_set.remove(c)
                            S_set.add(old_val)
                    if (a + b) % 2 == 0:
                        c = (a + b) // 2
                        if 1 <= c <= 512 and c not in S_set:
                            S_set.remove(old_val)
                            S_set.add(c)
                            best_S[i] = c
                            new_score = evaluate(best_S)
                            if new_score > best_score:
                                best_score = new_score
                                improved = True
                                break
                            else:
                                best_S[i] = old_val
                                S_set.remove(c)
                                S_set.add(old_val)
                    c = 2*a - b
                    if 1 <= c <= 512 and c not in S_set:
                        S_set.remove(old_val)
                        S_set.add(c)
                        best_S[i] = c
                        new_score = evaluate(best_S)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            best_S[i] = old_val
                            S_set.remove(c)
                            S_set.add(old_val)
        if not improved:
            break
    
    return best_S
