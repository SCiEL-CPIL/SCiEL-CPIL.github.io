import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256
    ALL = (1 << N) - 1  # bitmask for [1..N] where bit i represents element i+1
    
    # Precompute: for each element x (1-indexed), the mask of elements that would form a 4-AP with x as a specific position
    # Actually, let's precompute conflicts: for each pair (x, y), check if adding both creates issues
    # Better: precompute for each x, the set of y such that {x,y} is part of some 4-AP
    # Most efficient: for each x, precompute which existing elements would make x invalid
    
    # Let's use a different approach: represent A as a set and use efficient membership checks
    # But for speed in SA, let's use a boolean array
    
    def time_left():
        return deadline - time.time()
    
    # Greedy construction: pick elements in order of 3-AP gain
    def greedy_construct():
        A = set()
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        while remaining and time_left() > 10:
            best_x = -1
            best_gain = -1
            for x in remaining:
                if _is_valid(x, A, N):
                    gain = _gain(x, A, N)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            if best_x == -1:
                break
            A.add(best_x)
            remaining.remove(best_x)
        
        return A
    
    def _is_valid(x, A, N):
        # Check if adding x to A creates a 4-AP
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in A and (x + 2*d) in A and (x + 3*d) in A:
                return False
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in A and (x + d) in A and (x + 2*d) in A:
                return False
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min(x // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in A and (x - d) in A and (x + d) in A:
                return False
        # x as 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in A and (x - 2*d) in A and (x - d) in A:
                return False
        return True
    
    def _gain(x, A, N):
        # Count 3-APs that x would complete
        gain = 0
        # x as 1st: x, x+d, x+2d
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if (x + d) in A and (x + 2*d) in A:
                gain += 1
        # x as 2nd: x-d, x, x+d
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (x - d) in A and (x + d) in A:
                gain += 1
        # x as 3rd: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if (x - 2*d) in A and (x - d) in A:
                gain += 1
        return gain
    
    def count_3aps(A, N):
        count = 0
        for b in A:
            max_d = min(b - 1, N - b)
            for d in range(1, max_d + 1):
                if (b - d) in A and (b + d) in A:
                    count += 1
        return count
    
    # Phase 1: Greedy construction
    A = greedy_construct()
    current_score = count_3aps(A, N)
    
    # Phase 2: Simulated Annealing
    # We'll do add/remove/swap moves
    # Temperature schedule: start high, cool geometrically
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    best_A = set(A)
    best_score = current_score
    
    while T > T_min and time_left() > 5:
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            if _is_valid(x, A, N):
                gain = _gain(x, A, N)
                new_score = current_score + gain
                delta = new_score - current_score
                if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0):
                    A.add(x)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
        
        elif move_type < 0.7:
            # Remove a random element
            if not A:
                T *= cooling
                continue
            x = random.choice(list(A))
            loss = _gain(x, A, N)  # This is the number of 3-APs x participates in
            new_score = current_score - loss
            delta = new_score - current_score
            if delta > 0 or random.random() < min(1.0, abs(delta) / T if T > 0 else 0):
                A.remove(x)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
        
        else:
            # Swap: remove a, add b
            if not A:
                T *= cooling
                continue
            candidates_out = list(A)
            candidates_in = [x for x in range(1, N + 1) if x not in A]
            if not candidates_in:
                T *= cooling
                continue
            a = random.choice(candidates_out)
            b = random.choice(candidates_in)
            
            # Check if b is valid in A - {a}
            A.remove(a)
            if _is_valid(b, A, N):
                loss_a = _gain(a, {x for x in A if x != b}, N)  # approximate
                # More precisely: loss of a from original A
                # Let's recompute properly
                A.add(a)
                loss_a = _gain(a, A, N)
                
                A.remove(a)
                gain_b = _gain(b, A, N)
                A.add(a)
                
                new_score = current_score - loss_a + gain_b
                delta = new_score - current_score
                if delta > 0 or random.random() < min(1.0, abs(delta) / T if T > 0 else 0):
                    A.remove(a)
                    A.add(b)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
            else:
                A.add(a)
        
        T *= cooling
    
    # Phase 3: Final hill-climbing polish
    A = set(best_A)
    current_score = best_score
    
    improved = True
    while improved and time_left() > 2:
        improved = False
        
        # Try adding any valid element
        for x in range(1, N + 1):
            if x not in A and _is_valid(x, A, N):
                gain = _gain(x, A, N)
                if gain > 0:
                    A.add(x)
                    current_score += gain
                    improved = True
        
        # Try removing elements that don't contribute
        for x in list(A):
            loss = _gain(x, A, N)
            if loss == 0:
                A.remove(x)
                improved = True
        
        # Try swaps
        for a in list(A):
            for b in range(1, N + 1):
                if b in A or b == a:
                    continue
                A.remove(a)
                if _is_valid(b, A, N):
                    gain_b = _gain(b, A, N)
                    A.add(a)
                    loss_a = _gain(a, A, N)
                    if gain_b > loss_a:
                        A.remove(a)
                        A.add(b)
                        current_score += gain_b - loss_a
                        improved = True
                        break
                else:
                    A.add(a)
    
    # Return the better of the two
    if current_score > best_score:
        best_A = set(A)
        best_score = current_score
    
    return sorted(best_A)
