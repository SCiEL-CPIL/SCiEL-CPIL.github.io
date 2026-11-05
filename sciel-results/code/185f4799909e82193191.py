import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    prescribed = [1, 2, 5, 11]
    N = 284
    max_sum = 2 * N
    
    # sum_counts[s] = number of pairs in A that sum to s
    # We will maintain this incrementally.
    
    def init_state(A):
        sum_counts = [0] * (max_sum + 1)
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                sum_counts[s] += 1
        penalty = 0
        for c in sum_counts:
            penalty += c * (c - 1) // 2
        return sum_counts, penalty

    def remove_val(A, val, sum_counts):
        # Returns delta_penalty
        delta = 0
        # Find index of val
        idx = A.index(val)
        # For each other element p in A
        for i, p in enumerate(A):
            if i == idx:
                continue
            s = val + p
            c = sum_counts[s]
            # Removing one instance from count c
            # Penalty change: binom(c, 2) - binom(c-1, 2) = c - 1
            delta -= (c - 1)
            sum_counts[s] -= 1
        # Handle 2*val
        s = 2 * val
        c = sum_counts[s]
        delta -= (c - 1)
        sum_counts[s] -= 1
        
        # Remove val from A
        A.pop(idx)
        return delta

    def add_val(A, val, sum_counts):
        # Returns delta_penalty
        delta = 0
        # For each existing element p in A
        for p in A:
            s = val + p
            c = sum_counts[s]
            # Adding one instance to count c
            # Penalty change: binom(c+1, 2) - binom(c, 2) = c
            delta += c
            sum_counts[s] += 1
        # Handle 2*val
        s = 2 * val
        c = sum_counts[s]
        delta += c
        sum_counts[s] += 1
        
        A.append(val)
        return delta

    # Main search
    best_result = None
    
    # Try sizes 20 down to 10
    for K in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 10:
            break
            
        num_vars = K - len(prescribed)
        if num_vars < 0:
            continue
            
        V = list(range(1, N + 1))
        for p in prescribed:
            V.remove(p)
        
        # Initialize
        current_X = random.sample(V, num_vars)
        current_A = prescribed.copy()
        current_A.extend(current_X)
        current_A.sort()
        
        sum_counts, current_penalty = init_state(current_A)
        
        best_A = current_A.copy()
        best_penalty = current_penalty
        
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        steps = 0
        max_steps = 200000 # Adjust
        
        while T > T_min and time.time() < deadline - 5:
            # Generate Move
            # 1. Replace 1 element
            # 2. Replace 2 elements
            
            move_type = random.random()
            old_vals = []
            new_vals = []
            
            if num_vars >= 2 and move_type < 0.3:
                # Replace 2
                idxs = random.sample(range(num_vars), 2)
                # Map indices to values? current_X is a list.
                # But current_A is sorted. It's easier to work with sets for X.
                # Let's stick to current_X list for move generation, but current_A for penalty.
                # Wait, current_A contains prescribed too.
                # We need to identify which elements in current_A are "variable".
                # Let's maintain current_X as a set of values.
                
                pass # This logic is getting tangled. Let's use sets for X.
            
            # Reset approach for move generation with sets
            current_X_set = set(current_X)
            
            if num_vars >= 2 and random.random() < 0.3:
                old_vals = list(random.sample(list(current_X_set), 2))
                available = [v for v in V if v not in current_X_set]
                if len(available) < 2:
                    continue
                new_vals = random.sample(available, 2)
            else:
                old_vals = [random.choice(list(current_X_set))]
                available = [v for v in V if v not in current_X_set]
                if not available:
                    continue
                new_vals = [random.choice(available)]
            
            # Apply Move
            # 1. Remove old_vals
            delta = 0
            temp_A = current_A.copy()
            temp_sum_counts = sum_counts.copy() # This is expensive! 570 list copy.
            # Better to update in place and rollback if rejected?
            # Or just calculate new penalty from scratch? No, too slow.
            # Let's update in place, keep track of changes, and rollback if needed.
            
            # This is getting complicated.
            # Alternative: Since K is small, just recalculate penalty from scratch?
            # 210 sums. 200k steps. 42 million sum ops.
            # Python can do ~10-50M simple ops/sec.
            # 42M might take 1-4 seconds.
            # If we do 20 sizes, it might be too slow.
            # But we stop early if we find 0.
            # And we can reduce max_steps for larger K if time is tight.
            
            # Let's try the "recalculate from scratch" approach first for simplicity,
            # but optimize the sum calculation.
            
            new_X_set = (current_X_set - set(old_vals)) | set(new_vals)
            new_A = prescribed.copy()
            new_A.extend(list(new_X_set))
            
            # Calculate penalty
            sums = {}
            for i in range(len(new_A)):
                ai = new_A[i]
                for j in range(i, len(new_A)):
                    s = ai + new_A[j]
                    sums[s] = sums.get(s, 0) + 1
            
            new_penalty = 0
            for c in sums.values():
                new_penalty += c * (c - 1) // 2
            
            delta = new_penalty - current_penalty
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_X_set = new_X_set
                current_A = new_A
                current_penalty = new_penalty
                
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_A = current_A.copy()
                    
                if best_penalty == 0:
                    return best_A
            
            T *= cooling
            steps += 1
            
            if steps % 1000 == 0 and time.time() > deadline - 20:
                break
                
        if best_penalty == 0:
            return best_A

    # Fallback Greedy
    A = prescribed.copy()
    sums = set()
    for i, x in enumerate(A):
        for y in A[i:]:
            sums.add(x + y)
    for x in range(12, 285):
        if time.time() > deadline:
            break
        new = {x + y for y in A} | {2 * x}
        if len(new) == len(A) + 1 and not new & sums:
            A.append(x)
            sums |= new
    return A
