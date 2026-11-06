import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def greedy_construct(skip_prob=0.0):
        """
        Greedy construction with optional random skipping.
        Returns a valid sequence.
        """
        values = []
        products = set()
        for x in range(2, 129):
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            
            # New products when adding x: x, x*last, x*last*second_last, ...
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            
            # Check if any new product conflicts with existing ones
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    def compute_products(seq):
        """Compute all consecutive block products for a sequence."""
        n = len(seq)
        if n == 0:
            return []
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods

    def count_conflicts(seq):
        """Count number of duplicate product pairs."""
        if len(seq) < 2:
            return 0
        prods = compute_products(seq)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """
        Score function:
        If valid (0 conflicts): score = length
        Else: score = length - penalty * conflicts
        We use a high penalty to strongly prefer valid solutions.
        """
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 100 * conf

    # 1. Generate initial solutions
    # Standard greedy
    best_seq = greedy_construct(0.0)
    best_len = len(best_seq)
    
    # Try a few randomized greedy starts to find a better baseline
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        candidate = greedy_construct(0.1)
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_seq = candidate

    # 2. Simulated Annealing / Local Search
    # We maintain a current state and a best valid state.
    # Moves: Add a random missing number, or Remove a random existing number.
    # We accept worse states with probability exp(-delta/T).
    
    current_seq = best_seq[:]
    current_score = get_score(current_seq)
    
    T = 10.0
    cool_rate = 0.999
    min_T = 0.1
    
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        # Generate neighbor
        r = random.random()
        
        if r < 0.7:
            # Try to add an element (bias towards adding to increase length)
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            # Insert in sorted position
            # Using bisect is faster
            import bisect
            pos = bisect.bisect_left(current_seq, x)
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        else:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        
        new_score = get_score(new_seq)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_seq = new_seq
            current_score = new_score
            
            # Check if this is a new best valid solution
            # We only update best if it's valid. 
            # count_conflicts is O(N^2). 
            # We can optimize: if new_score == len(new_seq), it's valid.
            if new_score == len(new_seq):
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = current_seq[:]
                    no_improve_count = 0
                    # Boost temperature slightly to explore around new best
                    T = max(T, 5.0)
            else:
                no_improve_count += 1
        
        # Cooling
        T *= cool_rate
        if T < min_T:
            T = min_T
        
        # Restart logic: if stuck for a long time, perturb or restart
        if no_improve_count > 2000:
            no_improve_count = 0
            T = max(T, 5.0)
            # Perturb current state
            for _ in range(5):
                if current_seq:
                    idx = random.randint(0, len(current_seq) - 1)
                    current_seq = current_seq[:idx] + current_seq[idx+1:]
                else:
                    break
                in_seq = set(current_seq)
                available = [x for x in range(2, 129) if x not in in_seq]
                if available:
                    x = random.choice(available)
                    import bisect
                    pos = bisect.bisect_left(current_seq, x)
                    current_seq = current_seq[:pos] + [x] + current_seq[pos:]
            current_score = get_score(current_seq)

    # 3. Final Polish
    # Try to add any missing element that doesn't create conflicts
    # This is a deterministic hill-climbing step on the best valid solution found.
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(2, 129):
            if time.time() > deadline - 1:
                return best_seq
            if x not in in_best:
                import bisect
                pos = bisect.bisect_left(best_seq, x)
                test_seq = best_seq[:pos] + [x] + best_seq[pos:]
                if count_conflicts(test_seq) == 0:
                    best_seq = test_seq
                    in_best.add(x)
                    changed = True
                    break # Restart loop since indices changed

    return best_seq
