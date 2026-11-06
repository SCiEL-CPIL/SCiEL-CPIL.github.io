import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            # New products when adding x: x, x*last, x*last*second_last, ...
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values
    
    # Get initial solution
    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)
    
    # Precompute all possible products for fast lookup
    # We'll use a different approach: track products by their position
    
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
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 50 * conf
    
    # Simulated annealing
    current_seq = initial_seq[:]
    current_score = get_score(current_seq)
    
    T = 5.0
    cool_rate = 0.9995
    best_T = T
    no_improve = 0
    
    while time.time() < deadline - 3:
        # Generate a neighbor
        r = random.random()
        
        if r < 0.6:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            # Insert in sorted position
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        else:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        
        new_score = get_score(new_seq)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T) for worse
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_seq = new_seq
            current_score = new_score
            
            # Check if this is a new best valid solution
            if count_conflicts(current_seq) == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    # Reset temperature occasionally when finding new best
                    T = max(T, 2.0)
            else:
                no_improve += 1
        
        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1
        
        # Restart if stuck
        if no_improve > 500:
            no_improve = 0
            T = best_T * 0.5
            # Perturb current solution
            if current_seq:
                for _ in range(min(3, len(current_seq))):
                    if random.random() < 0.5:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq = current_seq[:idx] + current_seq[idx+1:]
                    else:
                        in_seq = set(current_seq)
                        available = [x for x in range(2, 129) if x not in in_seq]
                        if available:
                            x = random.choice(available)
                            pos = 0
                            for i, v in enumerate(current_seq):
                                if v < x:
                                    pos = i + 1
                                else:
                                    break
                            current_seq = current_seq[:pos] + [x] + current_seq[pos:]
                current_score = get_score(current_seq)
    
    # Final polish: try to add any missing elements that don't create conflicts
    in_best = set(best_seq)
    for x in range(2, 129):
        if x not in in_best:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            test_seq = best_seq[:pos] + [x] + best_seq[pos:]
            if count_conflicts(test_seq) == 0:
                best_seq = test_seq
                in_best.add(x)
    
    return best_seq
