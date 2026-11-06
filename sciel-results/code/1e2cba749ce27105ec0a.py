import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def compute_all_products(seq):
        """Compute set of all consecutive block products."""
        n = len(seq)
        prods = set()
        if n == 0:
            return prods
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def count_conflicts_from_set(prods, n):
        """
        Given a set of products, we can't directly get count of duplicates 
        unless we have the list. 
        For scoring, we need the number of duplicate pairs.
        It's easier to just use a list for conflict counting if we recompute, 
        but for incremental checks we use the set.
        
        Let's stick to: 
        Score = len(seq) if valid (len(set) == total_blocks) else len(seq) - penalty * (total_blocks - len(set))
        """
        total_blocks = n * (n + 1) // 2
        if len(prods) == total_blocks:
            return 0
        return total_blocks - len(prods)

    def get_score(seq, prods_set=None):
        """
        Score: length if valid, else length - penalty * conflicts.
        If prods_set is not provided, it computes it.
        """
        n = len(seq)
        if n == 0:
            return 0
        if prods_set is None:
            prods_set = compute_all_products(seq)
        
        total_blocks = n * (n + 1) // 2
        conflicts = total_blocks - len(prods_set)
        
        if conflicts == 0:
            return n
        return n - 100 * conflicts

    def greedy_construct(randomize=False, skip_prob=0.0):
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            if randomize and random.random() < skip_prob:
                continue
            # New products when adding x to the end (since we iterate in increasing order)
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            
            # Check intersection
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values, products

    # Multi-start initialization
    best_seq = []
    best_len = 0
    best_prods = set()
    best_score = -1
    
    # Run several randomized greedy passes
    for i in range(30):
        seq, prods = greedy_construct(randomize=True, skip_prob=0.05)
        score = get_score(seq, prods)
        if score > best_score:
            best_score = score
            best_len = len(seq)
            best_seq = seq[:]
            best_prods = prods.copy()
            
    # Also try deterministic
    seq_det, prods_det = greedy_construct(randomize=False)
    score_det = get_score(seq_det, prods_det)
    if score_det > best_score:
        best_score = score_det
        best_len = len(seq_det)
        best_seq = seq_det[:]
        best_prods = prods_det.copy()

    # Simulated Annealing
    current_seq = best_seq[:]
    current_prods = best_prods.copy()
    current_score = best_score
    
    T = 10.0
    cool_rate = 0.9995
    best_T = T
    no_improve = 0
    
    all_vals = list(range(2, 129))
    
    while time.time() < deadline - 3:
        r = random.random()
        
        # Decide move type based on validity
        n_curr = len(current_seq)
        total_blocks_curr = n_curr * (n_curr + 1) // 2
        is_valid = (len(current_prods) == total_blocks_curr)
        
        if is_valid:
            # Try to grow
            if r < 0.7:
                move_type = 'add'
            else:
                move_type = 'swap'
        else:
            # Try to fix
            if r < 0.6:
                move_type = 'remove'
            else:
                move_type = 'swap'
        
        new_seq = None
        new_prods = None
        new_score = -10000
        
        if move_type == 'add':
            in_seq = set(current_seq)
            available = [x for x in all_vals if x not in in_seq]
            if available:
                x = random.choice(available)
                # Find insertion pos
                pos = 0
                for i, v in enumerate(current_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                
                # Check incremental validity
                # New tails: x, x*current_seq[pos-1], ...
                new_tails = [x]
                p = x
                for k in range(pos - 1, -1, -1):
                    p *= current_seq[k]
                    new_tails.append(p)
                
                if not current_prods.intersection(new_tails):
                    new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                    new_prods = current_prods.copy()
                    new_prods.update(new_tails)
                    new_score = get_score(new_seq, new_prods)
        
        elif move_type == 'remove':
            if current_seq:
                idx = random.randint(0, len(current_seq) - 1)
                new_seq = current_seq[:idx] + current_seq[idx+1:]
                # Must recompute products for removal
                new_prods = compute_all_products(new_seq)
                new_score = get_score(new_seq, new_prods)
        
        elif move_type == 'swap':
            if current_seq:
                in_seq = set(current_seq)
                available = [x for x in all_vals if x not in in_seq]
                if available:
                    idx_remove = random.randint(0, len(current_seq) - 1)
                    x_add = random.choice(available)
                    
                    # Construct new seq
                    new_seq_set = set(current_seq)
                    new_seq_set.remove(current_seq[idx_remove])
                    new_seq_set.add(x_add)
                    new_seq = sorted(new_seq_set)
                    
                    new_prods = compute_all_products(new_seq)
                    new_score = get_score(new_seq, new_prods)
        
        if new_seq is not None:
            delta = new_score - current_score
            # Accept if better or with prob exp(delta/T)
            if delta >= 0 or (T > 0 and random.random() < np.exp(delta / T)):
                current_seq = new_seq
                current_prods = new_prods
                current_score = new_score
                
                # Check for new best
                if current_score > best_score:
                    best_score = current_score
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    best_prods = current_prods.copy()
                    no_improve = 0
                    # Reset temp if we found a new best valid solution
                    if len(current_prods) == len(current_seq)*(len(current_seq)+1)//2:
                        T = max(T, 5.0)
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        # Cooling
        T *= cool_rate
        if T < 0.01:
            T = 0.01
            
        # Restart if stuck
        if no_improve > 1000:
            no_improve = 0
            T = best_T * 0.5
            # Perturb
            if current_seq:
                # Remove 2 random elements to break local structure
                for _ in range(min(2, len(current_seq))):
                    if current_seq:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq = current_seq[:idx] + current_seq[idx+1:]
                current_prods = compute_all_products(current_seq)
                current_score = get_score(current_seq, current_prods)
                
    # Final polish: try to add any missing elements that don't create conflicts
    in_best = set(best_seq)
    prods_best = compute_all_products(best_seq) # Ensure fresh
    changed = True
    while changed:
        changed = False
        for x in range(2, 129):
            if x not in in_best:
                pos = 0
                for i, v in enumerate(best_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                new_tails = [x]
                p = x
                for k in range(pos - 1, -1, -1):
                    p *= best_seq[k]
                    new_tails.append(p)
                
                if not prods_best.intersection(new_tails):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    prods_best.update(new_tails)
                    changed = True
                    break # Restart loop as indices changed
    
    return best_seq
