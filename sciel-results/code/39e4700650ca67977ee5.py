import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def get_primes():
        """Return all primes in [2, 128]."""
        primes = []
        for i in range(2, 129):
            is_prime = True
            for j in range(2, int(i**0.5) + 1):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(i)
        return primes

    def greedy_insert(seq, candidates):
        """Try to insert all candidates into seq, maintaining validity."""
        seq = seq[:]
        candidates = candidates[:]
        random.shuffle(candidates)
        
        for x in candidates:
            # Try to find a position for x
            inserted = False
            n = len(seq)
            # Try positions from left to right, or random? 
            # Random positions might be better to avoid bias.
            positions = list(range(n + 1))
            random.shuffle(positions)
            
            for pos in positions:
                # Check if x fits in terms of sorted order
                # Since seq must be strictly increasing, x must be > seq[pos-1] and < seq[pos]
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < n and x >= seq[pos]:
                    continue
                
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    seq = new_seq
                    inserted = True
                    break
        
        return seq

    def local_search(seq, deadline):
        """Local search with swaps and insertions."""
        best_seq = seq[:]
        best_len = len(seq)
        in_seq = set(seq)
        missing = [x for x in range(2, 129) if x not in in_seq]
        
        # Pre-calculate valid positions for missing numbers? 
        # It's dynamic, so we check on the fly.
        
        no_improve = 0
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        while time.time() < deadline - 5:
            if no_improve > 5000:
                break
            
            move_type = random.random()
            new_seq = best_seq[:]
            improved = False
            
            if move_type < 0.4 and missing:
                # Try to insert a missing number
                x = random.choice(missing)
                n = len(new_seq)
                pos = random.randint(0, n)
                if pos > 0 and new_seq[pos-1] >= x:
                    continue
                if pos < n and x >= new_seq[pos]:
                    continue
                
                new_seq = new_seq[:pos] + [x] + new_seq[pos:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    best_len = len(new_seq)
                    in_seq.add(x)
                    missing.remove(x)
                    no_improve = 0
                else:
                    no_improve += 1
            
            elif move_type < 0.7 and len(best_seq) > 2:
                # Try to swap two elements
                i, j = random.sample(range(len(best_seq)), 2)
                if i > j:
                    i, j = j, i
                # Check if swap maintains sorted order
                if j > 0 and best_seq[j] <= best_seq[j-1]: # This check is for the new position
                    pass
                # Actually, just swap and check validity + sorted order
                new_seq = best_seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                
                # Check sorted order
                valid_sorted = True
                for k in range(len(new_seq) - 1):
                    if new_seq[k] >= new_seq[k+1]:
                        valid_sorted = False
                        break
                
                if valid_sorted and is_valid(new_seq):
                    # Swaps don't change length, but might allow future insertions
                    # We accept swaps if they don't reduce length (which they don't)
                    # To avoid oscillation, we only accept if it leads to a better "state" 
                    # or just accept to explore.
                    # For now, accept valid swaps.
                    best_seq = new_seq
                    no_improve = 0
                else:
                    no_improve += 1
            
            else:
                # Try to remove an element to allow a better insertion later?
                # Or just a random perturbation.
                if len(best_seq) > 1:
                    rm_idx = random.randint(0, len(best_seq) - 1)
                    new_seq = best_seq[:rm_idx] + best_seq[rm_idx+1:]
                    if is_valid(new_seq):
                        # We accept removals only if we are stuck? 
                        # Or we track the best length and revert if we drop too low.
                        # Let's just accept valid removals to explore.
                        best_seq = new_seq
                        best_len = len(best_seq)
                        in_seq = set(best_seq)
                        missing = [x for x in range(2, 129) if x not in in_seq]
                        no_improve += 1
                    else:
                        no_improve += 1
            
            T *= cooling

        return best_seq

    # Strategy:
    # 1. Start with primes.
    # 2. Greedily insert composites.
    # 3. Local search to improve.
    # 4. Repeat with different random seeds/orderings.

    primes = get_primes()
    composites = [x for x in range(2, 129) if x not in primes]
    
    best_overall = []
    best_len = 0

    # Run multiple trials
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        random.seed(trial * 123 + 456)
        
        # Initialization: Primes
        current_seq = primes[:]
        
        # Greedy insertion of composites
        # Shuffle composites to vary the order
        comp_copy = composites[:]
        random.shuffle(comp_copy)
        
        current_seq = greedy_insert(current_seq, comp_copy)
        
        # Local search
        current_seq = local_search(current_seq, deadline)
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_overall = current_seq[:]

    # Final check
    if not is_valid(best_overall):
        # Fallback to a known valid sequence (e.g., primes only)
        best_overall = primes[:]

    return best_overall

# EVOLVE-BLOCK-END
