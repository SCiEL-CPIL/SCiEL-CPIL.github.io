import random
import time

def construct():
    # Set random seed for reproducibility in debugging, but for evaluation, we want exploration.
    # However, deterministic output is often preferred unless specified. 
    # The task says "Maximize length". 
    # We will use a fixed seed for the random generator to ensure consistent results if the evaluator runs multiple times, 
    # or just let it be random. Given "search score", consistent high performance is key.
    random.seed(42)
    
    def is_valid(seq):
        """Check if all consecutive subarray products are distinct."""
        n = len(seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def get_score(seq):
        """Score is the length of the sequence."""
        return len(seq)

    def try_insert(seq, x):
        """Try to insert x into seq at any position. Return new seq if valid, else None."""
        n = len(seq)
        # Try inserting at each position
        for i in range(n + 1):
            # Check if x fits in sorted order? 
            # The problem statement says "Return strictly increasing integers".
            # So the sequence MUST be strictly increasing.
            # This restricts insertion positions significantly.
            # x must be > seq[i-1] (if i>0) and < seq[i] (if i<n).
            
            if i > 0 and x <= seq[i-1]:
                continue
            if i < n and x >= seq[i]:
                continue
                
            new_seq = seq[:i] + [x] + seq[i:]
            if is_valid(new_seq):
                return new_seq
        return None

    def try_swap(seq, i, j):
        """Swap elements at i and j. Check if result is strictly increasing and valid."""
        if i == j:
            return None
        # Check strictly increasing constraint after swap
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        # Check strictly increasing
        for k in range(len(new_seq)-1):
            if new_seq[k] >= new_seq[k+1]:
                return None
                
        if is_valid(new_seq):
            return new_seq
        return None

    def try_move(seq, i, j):
        """Move element at i to position j. Check if result is strictly increasing and valid."""
        if i == j:
            return None
        x = seq[i]
        new_seq = seq[:i] + seq[i+1:]
        # Adjust j if i < j
        if i < j:
            j -= 1
        # Insert x at j
        # Check strictly increasing constraint
        if j > 0 and x <= new_seq[j-1]:
            return None
        if j < len(new_seq) and x >= new_seq[j]:
            return None
            
        new_seq = new_seq[:j] + [x] + new_seq[j:]
        
        if is_valid(new_seq):
            return new_seq
        return None

    # Phase 1: Greedy Initialization
    # Start with a small set and grow it.
    # A good strategy is to try to add the smallest possible numbers first, 
    # but allow backtracking if we get stuck? 
    # For now, simple greedy: iterate 2..128, add if possible.
    # This might get stuck in a local optimum.
    
    best_seq = []
    for x in range(2, 129):
        if try_insert(best_seq, x) is not None:
            # We need to update best_seq
            # try_insert returns the new seq or None
            new_seq = try_insert(best_seq, x)
            if new_seq:
                best_seq = new_seq
                
    # Phase 2: Local Search with Hill Climbing
    # We will try to improve the length by inserting, or rearranging to allow future insertions.
    # Since we can only insert to increase length, we focus on insertions.
    # But we might need to rearrange (swap/move) to create a "gap" or valid configuration for an insertion.
    # However, swaps/moves don't change the set of numbers, just the order.
    # Does order matter for validity? Yes, products depend on order.
    # Does order matter for "strictly increasing"? Yes, the values must be increasing.
    # So the sequence is determined by the SET of numbers? 
    # NO. The sequence must be strictly increasing. 
    # If we have a set of numbers {2, 3, 5}, the ONLY strictly increasing sequence is [2, 3, 5].
    # Therefore, the "order" is fixed by the set!
    # 
    # WAIT.
    # "Return strictly increasing integers".
    # This means if I choose a set S, the sequence is uniquely determined: sorted(S).
    # Therefore, "swapping" or "moving" is USELESS if the values are distinct and the sequence must be sorted.
    # Swapping 2 and 3 in [2, 3, 5] gives [3, 2, 5] which is NOT strictly increasing.
    # So, the search space is simply the SUBSETS of {2, ..., 128}.
    # The validity depends only on the subset (because the order is fixed as sorted).
    #
    # This simplifies the problem drastically!
    # We don't need swaps or moves. We just need to find the largest subset S of {2..128} such that
    # the sequence sorted(S) has distinct consecutive subarray products.
    #
    # The current seed does this: it greedily adds numbers.
    # My "new approach" should be a different way to find the optimal subset.
    # Since the order is fixed, the "local search" neighborhood is:
    # 1. Add an element (Insert)
    # 2. Remove an element (Delete) - useful to escape local optima where a small number blocks many others?
    #    Actually, if we want to MAXIMIZE length, we rarely want to remove unless it allows adding TWO others?
    #    Or if the current set is stuck, maybe removing one allows adding two?
    #    Let's check if removing helps.
    #    If we have a set S, and no x in [2..128]\S can be added to S to form a valid sorted sequence.
    #    Can we remove y from S, and then add x1 and x2?
    #    This is a "2-for-1" trade.
    #
    # So, the new approach: **Subset Optimization with Add and Delete-Add-Add moves**.
    # Or simply **Simulated Annealing** on the subset.
    
    def evaluate_subset(subset):
        # Convert to sorted list
        seq = sorted(subset)
        return is_valid(seq)

    # Let's restart with a smarter initialization or just use the greedy result as a seed.
    # We will use Random Restart Hill Climbing on the SET of elements.
    
    def random_restart_hc():
        # Start with a random valid subset or the greedy one
        current = list(best_seq)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        # Try to improve
        no_improve_count = 0
        max_iters = 2000
        
        for _ in range(max_iters):
            # Generate a neighbor
            # Option 1: Add a random unused number
            unused = [x for x in range(2, 129) if x not in current_set]
            if unused:
                if random.random() < 0.7: # Bias towards adding
                    x = random.choice(unused)
                    new_set = current_set | {x}
                    if evaluate_subset(new_set):
                        current = sorted(new_set)
                        current_set = new_set
                        if len(current) > best_len:
                            best = current[:]
                            best_len = len(best)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    # Option 2: Remove a random used number, then try to add one or two others
                    if current:
                        y = random.choice(current)
                        new_set = current_set - {y}
                        # Try to add back y? No, that's a revert.
                        # Try to add a random other number
                        unused_after_rem = [x for x in range(2, 129) if x not in new_set]
                        if unused_after_rem:
                            # Try adding one random
                            x1 = random.choice(unused_after_rem)
                            cand1 = new_set | {x1}
                            if evaluate_subset(cand1):
                                current = sorted(cand1)
                                current_set = cand1
                                if len(current) > best_len:
                                    best = current[:]
                                    best_len = len(best)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                            else:
                                # Try adding two? (Expensive, maybe skip for now)
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
            
            if no_improve_count > 100:
                break
        
        return best

    # Run multiple restarts
    final_best = best_seq[:]
    for i in range(20):
        candidate = random_restart_hc()
        if len(candidate) > len(final_best):
            final_best = candidate
            
    return final_best
