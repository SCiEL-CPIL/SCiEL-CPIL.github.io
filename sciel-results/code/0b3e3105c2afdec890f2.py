import time
import random
import sys

# Primes up to 128
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
NUM_PRIMES = len(PRIMES)

def get_prime_vector(n):
    """Returns a tuple of exponents for n with respect to PRIMES."""
    vec = [0] * NUM_PRIMES
    temp = n
    for i, p in enumerate(PRIMES):
        if p * p > temp:
            break
        while temp % p == 0:
            vec[i] += 1
            temp //= p
    if temp > 1:
        # temp must be a prime in our list since n <= 128
        for i, p in enumerate(PRIMES):
            if p == temp:
                vec[i] += 1
                break
    return tuple(vec)

# Precompute prime vectors for all numbers 2..128
PRIME_VECTORS = {i: get_prime_vector(i) for i in range(2, 129)}

class Solver:
    def __init__(self, deadline):
        self.deadline = deadline
        self.best_len = 0
        self.best_seq = []
        self.nodes_explored = 0
        self.timeout = False

    def check_conflict_fast(self, seq, x):
        """
        Check if adding x to the end of seq creates a duplicate block product.
        We only need to check the new blocks ending at x.
        These blocks are: [x], [seq[-1], x], [seq[-2], seq[-1], x], ...
        
        We maintain a set of all existing block product vectors.
        However, passing the set is expensive. 
        Alternative: 
        For a fixed sequence `seq`, the set of all block products is fixed.
        When we add `x`, the new block products are the suffix products of `seq + [x]`.
        Let `suffix_prods` be the list of prime vectors for blocks ending at x.
        We need to check if any of these are already in the global set of products.
        
        To make this fast in recursion, we should pass the set of products.
        But sets of tuples are heavy.
        
        Optimization: 
        Since the sequence is strictly increasing and we are building it, 
        we can precompute the "prefix" sums of prime vectors? No, it's suffixes.
        
        Let's stick to the set of product vectors.
        """
        # Compute new product vectors for blocks ending at x
        new_prods = []
        current_vec = [0] * NUM_PRIMES
        # Add x
        x_vec = PRIME_VECTORS[x]
        for i in range(NUM_PRIMES):
            current_vec[i] += x_vec[i]
        new_prods.append(tuple(current_vec))
        
        # Multiply by previous elements in reverse
        for a in reversed(seq):
            a_vec = PRIME_VECTORS[a]
            for i in range(NUM_PRIMES):
                current_vec[i] += a_vec[i]
            new_prods.append(tuple(current_vec))
            
        return new_prods

    def search(self, seq, products_set):
        if self.timeout:
            return
        
        self.nodes_explored += 1
        if self.nodes_explored % 5000 == 0:
            if time.time() > self.deadline:
                self.timeout = True
                return

        current_len = len(seq)
        
        # Update best
        if current_len > self.best_len:
            self.best_len = current_len
            self.best_seq = seq.copy()
        
        # Pruning
        if seq:
            max_possible = 128 - seq[-1]
        else:
            max_possible = 127
            
        if current_len + max_possible <= self.best_len:
            return

        # Candidates
        start_val = seq[-1] + 1 if seq else 2
        
        # Heuristic: Try smaller numbers first to allow more room? 
        # Or try numbers that are "compatible"?
        # Let's iterate normally.
        for x in range(start_val, 129):
            if self.timeout:
                return
            
            new_prods = self.check_conflict_fast(seq, x)
            
            # Check for conflicts with existing products
            conflict = False
            for p in new_prods:
                if p in products_set:
                    conflict = True
                    break
            
            if not conflict:
                # Add new products to set
                new_products_set = products_set.union(new_prods)
                new_seq = seq + [x]
                self.search(new_seq, new_products_set)
                
                if self.timeout:
                    return

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    solver = Solver(deadline)
    
    # 1. Initial Greedy (Ascending)
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        new_prods = solver.check_conflict_fast(greedy_seq, x)
        conflict = False
        for p in new_prods:
            if p in greedy_prods:
                conflict = True
                break
        if not conflict:
            greedy_seq.append(x)
            greedy_prods.update(new_prods)
            
    if len(greedy_seq) > solver.best_len:
        solver.best_len = len(greedy_seq)
        solver.best_seq = greedy_seq.copy()

    # 2. Randomized Greedy Exploration
    # Try to find a good initial solution to help pruning
    for attempt in range(500):
        if time.time() > deadline - 10.0: # Leave 10s for search
            break
            
        candidates = list(range(2, 129))
        # Bias towards smaller numbers? Or just shuffle?
        # Shuffling breaks the "increasing" constraint if we pick out of order.
        # We must pick an increasing subsequence.
        # Strategy: Pick a random subset, sort it, then greedily add.
        # Or: Iterate through sorted candidates, but randomly skip some?
        
        # Better: Randomized Greedy on sorted list
        # We can't just shuffle because we need strictly increasing order.
        # We can decide probabilistically whether to include each number in order.
        # But that's just one path.
        
        # Alternative: Randomly select a "start" and "end" range?
        # Let's try: Randomly permute the "decision to skip or take" for a subset?
        
        # Simpler Randomized Approach:
        # Pick a random increasing subsequence as a "seed" and try to extend?
        # Or just run the greedy but with random skips.
        
        seq = []
        prods = set()
        for x in range(2, 129):
            # Randomly skip with probability 0.1?
            if random.random() < 0.1:
                continue
            new_prods = solver.check_conflict_fast(seq, x)
            conflict = False
            for p in new_prods:
                if p in prods:
                    conflict = True
                    break
            if not conflict:
                seq.append(x)
                prods.update(new_prods)
                
        if len(seq) > solver.best_len:
            solver.best_len = len(seq)
            solver.best_seq = seq.copy()

    # 3. Local Search / Insertion
    # Try to insert missing numbers into the best sequence found so far
    best_seq = solver.best_seq
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
            
        missing = [x for x in range(2, 129) if x not in best_seq]
        if not missing:
            break
            
        random.shuffle(missing)
        
        # Try to insert each missing number into the current best_seq
        # We check all positions? No, sequence must be strictly increasing.
        # So we can only insert x if it fits between two existing elements.
        # Actually, we can just rebuild the sequence by iterating through all numbers 2..128
        # and deciding whether to include them, starting from the current set.
        
        # Better: Try to add one missing number at a time to the best_seq
        # Check if adding x to best_seq (in correct sorted position) is valid.
        
        # To add x to seq, we need to check all blocks that include x.
        # This is complex if x is in the middle.
        # Easier: Try to form a new sequence by taking best_seq + [x] and sorting, then checking validity.
        # This is O(N^2 * P) per check. N is ~50-100. 100^2 * 30 = 300,000 ops. Acceptable.
        
        def is_valid_full(seq):
            if not seq: return True
            prods = set()
            n = len(seq)
            for i in range(n):
                vec = [0] * NUM_PRIMES
                for j in range(i, n):
                    v_j = PRIME_VECTORS[seq[j]]
                    for k in range(NUM_PRIMES):
                        vec[k] += v_j[k]
                    t = tuple(vec)
                    if t in prods:
                        return False
                    prods.add(t)
            return True

        improved = False
        for x in missing:
            if time.time() > deadline - 5.0:
                break
            # Try to insert x into best_seq
            # Find position
            new_seq = sorted(best_seq + [x])
            if is_valid_full(new_seq):
                best_seq = new_seq
                improved = True
                if len(best_seq) > solver.best_len:
                    solver.best_len = len(best_seq)
                    solver.best_seq = best_seq.copy()
                # If we improved, we might want to restart the missing list?
                # Or just continue with the next missing number.
                # Since we sorted, the indices of other missing numbers might change?
                # No, we are just checking if x can be added.
                # If we add x, the set of missing numbers changes.
                # Let's just break and restart the loop if we made an improvement.
                break
                
        if not improved:
            # If no single insertion works, try random swaps? Or just stop.
            # For now, if no insertion works, we might be stuck.
            # We could try removing one element and inserting two? Too complex.
            # Let's just stop if no improvement in a few rounds?
            # Or try a different random order.
            pass

    # 4. Backtracking Search
    # Use the best found so far as a lower bound for pruning.
    # Start search from empty or from a prefix of the best solution?
    # Starting from empty is safer but slower.
    # Let's start from empty but with a tight deadline.
    
    # If we have a good solution, we can try to extend it?
    # The search function adds to the end.
    # We can call search on the prefix of the best solution?
    # No, the best solution might not be a prefix of the optimal one.
    
    # Let's just run the search from empty with the current best_len for pruning.
    # This might be slow if best_len is high.
    # If best_len is e.g. 80, max_possible is small.
    
    solver.search([], set())
    
    return solver.best_seq

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
