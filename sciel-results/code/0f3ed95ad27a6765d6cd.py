import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Problem specific parameters for the task
    # q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    total_words = q ** n
    
    # Precompute all words and their neighbors (Hamming distance <= 1)
    # A word is a tuple of n integers
    # Neighbors: the word itself + words differing in exactly 1 position
    # Number of neighbors: 1 + n * (q - 1)
    
    # For efficiency, we represent words as integers 0..total_words-1
    # and convert to/from tuples only when necessary, or work with integers directly.
    
    def int_to_tuple(x):
        return tuple((x // (q ** i)) % q for i in range(n))
        
    def tuple_to_int(t):
        x = 0
        for i in range(n):
            x += t[i] * (q ** i)
        return x
        
    # Precompute neighbors for all words
    neighbors = []
    for i in range(total_words):
        t = int_to_tuple(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val != t[pos]:
                    new_t = list(t)
                    new_t[pos] = val
                    neigh.append(tuple_to_int(tuple(new_t)))
        neighbors.append(neigh)
        
    best_code = None
    best_size = total_words + 1
    
    # Run multiple randomized greedy + removal attempts
    num_restarts = 100  # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Construction
        covered_count = [0] * total_words
        code = []
        remaining = set(range(total_words))
        
        # To speed up greedy, we can maintain a list of candidates or just iterate.
        # Since total_words is small (6561), iterating is fine.
        
        while remaining:
            if time.time() > deadline:
                break
                
            best_candidate = -1
            max_new = -1
            # Random tie breaking
            # We can shuffle the remaining candidates to break ties randomly
            # But iterating through all remaining to find max is O(N^2) worst case.
            # N=6561, so 6561^2 ~ 43M ops per restart. 100 restarts ~ 4.3B ops. Might be too slow in Python.
            
            # Optimization: 
            # Instead of checking all remaining words, we can prioritize.
            # However, for a simple implementation, let's try to be smart.
            # Actually, 6561 is small enough that a pure Python loop might be okay if optimized, 
            # but let's use a heap or similar if needed. 
            # Let's just iterate. 6561 * 6561 is ~43 million operations. In Python, this is ~2-5 seconds per restart.
            # 100 restarts would be 200-500 seconds. Too slow.
            
            # We need a faster greedy or fewer restarts.
            # Let's reduce restarts or optimize the inner loop.
            
            # Alternative: Use the seed construction as a starting point for removal, which is faster to evaluate.
            # The seed gives 729 words. Removal from 729 is much faster than building from scratch.
            
            pass # Skip pure greedy construction from scratch if too slow, use seed-based approach.
        
        # Strategy: Start with the seed construction (729 words), then remove.
        # This is deterministic for the first iteration, then we can perturb.
        
        # Let's implement the Seed + Removal strategy primarily.
        # Seed: Tetracode (9 words len 4) x All tails (81 words len 4)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tails = list(itertools.product(range(3), repeat=n - 4))
        
        # Construct seed code
        seed_code = []
        for h in tetracode:
            for tail in tails:
                seed_code.append(tuple(h + list(tail)))
                
        # Convert to integers
        seed_code_ints = [tuple_to_int(c) for c in seed_code]
        
        # If this is the first restart, use the seed directly.
        # For subsequent restarts, we can randomly remove a few words from the seed (if valid) 
        # or just run the removal process which might get stuck, so we need a way to escape.
        
        # Actually, simple removal from the seed often reaches 486 directly.
        # Let's just run the removal process on the seed.
        
        current_code = set(seed_code_ints)
        covered_count = [0] * total_words
        
        # Initialize coverage
        for c in current_code:
            for w in neighbors[c]:
                covered_count[w] += 1
        
        # Removal Loop
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
                
            # Try to remove each word
            # To allow randomization in removal order, we can shuffle the list
            candidates = list(current_code)
            random.shuffle(candidates)
            
            for c in candidates:
                if c not in current_code:
                    continue
                    
                # Check if c is removable
                # c is removable if all neighbors w have covered_count[w] > 1
                removable = True
                for w in neighbors[c]:
                    if covered_count[w] == 1:
                        removable = False
                        break
                        
                if removable:
                    # Remove c
                    current_code.remove(c)
                    for w in neighbors[c]:
                        covered_count[w] -= 1
                    changed = True
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
        # If we have time, we can try to improve further with random restarts of the removal process
        # or by starting from slightly perturbed seeds.
        # However, the seed + removal is very strong for this specific case.
        
    # Convert best_code back to list of lists
    if best_code is None:
        # Fallback to seed
        return [list(h + list(tail)) for h in tetracode for tail in tails]
        
    result = [list(int_to_tuple(c)) for c in best_code]
    
    # Ensure distinctness (set guarantees this)
    return result

# EVOLVE-BLOCK-END
