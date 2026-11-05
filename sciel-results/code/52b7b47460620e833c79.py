import itertools
import random
import time

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # The evaluator passes these, but the seed hardcodes the logic for 3,7,1.
    # We should respect the passed q, n, radius if possible, but the seed is specific.
    # The prompt says "the evaluator passes this task's alphabet size q, length n and covering radius".
    # The seed ignores q, n, radius and hardcodes.
    # I will implement a general solution for q=3, n=7, radius=1 as per the task description,
    # but I should check if the evaluator calls with different params.
    # "Interface: construct(q, n, radius) -> list of codewords"
    # "The evaluator passes this task's alphabet size q, length n and covering radius."
    # It implies the evaluator passes q=3, n=7, radius=1.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed if parameters are different, though unlikely
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Generate initial code (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    num_cw = len(base_codewords) # 243
    
    # 2. Precompute neighbors for each codeword
    # Total words = 3^7 = 2187
    # Convert word to index: sum(w[i] * 3**(n-1-i))
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % 3)
            idx //= 3
        return w[::-1]

    neighbors = []
    for cw in base_codewords:
        cw_idx = word_to_idx(cw)
        neigh_indices = {cw_idx}
        for i in range(n):
            for delta in [-1, 1]:
                new_val = (cw[i] + delta) % 3
                new_cw = cw[:i] + [new_val] + cw[i+1:]
                neigh_indices.add(word_to_idx(new_cw))
        neighbors.append(list(neigh_indices))
        
    # 3. Greedy Removal with Random Restarts
    best_count = num_cw
    best_active = [True] * num_cw
    
    # Pre-compute cover counts for initial state
    init_cover_count = [0] * (3**n)
    for i in range(num_cw):
        for idx in neighbors[i]:
            init_cover_count[idx] += 1
            
    # We will try many random orders
    rng = random.Random(42)
    
    while time.time() < deadline:
        active = [True] * num_cw
        cover_count = init_cover_count[:]
        
        # Create a list of indices and shuffle
        order = list(range(num_cw))
        rng.shuffle(order)
        
        for i in order:
            if not active[i]:
                continue
            
            # Check if codeword i can be removed
            can_remove = True
            for idx in neighbors[i]:
                if cover_count[idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                active[i] = False
                for idx in neighbors[i]:
                    cover_count[idx] -= 1
        
        current_count = sum(active)
        if current_count < best_count:
            best_count = current_count
            best_active = active[:]
            
            # Optional: If we hit a very good number, we could stop or tighten
            # But let's keep searching for better.
            
    # 4. Construct the final list of codewords
    result = []
    for i in range(num_cw):
        if best_active[i]:
            result.append(base_codewords[i])
            
    return result
