import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Parameters for the specific task
    # q=3, n=7, radius=1
    total_words = q ** n
    
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute balls
    # ball[i] is an integer where bit j is 1 if word j is within distance radius of word i
    balls = [0] * total_words
    
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Compute Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    # For each word, find all words within distance radius
    for i in range(total_words):
        wi = all_words[i]
        mask = 0
        for j in range(total_words):
            if hamming_dist(wi, all_words[j]) <= radius:
                mask |= (1 << j)
        balls[i] = mask
        
    # Helper to check if a set of codewords covers the space
    def is_covering(code_indices):
        covered = 0
        for c in code_indices:
            covered |= balls[c]
        return covered == (1 << total_words) - 1
        
    # Helper to try to remove redundant codewords
    def simplify_code(code_indices):
        code_set = set(code_indices)
        changed = True
        while changed:
            changed = False
            # We need to know for each word, how many codewords cover it
            # Or simply, for each candidate removal, check if it's essential
            # Essential if there is a word w such that balls[c] covers w and no other c' in code covers w.
            # This is equivalent to: (balls[c] & ~union_of_others) != 0
            # But union_of_others is expensive to compute from scratch each time.
            # Instead, maintain cover counts.
            
            # Let's use a simpler approach for small sizes:
            # Try removing each one. If valid, remove it.
            # This is O(|code|^2 * bitops). |code| ~ 200. 200^2 = 40000. Fast.
            for i in range(len(code_indices)):
                c = code_indices[i]
                # Check if c is redundant
                # Create mask of all other codewords
                others_mask = 0
                for j in range(len(code_indices)):
                    if j != i:
                        others_mask |= balls[code_indices[j]]
                # If c covers any word not covered by others, it's essential
                if (balls[c] & ~others_mask) != 0:
                    continue
                # Otherwise, remove it
                code_indices.pop(i)
                changed = True
                break # Restart loop as indices changed
        return code_indices

    best_code = None
    best_size = total_words + 1
    
    # 1. Try the seed construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_code.append(word_to_idx(w))
            
    # Simplify seed
    simplified_seed = simplify_code(seed_code)
    if len(simplified_seed) < best_size:
        best_size = len(simplified_seed)
        best_code = simplified_seed
        
    if time.time() > deadline:
        return [idx_to_word(c) for c in best_code]
        
    # 2. Randomized Greedy from scratch
    random.seed(42)
    
    def greedy_randomized():
        uncovered = (1 << total_words) - 1
        code = []
        while uncovered:
            best_score = 0
            best_candidates = []
            for c in range(total_words):
                s = (balls[c] & uncovered).bit_count()
                if s > best_score:
                    best_score = s
                    best_candidates = [c]
                elif s == best_score and s > 0:
                    best_candidates.append(c)
            if not best_candidates:
                break # Should not happen if space is connected
            c = random.choice(best_candidates)
            code.append(c)
            uncovered &= ~balls[c]
        return code

    runs = 0
    while time.time() < deadline:
        runs += 1
        code = greedy_randomized()
        # Simplify the greedy result
        simplified = simplify_code(code)
        if len(simplified) < best_size:
            best_size = len(simplified)
            best_code = simplified
            
    return [idx_to_word(c) for c in best_code]
