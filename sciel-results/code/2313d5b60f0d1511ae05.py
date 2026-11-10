"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate initial code using Direct Sum of Tetracode
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct initial code
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    if q != 3:
        # Fallback for non-ternary or different n: simple greedy or direct sum
        # For now, just return initial code if q!=3, as optimization is tuned for q=3, n=6
        return initial_code

    # 2. Optimize using Greedy Removal and Pairwise Swapping
    
    # Map words to integers for fast indexing
    def word_to_int(w):
        val = 0
        for i in range(n):
            val += w[i] * (3 ** i)
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[i] = val % 3
            val //= 3
        return w

    # Precompute balls for all words in the space? 
    # 3^6 = 729 words. Ball size 13.
    # This is very fast and allows O(1) ball lookup.
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute ball for each word index
    balls = []
    for w in all_words:
        w_int = word_to_int(w)
        ball = [w_int]
        for pos in range(n):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(word_to_int(new_w))
        balls.append(ball)

    size_space = 3 ** n
    coverage = [0] * size_space
    code = [list(c) for c in initial_code]
    
    # Initialize coverage
    for cw in code:
        cw_int = word_to_int(cw)
        for w in balls[cw_int]:
            coverage[w] += 1

    # Helper to get ball of a word (list)
    def get_ball(cw):
        return balls[word_to_int(cw)]

    # Helper to check if a word (int) is covered by a candidate word (int)
    # i.e., dist(candidate, word) <= 1
    # Since we have balls, we can just check if word is in balls[candidate]
    # But for speed in swap logic, we might need set membership.
    # Let's keep balls as lists for iteration, but maybe use sets for membership?
    # 729 * 13 is small. Sets are fine.
    ball_sets = [set(b) for b in balls]

    # Greedy Removal Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Single Removal
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
                
            cw = code[i]
            cw_int = word_to_int(cw)
            ball = balls[cw_int]
            
            # Check if removable
            removable = True
            for w in ball:
                if coverage[w] <= 1:
                    removable = False
                    break
            
            if removable:
                for w in ball:
                    coverage[w] -= 1
                code.pop(i)
                improved = True
                break # Restart loop after a change to ensure consistency

        if time.time() > deadline - 5.0:
            break

        # Phase 2: Pairwise Swapping
        # Try to replace two codewords with one
        found_swap = False
        n_code = len(code)
        for i in range(n_code):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, n_code):
                if time.time() > deadline - 5.0:
                    break
                
                c1 = code[i]
                c2 = code[j]
                c1_int = word_to_int(c1)
                c2_int = word_to_int(c2)
                
                # Unique coverage of c1 and c2
                # Words in ball(c1) with coverage 1
                u1 = []
                for w in balls[c1_int]:
                    if coverage[w] == 1:
                        u1.append(w)
                if not u1:
                    continue # c1 is redundant, should have been removed in Phase 1
                
                u2 = []
                for w in balls[c2_int]:
                    if coverage[w] == 1:
                        u2.append(w)
                if not u2:
                    continue # c2 is redundant, should have been removed in Phase 1
                
                target = set(u1).union(set(u2))
                
                # Find candidate w in Ball(c1) intersection Ball(c2)
                # such that Ball(w) covers target
                # Candidates are in ball_sets[c1_int] intersect ball_sets[c2_int]
                # Since balls are small, we can just iterate ball(c1) and check if in ball(c2)
                for w_cand_int in balls[c1_int]:
                    if w_cand_int not in ball_sets[c2_int]:
                        continue
                    
                    # Check if Ball(w_cand_int) covers target
                    covers = True
                    for t in target:
                        if t not in ball_sets[w_cand_int]:
                            covers = False
                            break
                    
                    if covers:
                        # Perform swap
                        # Remove c1, c2
                        for w in balls[c1_int]:
                            coverage[w] -= 1
                        for w in balls[c2_int]:
                            coverage[w] -= 1
                        
                        # Add w_cand
                        w_cand_word = int_to_word(w_cand_int)
                        for w in balls[w_cand_int]:
                            coverage[w] += 1
                        
                        # Update code list
                        # Remove j first (higher index) then i
                        code.pop(j)
                        code.pop(i)
                        code.append(w_cand_word)
                        
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
        if not found_swap:
            # No more improvements possible with this heuristic
            break

    return code

# EVOLVE-BLOCK-END
