import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    N = q ** n
    
    # Precompute balls
    # Map int index to tuple
    def int_to_word(idx):
        return [idx // (q ** (n - i - 1)) % q for i in range(n)]
    
    def word_to_int(w):
        res = 0
        for i in range(n):
            res = res * q + w[i]
        return res

    # Precompute balls for all words
    balls = []
    for idx in range(N):
        w = int_to_word(idx)
        ball = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[i] = (new_w[i] + delta) % q
                ball.append(word_to_int(new_w))
        balls.append(ball)
    
    # Precompute reverse coverage: for each position u, which centers cover it
    # Since distance is symmetric, coverage[u] = [v for v in range(N) if u in balls[v]]
    # But balls[u] already gives us the neighbors of u, which is the same set.
    # So balls[u] is both the ball around u AND the set of centers that cover u.
    
    best_code = None
    best_len = N # Upper bound
    
    def greedy_set_cover():
        """Run one greedy set cover with random tie-breaking."""
        uncovered = [True] * N
        counts = [len(balls[i]) for i in range(N)]
        code = []
        
        while True:
            # Check if all covered
            remaining = sum(1 for u in range(N) if uncovered[u])
            if remaining == 0:
                break
            
            # Find index with max count among those that still cover uncovered elements
            max_count = -1
            max_indices = []
            for i in range(N):
                if counts[i] > max_count:
                    max_count = counts[i]
                    max_indices = [i]
                elif counts[i] == max_count and max_count > 0:
                    max_indices.append(i)
            
            if max_count == 0:
                # Should not happen if uncovered exists and logic is correct
                break
                
            # Random tie break
            w = random.choice(max_indices)
            code.append(w)
            
            # Update uncovered and counts
            for u in balls[w]:
                if uncovered[u]:
                    uncovered[u] = False
                    # Decrement counts for all centers covering u
                    for v in balls[u]:
                        counts[v] -= 1
        
        return code
    
    def local_search(code):
        """Try to remove elements from code while maintaining coverage."""
        code_set = set(code)
        # Build coverage: for each element u, how many centers in code cover it
        cover_count = [0] * N
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
        
        improved = True
        while improved:
            improved = False
            # Try removing each element in reverse order (arbitrary)
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                # Check if removing w still covers everything
                can_remove = True
                for u in balls[w]:
                    if cover_count[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove w
                    for u in balls[w]:
                        cover_count[u] -= 1
                    code.pop(i)
                    improved = True
                    break  # Restart after removal
        
        return code
    
    # Run multiple random restarts
    while time.time() < deadline:
        code = greedy_set_cover()
        
        # Local search: try to remove redundant codewords
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
            # Early termination if we hit the known optimum
            if best_len <= 73:
                # Keep going a bit to see if we can do better
                pass
    
    if best_code is None:
        # Fallback if something went wrong
        return [int_to_word(i) for i in range(N)]
        
    return [int_to_word(w) for w in best_code]
